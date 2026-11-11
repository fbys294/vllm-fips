# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

from .base import (
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)

try:
    import av
except ImportError:
    av = PlaceholderModule("av")  # type: ignore[assignment]

logger = init_logger(__name__)


def decode_pyav(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    frame_recovery: bool = False,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    with PyAVVideoBackendMixin.open_video_container(data) as container:
        stream = container.streams.video[0]
        check_frame_pixel_limit(
            int(stream.width or 0),
            int(stream.height or 0),
        )
    source = loader_cls._prepare_source(PyAVVideoBackendMixin.get_video_metadata(data))
    frame_idx = loader_cls.compute_frames_index_to_sample(
        source=source, target=target, **sampling_kwargs
    )
    with PyAVVideoBackendMixin.open_video_container(data) as container:
        frames, valid = loader_cls.read_frames(
            container,
            frame_idx,
            total_frames_num=source.total_frames_num,
            frame_recovery=frame_recovery,
        )
    return frames, source, frame_idx, valid


class PyAVVideoBackendMixin:
    @classmethod
    def open_video_container(cls, data: bytes) -> Any:
        try:
            container = av.open(BytesIO(data))
        except Exception as e:
            raise ValueError("Could not open video stream") from e
        if not container.streams.video:
            container.close()
            raise ValueError("Could not open video stream: no video stream found")
        return container

    @classmethod
    def demux_count_frames(cls, data: bytes) -> int:
        """Count frames with PyAV's demuxer-only mode."""
        count = 0
        with cls.open_video_container(data) as container:
            stream = container.streams.video[0]
            try:
                for packet in container.demux(stream):
                    if packet.size > 0:
                        if packet.pts is None or packet.pts >= 0:
                            count += 1
            except Exception:
                pass
        if count == 0:
            raise ValueError("Could not determine the number of video frames")
        return count

    @staticmethod
    def _stream_fps(stream: Any) -> float:
        for rate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
            if rate is not None and float(rate) > 0:
                return float(rate)
        return 0.0

    @staticmethod
    def get_video_metadata(data: bytes) -> VideoSourceMetadata:
        with PyAVVideoBackendMixin.open_video_container(data) as container:
            stream = container.streams.video[0]
            total_frames_num = int(stream.frames or 0)
            original_fps = PyAVVideoBackendMixin._stream_fps(stream)

            # Check for edit-list trimmed videos (negative PTS lead-in packets)
            # or missing header frame count (streamed WebM/MKV, raw H.264).
            visible_count = 0
            has_negative_pts = False
            try:
                for packet in container.demux(stream):
                    if packet.size > 0:
                        if packet.pts is not None and packet.pts < 0:
                            has_negative_pts = True
                        else:
                            visible_count += 1
            except Exception:
                pass

            if total_frames_num <= 0:
                if visible_count <= 0:
                    raise ValueError("Could not determine the number of video frames")
                total_frames_num = visible_count
                duration = total_frames_num / original_fps if original_fps > 0 else 0.0
                return VideoSourceMetadata(total_frames_num, original_fps, duration)

            if has_negative_pts and 0 < visible_count < total_frames_num - 1:
                logger.warning(
                    "Video header claims %d frames but only %d are "
                    "presentable; sampling over the true frame count.",
                    total_frames_num,
                    visible_count,
                )
                total_frames_num = visible_count

            duration = total_frames_num / original_fps if original_fps > 0 else 0.0
            return VideoSourceMetadata(
                total_frames_num=total_frames_num,
                original_fps=original_fps,
                duration=duration,
            )

    @classmethod
    def _can_use_for_recovery(
        cls,
        idx: int,
        failed_frames: list[int],
        next_target_map: dict[int, int],
        total_frames: int,
    ) -> bool:
        """Check if current frame can recover the oldest failed frame."""
        if not failed_frames:
            return False
        oldest_failed = failed_frames[0]
        limit = next_target_map.get(oldest_failed, total_frames)
        return idx < limit

    @classmethod
    def _iter_decoded_frames(cls, container: Any, max_frame_idx: int):
        """Yield `(idx, rgb_ndarray | None)` for each packet up to `max_frame_idx`."""
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        idx = 0

        for packet in container.demux(stream):
            if idx > max_frame_idx:
                break
            is_flush = packet.size == 0
            try:
                decoded = packet.decode()
            except Exception:
                if not is_flush:
                    yield idx, None
                    idx += 1
                continue

            if not decoded:
                continue

            for frame in decoded:
                if idx > max_frame_idx:
                    break
                if frame.pts is not None and frame.pts < 0:
                    continue
                try:
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb is not None and rgb.size > 0:
                        yield idx, rgb
                    else:
                        yield idx, None
                except Exception:
                    yield idx, None
                idx += 1

    @classmethod
    def _read_frames_with_recovery(
        cls,
        container: Any,
        frame_indices: list[int],
        total_frames: int,
    ) -> tuple[npt.NDArray, list[int], dict[int, int]]:
        """Read frames with dynamic window forward-scan recovery.

        When a target frame fails to load, the next successfully decoded
        frame (before the next target frame) will be used to recover it.
        """
        stream = container.streams.video[0]
        width = int(stream.width or 0)
        height = int(stream.height or 0)

        assert width > 0 and height > 0, (
            f"Invalid video frame size: width={width}, height={height}"
        )

        frame_idx_set = set(frame_indices)
        max_frame_idx = frame_indices[-1] if frame_indices else 0

        next_target_map: dict[int, int] = {}
        for k in range(len(frame_indices) - 1):
            next_target_map[frame_indices[k]] = frame_indices[k + 1]
        if frame_indices:
            next_target_map[frame_indices[-1]] = total_frames

        frames_list: list[npt.NDArray] = []
        valid_frame_indices: list[int] = []
        failed_frames_idx: list[int] = []
        recovered_map: dict[int, int] = {}

        seen_indices: set[int] = set()
        for idx, rgb_frame in cls._iter_decoded_frames(container, max_frame_idx):
            seen_indices.add(idx)
            is_target_frame = idx in frame_idx_set

            if rgb_frame is None:
                if is_target_frame:
                    logger.debug(
                        "Failed to decode frame %d during video loading.",
                        idx,
                    )
                    failed_frames_idx.append(idx)
                continue

            can_recover = cls._can_use_for_recovery(
                idx, failed_frames_idx, next_target_map, total_frames
            )

            if is_target_frame or can_recover:
                frames_list.append(rgb_frame)
                valid_frame_indices.append(idx)

                if can_recover:
                    recovered_idx = failed_frames_idx.pop(0)
                    recovered_map[recovered_idx] = idx
                    logger.info(
                        "Recovered frame %d using frame %d (delay: %d)",
                        recovered_idx,
                        idx,
                        idx - recovered_idx,
                    )

        for target_idx in frame_indices:
            if target_idx not in seen_indices and target_idx <= max_frame_idx:
                failed_frames_idx.append(target_idx)

        for failed_idx in failed_frames_idx:
            logger.debug(
                "Frame %d could not be recovered (end of video).",
                failed_idx,
            )

        if frames_list:
            frames = np.stack(frames_list)
        else:
            frames = np.empty((0, height, width, 3), dtype=np.uint8)

        return frames, valid_frame_indices, recovered_map

    @classmethod
    def _read_frames_no_recovery(
        cls,
        container: Any,
        frame_indices: set[int],
        max_frame_idx: int,
    ) -> tuple[npt.NDArray, list[int]]:
        num_expected_frames = len(frame_indices)
        stream = container.streams.video[0]
        width = int(stream.width or 0)
        height = int(stream.height or 0)
        frames = np.empty((num_expected_frames, height, width, 3), dtype=np.uint8)

        i = 0
        valid_frame_indices = []
        for idx, rgb_frame in cls._iter_decoded_frames(container, max_frame_idx):
            if rgb_frame is None:
                if idx in frame_indices:
                    logger.debug(
                        "Failed to decode frame %d during video loading. "
                        "This frame will be skipped.",
                        idx,
                    )
                continue
            if idx in frame_indices:
                frames[i] = rgb_frame
                valid_frame_indices.append(idx)
                i += 1

        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_expected_frames:
            logger.warning(
                "Video loading completed with %d broken/unreadable frames. "
                "Expected %d frames but only loaded %d frames.",
                num_expected_frames - valid_num_frames,
                num_expected_frames,
                valid_num_frames,
            )

        return frames[:valid_num_frames], valid_frame_indices

    @classmethod
    def read_frames(
        cls,
        container: Any,
        frame_idx: list[int],
        total_frames_num: int,
        *,
        frame_recovery: bool = False,
    ) -> tuple[npt.NDArray, list[int]]:
        if frame_recovery:
            num_frames_to_sample = len(frame_idx)
            frames, valid_frame_indices, recovered_map = cls._read_frames_with_recovery(
                container, frame_idx, total_frames_num
            )

            if recovered_map:
                logger.info(
                    "Frame recovery: %d frames recovered using forward scan.",
                    len(recovered_map),
                )
        else:
            frame_idx_set = set(frame_idx)
            num_frames_to_sample = len(frame_idx_set)
            frames, valid_frame_indices = cls._read_frames_no_recovery(
                container, frame_idx_set, max(frame_idx) if frame_idx else -1
            )
        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_frames_to_sample:
            logger.warning(
                "Video loading completed with %d broken/unreadable frames. "
                "Expected to sample %d frames but only loaded %d frames.",
                num_frames_to_sample - valid_num_frames,
                num_frames_to_sample,
                valid_num_frames,
            )
        return frames, valid_frame_indices


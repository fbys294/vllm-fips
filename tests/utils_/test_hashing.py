# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import pickle

import pytest

from vllm.utils.hashing import safe_hash, sha256


@pytest.mark.parametrize("input", [(), ("abc",), (None,), (None, bool, [1, 2, 3])])
def test_sha256(input: tuple):
    digest = sha256(input)
    assert digest is not None
    assert isinstance(digest, bytes)
    assert digest != b""

    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    assert digest == hashlib.sha256(input_bytes).digest()

    # hashing again, returns the same value
    assert digest == sha256(input)

    # hashing different input, returns different value
    assert digest != sha256(input + (1,))


@pytest.mark.parametrize("usedforsecurity", [True, False])
def test_safe_hash_uses_sha256(usedforsecurity: bool):
    data = b"fips-140-3-compliant-data"
    h = safe_hash(data, usedforsecurity=usedforsecurity)
    assert h.name == "sha256"
    assert h.hexdigest() == hashlib.sha256(data).hexdigest()

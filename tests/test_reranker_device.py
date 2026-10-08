"""The device check of the scoring stage, tested without torch.

``check_device`` takes the torch module as an argument, so a namespace with
the two ``is_available`` functions stands in for it here; the CPU test suite
never imports torch.
"""

from types import SimpleNamespace

import pytest

from cruxes.reranker.model import check_device


def fake_torch(cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 2),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_cuda_needs_a_cuda_device():
    with pytest.raises(RuntimeError, match="CUDA"):
        check_device("cuda", fake_torch(cuda=False, mps=True))
    check_device("cuda", fake_torch(cuda=True, mps=False))
    check_device("cuda:1", fake_torch(cuda=True, mps=False))


def test_mps_needs_an_mps_device():
    with pytest.raises(RuntimeError, match="mps"):
        check_device("mps", fake_torch(cuda=True, mps=False))
    check_device("mps", fake_torch(cuda=False, mps=True))


def test_cpu_is_allowed_with_a_warning(capsys):
    check_device("cpu", fake_torch(cuda=False, mps=False))
    assert "warning" in capsys.readouterr().err


def test_unknown_device_is_an_error():
    for device in ("tpu", "cudax", "cuda:", "cuda:abc", "cuda:-1", "cuda:0:1"):
        with pytest.raises(ValueError, match="unknown device"):
            check_device(device, fake_torch(cuda=True, mps=True))


def test_a_cuda_ordinal_past_the_device_count_is_an_error():
    with pytest.raises(ValueError, match="device_count"):
        check_device("cuda:7", fake_torch(cuda=True, mps=True))

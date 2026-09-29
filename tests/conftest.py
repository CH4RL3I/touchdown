import cv2
import pytest

cv2.setNumThreads(1)


@pytest.fixture(scope="session")
def assets():
    from landsim.closed_loop import get_assets

    return get_assets(True)

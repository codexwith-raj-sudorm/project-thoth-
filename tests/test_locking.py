from pathlib import Path

import pytest

from thoth.exceptions import ConcurrentRunError
from thoth.locking import SingleInstanceLock


def test_lock_excludes_and_cleans_up(tmp_path: Path):
    path = tmp_path / "lock"
    first = SingleInstanceLock(path).acquire()
    try:
        with pytest.raises(ConcurrentRunError):
            SingleInstanceLock(path).acquire()
    finally:
        first.release()
    assert not path.exists()

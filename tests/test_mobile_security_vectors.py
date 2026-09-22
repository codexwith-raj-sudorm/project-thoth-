"""Curated mobile threat patterns required by FR-11 and the hardening plan."""

import pytest

from thoth.exceptions import SecurityViolationException
from thoth.security import scan_code


VECTORS = [
    "import ctypes",
    "from pty import spawn",
    "import code",
    "import codeop",
    "import pickle as serializer",
    "from multiprocessing import Process",
    "import builtins",
    "eval('40 + 2')",
    "exec('pass')",
    "__import__('os')",
    "compile('pass', '<x>', 'exec')",
    "globals()",
    "locals()",
    "import os",
    "from pathlib import Path",
    "import socket",
    "from urllib.request import urlopen",
    "open('/data/data/com.termux/files/home/.profile')",
    "open('../outside.txt', 'w')",
    "getattr(object, '__subclasses__')",
]


@pytest.mark.parametrize("code", VECTORS, ids=[f"vector-{index:02d}" for index in range(1, 21)])
def test_mobile_adversarial_vector_is_blocked(code: str):
    with pytest.raises(SecurityViolationException) as error:
        scan_code(code)
    assert "line" in str(error.value)

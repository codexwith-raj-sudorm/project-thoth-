import pytest

from thoth.exceptions import SecurityViolationException
from thoth.security import count_tautological_assertions, scan_code


@pytest.mark.parametrize("code", [
    "import pickle", "from ctypes import CDLL", "eval('1')", "thing.exec('x')",
    "import builtins", "compile('x', 'x', 'exec')",
])
def test_tier_zero_is_always_blocked(code):
    with pytest.raises(SecurityViolationException):
        scan_code(code, allow_filesystem=True, allow_network=True)


def test_tier_one_permissions():
    with pytest.raises(SecurityViolationException, match="filesystem"):
        scan_code("import pathlib")
    assert scan_code("import pathlib", allow_filesystem=True).passed
    with pytest.raises(SecurityViolationException, match="network"):
        scan_code("from urllib.parse import quote")
    assert scan_code("import socket", allow_network=True).passed


@pytest.mark.parametrize("path", ["/etc/passwd", r"C:\\secret", r"\\server\\share", "a/../b"])
def test_unsafe_literal_open_paths(path):
    with pytest.raises(SecurityViolationException, match="unsafe open"):
        scan_code(f"open({path!r})", allow_filesystem=True)


def test_open_constraints():
    assert scan_code("open('output.txt', 'w')").passed
    assert scan_code("open()").passed
    with pytest.raises(SecurityViolationException, match="non-literal"):
        scan_code("open(file=name)")
    assert scan_code("open(name)", allow_filesystem=True).passed


def test_dunder_attribute_rejected():
    with pytest.raises(SecurityViolationException, match="dunder"):
        scan_code("getattr(x, '__subclasses__')")
    assert scan_code("getattr(x, 'value')").passed


def test_tautologies():
    code = """
assert True
assert 1
assert 'yes'
assert [1]
assert not False
assert not []
x = 2
assert x == x
assert x == 2
assert not True
"""
    assert count_tautological_assertions(code) == 7

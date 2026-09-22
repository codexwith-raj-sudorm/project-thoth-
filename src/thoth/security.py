"""AST-based security and assertion-quality checks.

This scanner is intentionally a heuristic safety layer, not a containment boundary.
"""

from __future__ import annotations

import ast
import ntpath
from dataclasses import dataclass
from pathlib import PurePosixPath

from .exceptions import SecurityViolationException

TIER0_MODULES = {"ctypes", "pty", "code", "codeop", "pickle", "multiprocessing", "builtins"}
TIER0_CALLS = {"eval", "exec", "__import__", "compile", "globals", "locals"}
FILESYSTEM_MODULES = {"os", "pathlib", "shutil", "subprocess"}
NETWORK_MODULES = {"socket", "http", "urllib", "requests"}
DYNAMIC_ATTRIBUTE_CALLS = {"getattr", "setattr", "delattr"}


@dataclass(frozen=True)
class ScanResult:
    passed: bool
    blocked_item: str | None = None

    @property
    def log_value(self) -> str:
        return "pass" if self.passed else (self.blocked_item or "blocked")


def _root_module(name: str | None) -> str:
    return (name or "").split(".", 1)[0]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _unsafe_literal_path(path: str) -> bool:
    # Treat both POSIX and Windows syntax as dangerous on every host.
    normalized = path.replace("\\", "/")
    posix = PurePosixPath(normalized)
    return (
        path.startswith(("/", "\\"))
        or bool(ntpath.splitdrive(path)[0])
        or ".." in posix.parts
    )


def _violation(message: str, node: ast.AST) -> SecurityViolationException:
    return SecurityViolationException(f"{message} on line {getattr(node, 'lineno', '?')}")


def scan_code(code: str, *, allow_filesystem: bool = False, allow_network: bool = False) -> ScanResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SecurityViolationException(f"invalid Python syntax: {exc}") from exc

    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = _root_module(alias.name)
                _check_module(module, allow_filesystem, allow_network, node)
        elif isinstance(node, ast.ImportFrom):
            module = _root_module(node.module)
            _check_module(module, allow_filesystem, allow_network, node)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in TIER0_CALLS:
                raise _violation(f"Tier 0 call is always banned: {name}", node)
            if name == "open":
                path_node = node.args[0] if node.args else next(
                    (kw.value for kw in node.keywords if kw.arg in {"file", "path"}), None
                )
                if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
                    if _unsafe_literal_path(path_node.value):
                        raise _violation(f"unsafe open() path: {path_node.value!r}", node)
                elif path_node is not None and not allow_filesystem:
                    raise _violation("non-literal open() path requires --allow-filesystem", node)
            if name in DYNAMIC_ATTRIBUTE_CALLS and len(node.args) >= 2:
                attr = node.args[1]
                if (
                    isinstance(attr, ast.Constant)
                    and isinstance(attr.value, str)
                    and len(attr.value) >= 4
                    and attr.value.startswith("__")
                    and attr.value.endswith("__")
                ):
                    raise _violation(f"dunder access is banned: {attr.value}", node)
    return ScanResult(True)


def _check_module(
    module: str, allow_filesystem: bool, allow_network: bool, node: ast.AST
) -> None:
    if module in TIER0_MODULES:
        raise _violation(f"Tier 0 module is always banned: {module}", node)
    if module in FILESYSTEM_MODULES and not allow_filesystem:
        raise _violation(f"filesystem module requires --allow-filesystem: {module}", node)
    if module in NETWORK_MODULES and not allow_network:
        raise _violation(f"network module requires --allow-network: {module}", node)


def count_tautological_assertions(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(_is_tautology(node.test) for node in ast.walk(tree) if isinstance(node, ast.Assert))


def _constant_truthiness(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        try:
            return bool(ast.literal_eval(node))
        except (ValueError, TypeError):
            return None
    return None


def _is_tautology(test: ast.AST) -> bool:
    truth = _constant_truthiness(test)
    if truth is True:
        return True
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _constant_truthiness(test.operand) is False
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
    ):
        return ast.dump(test.left, include_attributes=False) == ast.dump(test.comparators[0], include_attributes=False)
    return False

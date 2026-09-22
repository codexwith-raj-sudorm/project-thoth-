from thoth.executor import execute_code


def test_executes_in_tempdir():
    result = execute_code("from pathlib import Path\nassert Path.cwd().name.startswith('thoth-')")
    assert result.exit_code == 0


def test_timeout():
    result = execute_code("while True: pass", timeout=0.05)
    assert result.timed_out
    assert result.exit_code is None

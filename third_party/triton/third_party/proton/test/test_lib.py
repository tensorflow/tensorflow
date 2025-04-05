import pathlib
import pytest

import triton._C.libproton.proton as libproton
from triton.profiler.profile import _select_backend


def test_record():
    id0 = libproton.record_scope()
    id1 = libproton.record_scope()
    assert id1 == id0 + 1


def test_state():
    libproton.enter_state("zero")
    libproton.exit_state()


def test_scope():
    id0 = libproton.record_scope()
    libproton.enter_scope(id0, "zero")
    id1 = libproton.record_scope()
    libproton.enter_scope(id1, "one")
    libproton.exit_scope(id1, "one")
    libproton.exit_scope(id0, "zero")


def test_op():
    id0 = libproton.record_scope()
    libproton.enter_op(id0, "zero")
    libproton.exit_op(id0, "zero")


@pytest.mark.parametrize("source", ["shadow", "python"])
def test_context(source: str, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_context.hatchet"
    session_id = libproton.start(str(temp_file.with_suffix("")), source, "tree", _select_backend(), "")
    depth = libproton.get_context_depth(session_id)
    libproton.finalize(session_id, "hatchet")
    assert depth >= 0
    assert temp_file.exists()


def test_session(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_session.hatchet"
    session_id = libproton.start(str(temp_file.with_suffix("")), "shadow", "tree", _select_backend(), "")
    libproton.deactivate(session_id)
    libproton.activate(session_id)
    libproton.finalize(session_id, "hatchet")
    libproton.finalize_all("hatchet")
    assert temp_file.exists()


def test_add_metrics(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_add_metrics.hatchet"
    libproton.start(str(temp_file.with_suffix("")), "shadow", "tree", _select_backend(), "")
    id1 = libproton.record_scope()
    libproton.enter_scope(id1, "one")
    libproton.add_metrics(id1, {"a": 1.0, "b": 2.0})
    libproton.exit_scope(id1, "one")
    libproton.finalize_all("hatchet")
    assert temp_file.exists()

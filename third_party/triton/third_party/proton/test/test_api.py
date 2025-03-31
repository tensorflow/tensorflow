import json
import triton.profiler as proton
import pathlib


def test_profile_single_session(tmp_path: pathlib.Path):
    temp_file0 = tmp_path / "test_profile0.hatchet"
    session_id0 = proton.start(str(temp_file0.with_suffix("")))
    proton.activate()
    proton.deactivate()
    proton.finalize()
    assert session_id0 == 0
    assert temp_file0.exists()

    temp_file1 = tmp_path / "test_profile1.hatchet"
    session_id1 = proton.start(str(temp_file1.with_suffix("")))
    proton.activate(session_id1)
    proton.deactivate(session_id1)
    proton.finalize(session_id1)
    assert session_id1 == session_id0 + 1
    assert temp_file1.exists()

    session_id2 = proton.start("test")
    proton.activate(session_id2)
    proton.deactivate(session_id2)
    proton.finalize()
    assert session_id2 == session_id1 + 1
    assert pathlib.Path("test.hatchet").exists()
    pathlib.Path("test.hatchet").unlink()


def test_profile_multiple_sessions(tmp_path: pathlib.Path):
    temp_file0 = tmp_path / "test_profile0.hatchet"
    proton.start(str(temp_file0.with_suffix("")))
    temp_file1 = tmp_path / "test_profile1.hatchet"
    proton.start(str(temp_file1.with_suffix("")))
    proton.activate()
    proton.deactivate()
    proton.finalize()
    assert temp_file0.exists()
    assert temp_file1.exists()

    temp_file2 = tmp_path / "test_profile2.hatchet"
    session_id2 = proton.start(str(temp_file2.with_suffix("")))
    temp_file3 = tmp_path / "test_profile3.hatchet"
    session_id3 = proton.start(str(temp_file3.with_suffix("")))
    proton.deactivate(session_id2)
    proton.deactivate(session_id3)
    proton.finalize()
    assert temp_file2.exists()
    assert temp_file3.exists()


def test_profile_decorator(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_profile_decorator.hatchet"

    @proton.profile(name=str(temp_file.with_suffix("")))
    def foo0(a, b):
        return a + b

    foo0(1, 2)
    proton.finalize()
    assert temp_file.exists()

    @proton.profile
    def foo1(a, b):
        return a + b

    foo1(1, 2)
    proton.finalize()
    default_file = pathlib.Path(proton.DEFAULT_PROFILE_NAME + ".hatchet")
    assert default_file.exists()
    default_file.unlink()


def test_scope(tmp_path: pathlib.Path):
    # Scope can be annotated even when profiling is off
    with proton.scope("test"):
        pass

    temp_file = tmp_path / "test_scope.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.scope("test"):
        pass

    @proton.scope("test")
    def foo():
        pass

    foo()

    proton.enter_scope("test")
    proton.exit_scope()
    proton.finalize()
    assert temp_file.exists()


def test_hook(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_hook.hatchet"
    session_id0 = proton.start(str(temp_file.with_suffix("")), hook="triton")
    proton.activate(session_id0)
    proton.deactivate(session_id0)
    proton.finalize(None)
    assert temp_file.exists()


def test_scope_metrics(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_metrics.hatchet"
    session_id = proton.start(str(temp_file.with_suffix("")))
    # Test different scope creation methods
    with proton.scope("test0", {"a": 1.0}):
        pass

    @proton.scope("test1", {"a": 1.0})
    def foo():
        pass

    foo()

    # After deactivation, the metrics should be ignored
    proton.deactivate(session_id)
    proton.enter_scope("test2", metrics={"a": 1.0})
    proton.exit_scope()

    # Metrics should be recorded again after reactivation
    proton.activate(session_id)
    proton.enter_scope("test3", metrics={"a": 1.0})
    proton.exit_scope()

    proton.enter_scope("test3", metrics={"a": 1.0})
    proton.exit_scope()

    # exit_scope can also take metrics
    proton.enter_scope("test4")
    proton.exit_scope(metrics={"b": 1.0})

    proton.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 4
    for child in data[0]["children"]:
        if child["frame"]["name"] == "test3":
            assert child["metrics"]["a"] == 2.0
        elif child["frame"]["name"] == "test4":
            assert child["metrics"]["b"] == 1.0


def test_scope_properties(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_properties.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    # Test different scope creation methods
    # Different from metrics, properties could be str
    with proton.scope("test0", {"a (pty)": "1"}):
        pass

    @proton.scope("test1", {"a (pty)": "1"})
    def foo():
        pass

    foo()

    # Properties do not aggregate
    proton.enter_scope("test2", metrics={"a (pty)": 1.0})
    proton.exit_scope()

    proton.enter_scope("test2", metrics={"a (pty)": 1.0})
    proton.exit_scope()

    proton.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    for child in data[0]["children"]:
        if child["frame"]["name"] == "test2":
            assert child["metrics"]["a"] == 1.0
        elif child["frame"]["name"] == "test0":
            assert child["metrics"]["a"] == "1"


def test_scope_exclusive(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_exclusive.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    # metric a only appears in the outermost scope
    # metric b only appears in the innermost scope
    # both metrics do not appear in the root scope
    with proton.scope("test0", metrics={"a (exc)": "1"}):
        with proton.scope("test1", metrics={"b (exc)": "1"}):
            pass

    proton.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    root_metrics = data[0]["metrics"]
    assert len(root_metrics) == 0
    test0_frame = data[0]["children"][0]
    test0_metrics = test0_frame["metrics"]
    assert len(test0_metrics) == 1
    assert test0_metrics["a"] == "1"
    test1_frame = test0_frame["children"][0]
    test1_metrics = test1_frame["metrics"]
    assert len(test1_metrics) == 1
    assert test1_metrics["b"] == "1"


def test_state(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_state.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    proton.enter_scope("test0")
    proton.enter_state("state")
    proton.enter_scope("test1", metrics={"a": 1.0})
    proton.exit_scope()
    proton.exit_state()
    proton.exit_scope()
    proton.finalize()
    assert temp_file.exists()
    with temp_file.open() as f:
        data = json.load(f)
    # test0->test1->state
    assert len(data[0]["children"]) == 1
    child = data[0]["children"][0]
    assert child["frame"]["name"] == "test0"
    assert len(child["children"]) == 1
    child = child["children"][0]
    assert child["frame"]["name"] == "test1"
    assert len(child["children"]) == 1
    child = child["children"][0]
    assert child["frame"]["name"] == "state"
    assert child["metrics"]["a"] == 1.0


def test_context_depth(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_context_depth.hatchet"
    session_id = proton.start(str(temp_file.with_suffix("")))
    assert proton.context.depth(session_id) == 0
    proton.enter_scope("test0")
    assert proton.context.depth(session_id) == 1
    proton.enter_scope("test1")
    assert proton.context.depth(session_id) == 2
    proton.exit_scope()
    assert proton.context.depth(session_id) == 1
    proton.exit_scope()
    assert proton.context.depth(session_id) == 0
    proton.finalize()


def test_throw(tmp_path: pathlib.Path):
    # Catch an exception thrown by c++
    session_id = 100
    temp_file = tmp_path / "test_throw.hatchet"
    activate_error = ""
    try:
        session_id = proton.start(str(temp_file.with_suffix("")))
        proton.activate(session_id + 1)
    except Exception as e:
        activate_error = str(e)
    finally:
        proton.finalize()
    assert "Session has not been initialized: " + str(session_id + 1) in activate_error

    deactivate_error = ""
    try:
        session_id = proton.start(str(temp_file.with_suffix("")))
        proton.deactivate(session_id + 1)
    except Exception as e:
        deactivate_error = str(e)
    finally:
        proton.finalize()
    assert "Session has not been initialized: " + str(session_id + 1) in deactivate_error

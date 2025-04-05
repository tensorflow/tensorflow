import pytest
import subprocess
from triton.profiler.viewer import get_min_time_flops, get_min_time_bytes, get_raw_metrics, format_frames, derive_metrics, filter_frames, parse
from triton.profiler.hook import COMPUTE_METADATA_SCOPE_NAME
import numpy as np

file_path = __file__
triton_example_file = file_path.replace("test_viewer.py", "examples/triton.json")
cuda_example_file = file_path.replace("test_viewer.py", "examples/cuda.json")
hip_example_file = file_path.replace("test_viewer.py", "examples/hip.json")
frame_example_file = file_path.replace("test_viewer.py", "examples/frame.json")
leaf_example_file = file_path.replace("test_viewer.py", "examples/leaf_nodes.json")


def test_help():
    # Only check if the viewer can be invoked
    subprocess.check_call(["proton-viewer", "-h"], stdout=subprocess.DEVNULL)


def test_exclusive_metrics():
    with open(triton_example_file, "r") as f:
        gf, inclusive_metrics, exclusive_metrics, device_info = get_raw_metrics(f)
        gf.update_inclusive_columns()
        metrics = ["cpu_time/ns"]
        metrics = derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info)
        gf = filter_frames(gf, None, None, None, metrics[0])
        sorted_df = gf.dataframe.sort_values(by=[metrics[0]], ascending=False)
        actual = sorted_df.iloc[0:1]["name"].values[0]
        assert actual == "scope"


def test_sort():
    with open(leaf_example_file, "r") as f:
        gf, inclusive_metrics, exclusive_metrics, device_info = get_raw_metrics(f)
        gf = format_frames(gf, None)
        gf.update_inclusive_columns()
        metrics = ["time/s", "time/ms", "time/us", "time/ns"]
        metrics = derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info)
        gf = filter_frames(gf, None, None, None, metrics[0])
        sorted_df = gf.dataframe.sort_values(by=[metrics[0]], ascending=False)
        actual = sorted_df.iloc[0:5]["name"].values
        expected = ["ROOT", "kernel_1_1_1", "kernel_3_1_1", "kernel_3_2_2", "kernel_1_2_2"]
        assert len(actual) == len(expected)
        assert all(a == b for a, b in zip(actual, expected))


@pytest.mark.parametrize("option", ["full", "file_function_line", "function_line", "file_function"])
def test_format_frames(option):
    with open(frame_example_file, "r") as f:
        gf, _, _, _ = get_raw_metrics(f)
        gf = format_frames(gf, option)
        if option == "full":
            idx = gf.dataframe["name"] == "/home/user/projects/example.py/test.py:1@foo"
        elif option == "file_function_line":
            idx = gf.dataframe["name"] == "test.py:1@foo"
        elif option == "function_line":
            idx = gf.dataframe["name"] == "1@foo"
        elif option == "file_function":
            idx = gf.dataframe["name"] == "test.py@foo"
        assert idx.sum() == 1


@pytest.mark.parametrize("option", ["include", "exclude"])
def test_filter_frames(option):
    include = ""
    exclude = ""
    with open(frame_example_file, "r") as f:
        gf, _, _, _ = get_raw_metrics(f)
        if option == "include":
            include = ".*test0.*"
        elif option == "exclude":
            exclude = ".*test1.*"
        gf = filter_frames(gf, include=include, exclude=exclude)
        idx = gf.dataframe["name"] == "test1"
        assert idx.sum() == 0
        idx = gf.dataframe["name"] == "test0"
        assert idx.sum() == 1


def test_filter_metadata():
    with open(triton_example_file, "r") as f:
        gf, _, _, _ = get_raw_metrics(f)
        assert COMPUTE_METADATA_SCOPE_NAME not in gf.dataframe["name"].tolist()
        assert "cuda_kernel" not in gf.dataframe["name"].tolist()
        assert "scope" in gf.dataframe["name"].tolist()
        assert "triton_kernel" in gf.dataframe["name"].tolist()


def test_parse():
    gf, derived_metrics = parse(["time/s"], triton_example_file)
    for derived_metric in derived_metrics:
        assert derived_metric in gf.inc_metrics or derived_metric in gf.exc_metrics


def test_min_time_flops():
    with open(cuda_example_file, "r") as f:
        gf, _, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        device2_idx = gf.dataframe["device_id"] == "2"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000025]], atol=1e-5)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.00005]], atol=1e-5)
        # sm100
        np.testing.assert_allclose(ret[device2_idx].to_numpy(), [[0.000025]], atol=1e-5)
    with open(hip_example_file, "r") as f:
        gf, _, _, device_info = get_raw_metrics(f)
        ret = get_min_time_flops(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # CDNA2
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[0.000026]], atol=1e-5)
        # CDNA3
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[0.000038]], atol=1e-5)


def test_min_time_bytes():
    with open(cuda_example_file, "r") as f:
        gf, _, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # sm89
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[9.91969e-06]], atol=1e-6)
        # sm90
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[2.48584e-05]], atol=1e-6)
    with open(hip_example_file, "r") as f:
        gf, _, _, device_info = get_raw_metrics(f)
        ret = get_min_time_bytes(gf.dataframe, device_info)
        device0_idx = gf.dataframe["device_id"] == "0"
        device1_idx = gf.dataframe["device_id"] == "1"
        # CDNA2
        np.testing.assert_allclose(ret[device0_idx].to_numpy(), [[6.10351e-06]], atol=1e-6)
        # CDNA3
        np.testing.assert_allclose(ret[device1_idx].to_numpy(), [[1.93378e-05]], atol=1e-6)


def test_percentage():
    pass


def derivation_metrics_test(metrics, expected_data, sample_file, rtol=1e-7, atol=1e-6):
    with open(sample_file, "r") as f:
        gf, inclusive_metrics, exclusive_metrics, device_info = get_raw_metrics(f)
        assert len(inclusive_metrics + exclusive_metrics) > 0, "No metrics found in the input file"
        gf.update_inclusive_columns()
        derived_metrics = derive_metrics(gf, metrics, inclusive_metrics, exclusive_metrics, device_info)
        for derived_metric in derived_metrics:
            np.testing.assert_allclose(gf.dataframe[derived_metric].to_numpy(), expected_data[derived_metric],
                                       rtol=rtol, atol=atol)


def test_avg_time_derivation():
    derivation_metrics_test(
        metrics=["avg_time/s", "avg_time/ms", "avg_time/us", "avg_time/ns"], expected_data={
            "avg_time/s (inc)": [0.0000512, 0.0000205, 0.000205,
                                 0.000205], "avg_time/ms (inc)": [0.0512, 0.02048, 0.2048, 0.2048], "avg_time/us (inc)":
            [51.2, 20.48, 204.8, 204.8], "avg_time/ns (inc)": [51200.0, 20480.0, 204800.0, 204800.0]
        }, sample_file=cuda_example_file)


def test_util():
    derivation_metrics_test(metrics=["util"], expected_data={
        "util": [np.nan, 0.247044, 0.147830, 0.118451],
    }, sample_file=cuda_example_file)


def test_time_derivation():
    derivation_metrics_test(
        metrics=["time/s", "time/ms", "time/us", "time/ns"], expected_data={
            "time/s (inc)": [0.000614, 0.0002048, 0.0002048, 0.0002048],
            "time/ms (inc)": [0.6144, 0.2048, 0.2048, 0.2048],
            "time/us (inc)": [614.4, 204.8, 204.8, 204.8],
            "time/ns (inc)": [614400.0, 204800.0, 204800.0, 204800.0],
            "time/% (inc)": [100.0, 50.0, 50.0, 50.0],
        }, sample_file=cuda_example_file)


def test_bytes_derivation():
    derivation_metrics_test(
        metrics=["byte/s", "gbyte/s", "tbyte/s"], expected_data={
            "byte/s (inc)": [1.953125e+11, 4.88281250e+11, 4.88281250e+10,
                             4.88281250e+10], "gbyte/s (inc)": [195.3125, 488.28125, 48.828125, 48.828125],
            "tbyte/s (inc)": [0.195312, 0.48828125, 0.04882812, 0.04882812]
        }, sample_file=cuda_example_file)


def test_flops_derivation():
    derivation_metrics_test(
        metrics=["flop8/s", "gflop8/s", "tflop8/s"],
        expected_data={
            "flop8/s (inc)": [3.417969e+14, 4.88281250e+14, 4.88281250e+13,
                              4.88281250e+14], "gflop8/s (inc)": [341796.875, 488281.25, 48828.125, 488281.25],
            "tflop8/s (inc)": [341.796875, 488.28125, 48.828125, 488.28125]
        },
        sample_file=cuda_example_file,
    )

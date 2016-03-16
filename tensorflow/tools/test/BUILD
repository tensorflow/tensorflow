# Description:
# Tools for testing

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "system_info_lib",
    srcs = [
        "gpu_info_lib.py",
        "system_info_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "system_info",
    srcs = ["system_info.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":system_info_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "run_and_gather_logs_lib",
    srcs = [
        "run_and_gather_logs_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":system_info_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "run_and_gather_logs",
    srcs = ["run_and_gather_logs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":run_and_gather_logs_lib",
        "//tensorflow:tensorflow_py",
    ],
)

# Unit test that calls run_and_gather_logs on a benchmark, and
# prints the result.
cuda_py_test(
    name = "run_and_gather_logs_test",
    srcs = ["run_and_gather_logs.py"],
    additional_deps = [
        ":run_and_gather_logs",
    ],
    args = [
        "--test_name=" + "//tensorflow/core/kernels:cast_op_test",
        "--test_args=" + "'--benchmarks=BM_cpu_float_bfloat16'",
        "--compilation_mode='$(COMPILATION_MODE)'",
        "--cc_flags='$(CC_FLAGS)'",
    ],
    data = [
        "//tensorflow/core/kernels:cast_op_test",
    ],
    main = "run_and_gather_logs.py",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

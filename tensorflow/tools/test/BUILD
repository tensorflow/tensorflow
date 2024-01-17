# Description:
# Tools for testing

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_cc_logged_benchmark",
    "tf_py_logged_benchmark",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

exports_files([
    "run_and_gather_logs_lib.py",
    "run_and_gather_logs.py",
])

py_strict_library(
    name = "system_info_lib",
    srcs = ["system_info_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":gpu_info_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:device_lib",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:gfile",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "gpu_info_lib",
    srcs = ["gpu_info_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:gfile",
        "@six_archive//:six",
    ],
)

py_strict_binary(
    name = "system_info",
    srcs = ["system_info.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":system_info_lib",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "run_and_gather_logs_lib",
    srcs = [
        "run_and_gather_logs_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":gpu_info_lib",
        ":system_info_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:gfile",
        "@absl_py//absl/logging",
        "@six_archive//:six",
    ],
)

py_strict_binary(
    name = "run_and_gather_logs",
    srcs = ["run_and_gather_logs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":run_and_gather_logs_main_lib",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "run_and_gather_logs_main_lib",
    srcs = ["run_and_gather_logs.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":run_and_gather_logs_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

# Unit test that calls run_and_gather_logs on a benchmark, and
# prints the result.
#cuda_py_test(
#    name = "run_and_gather_logs_test",
#    srcs = ["run_and_gather_logs.py"],
#    deps = [
#        ":run_and_gather_logs",
#    ],
#    args = [
#        "--test_name=" + "//tensorflow/core/kernels:cast_op_test",
#        "--test_args=" + "'--benchmark_filter=BM_cpu_float'",
#    ],
#    data = [
#        "//tensorflow/core/kernels:cast_op_test",
#    ],
#    main = "run_and_gather_logs.py",
#)

tf_cc_logged_benchmark(
    name = "cast_op_benchmark",
    target = "//tensorflow/core/kernels:cast_op_test_gpu",
)

tf_py_logged_benchmark(
    name = "rnn_op_benchmark",
    target = "//tensorflow/python/kernel_tests/nn_ops:rnn_test",
)

tf_py_logged_benchmark(
    name = "sparse_csr_matrix_ops_benchmark",
    target = "//tensorflow/python/kernel_tests/linalg/sparse:csr_sparse_matrix_ops_test",
)

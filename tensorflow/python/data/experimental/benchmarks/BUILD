load("//tensorflow:strict.default.bzl", "py_strict_binary")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "autotune_benchmark_binary",
    srcs = ["autotune_benchmark.py"],
    main = "autotune_benchmark.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "autotune_benchmark",
    srcs = ["autotune_benchmark.py"],
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "parameter_value_benchmark",
    srcs = ["parameter_value_benchmark.py"],
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "csv_dataset_benchmark",
    srcs = ["csv_dataset_benchmark.py"],
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "map_and_batch_benchmark",
    srcs = ["map_and_batch_benchmark.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "map_defun_benchmark",
    srcs = ["map_defun_benchmark.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:map_defun",
        "//tensorflow/python/eager:def_function",
    ],
)

tf_py_strict_test(
    name = "matching_files_benchmark",
    size = "small",
    srcs = ["matching_files_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:matching_files",
    ],
)

tf_py_strict_test(
    name = "optimize_benchmark",
    srcs = ["optimize_benchmark.py"],
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
    ],
)

tf_py_strict_test(
    name = "rejection_resample_benchmark",
    srcs = ["rejection_resample_benchmark.py"],
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:resampling",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "snapshot_dataset_benchmark",
    srcs = ["snapshot_dataset_benchmark.py"],
    deps = [
        "//tensorflow/python:array_ops_gen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:snapshot",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_strict_test(
    name = "unbatch_benchmark",
    srcs = ["unbatch_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

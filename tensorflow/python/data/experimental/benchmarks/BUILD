load("//tensorflow:strict.default.bzl", "py_strict_binary")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_benchmark_test",
)

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
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "autotune_benchmark",
    srcs = ["autotune_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "parameter_value_benchmark",
    srcs = ["parameter_value_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:script_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "csv_dataset_benchmark",
    srcs = ["csv_dataset_benchmark.py"],
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_benchmark_test(
    name = "map_and_batch_benchmark",
    srcs = ["map_and_batch_benchmark.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "map_defun_benchmark",
    srcs = ["map_defun_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:map_defun",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
    ],
)

tf_py_benchmark_test(
    name = "matching_files_benchmark",
    size = "small",
    srcs = ["matching_files_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:matching_files",
    ],
)

tf_py_benchmark_test(
    name = "optimize_benchmark",
    srcs = ["optimize_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/ops:math_ops",
    ],
)

tf_py_benchmark_test(
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

tf_py_benchmark_test(
    name = "snapshot_dataset_benchmark",
    srcs = ["snapshot_dataset_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/experimental/ops:snapshot",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_benchmark_test(
    name = "unbatch_benchmark",
    srcs = ["unbatch_benchmark.py"],
    deps = [
        "//tensorflow/python/data/benchmarks:benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

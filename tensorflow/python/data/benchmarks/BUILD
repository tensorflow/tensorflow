load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "benchmark_base",
    srcs = ["benchmark_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "meta_benchmark",
    srcs = ["meta_benchmark.py"],
    deps = [
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "batch_benchmark",
    srcs = ["batch_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:random_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "filter_benchmark",
    srcs = ["filter_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops:array_ops",
    ],
)

tf_py_benchmark_test(
    name = "from_tensor_slices_benchmark",
    srcs = ["from_tensor_slices_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/experimental/ops:get_single_element",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:structured_function",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:dataset_ops_gen",
        "//third_party/py/numpy",
    ],
)

tf_py_benchmark_test(
    name = "interleave_benchmark",
    srcs = ["interleave_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_benchmark_test(
    name = "list_files_benchmark",
    srcs = ["list_files_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_benchmark_test(
    name = "map_benchmark",
    srcs = ["map_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:while_loop",
    ],
)

tf_py_benchmark_test(
    name = "prefetch_benchmark",
    srcs = ["prefetch_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_benchmark_test(
    name = "range_benchmark",
    srcs = ["range_benchmark.py"],
    deps = [
        ":benchmark_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
    ],
)

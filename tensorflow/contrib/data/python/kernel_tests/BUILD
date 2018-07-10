package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test", "py_test")

py_test(
    name = "batch_dataset_op_test",
    size = "medium",
    srcs = ["batch_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",  # (b/79552534)
        "no_pip",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "bucketing_test",
    size = "medium",
    srcs = ["bucketing_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:grouping",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "csv_dataset_op_test",
    size = "small",
    srcs = ["csv_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/data/python/ops:error_ops",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python/data/ops:readers",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "dataset_constructor_op_test",
    size = "medium",
    srcs = ["dataset_constructor_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "nomac",  # b/62040583
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

py_test(
    name = "directed_interleave_dataset_test",
    size = "medium",
    srcs = ["directed_interleave_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:interleave_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:random_seed",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "get_single_element_test",
    size = "small",
    srcs = ["get_single_element_test.py"],
    deps = [
        "//tensorflow/contrib/data/python/ops:get_single_element",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "interleave_dataset_op_test",
    size = "medium",
    srcs = ["interleave_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_oss",
        "no_pip",
        "notap",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:interleave_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "@six_archive//:six",
    ],
)

py_test(
    name = "iterator_ops_test",
    size = "small",
    srcs = ["iterator_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/data/python/ops:iterator_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
    ],
)

py_test(
    name = "map_dataset_op_test",
    size = "medium",
    srcs = ["map_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "noasan",  # times out
        "optonly",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/contrib/data/python/ops:error_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "optimize_dataset_op_test",
    size = "small",
    srcs = ["optimize_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:optimization",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

cuda_py_test(
    name = "prefetching_ops_test",
    size = "small",
    srcs = ["prefetching_ops_test.py"],
    additional_deps = [
        "//tensorflow/contrib/data/python/ops:prefetching_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:function",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/compat:compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
    tags = ["no_windows_gpu"],
)

py_test(
    name = "range_dataset_op_test",
    size = "small",
    srcs = ["range_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:counter",
        "//tensorflow/contrib/data/python/ops:enumerate_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "reader_dataset_ops_test_base",
    testonly = 1,
    srcs = [
        "reader_dataset_ops_test_base.py",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/contrib/data/python/kernel_tests:__pkg__",
        "//tensorflow/contrib/data/python/kernel_tests/serialization:__pkg__",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

py_test(
    name = "reader_dataset_ops_test",
    size = "medium",
    srcs = ["reader_dataset_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":reader_dataset_ops_test_base",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/ops:readers",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "resample_test",
    size = "medium",
    srcs = ["resample_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    tags = [
        "noasan",
        "optonly",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:resampling",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

py_test(
    name = "scan_dataset_op_test",
    size = "small",
    srcs = ["scan_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/data/python/ops:scan_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "shuffle_dataset_op_test",
    size = "medium",
    srcs = ["shuffle_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:shuffle_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "slide_dataset_op_test",
    size = "small",
    srcs = ["slide_dataset_op_test.py"],
    deps = [
        "//tensorflow/contrib/data/python/ops:sliding",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sql_dataset_op_test_base",
    srcs = ["sql_dataset_op_test_base.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/contrib/data/python/kernel_tests:__pkg__",
        "//tensorflow/contrib/data/python/kernel_tests/serialization:__pkg__",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "@org_sqlite//:python",
    ],
)

py_test(
    name = "sql_dataset_op_test",
    size = "small",
    srcs = ["sql_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":sql_dataset_op_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
    ],
)

py_test(
    name = "stats_dataset_ops_test",
    size = "medium",
    srcs = ["stats_dataset_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":reader_dataset_ops_test_base",
        "//tensorflow/contrib/data/python/ops:stats_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "threadpool_dataset_ops_test",
    size = "small",
    srcs = ["threadpool_dataset_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/data/python/ops:threadpool",
        "//tensorflow/contrib/data/python/ops:unique",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "unique_dataset_op_test",
    size = "small",
    srcs = ["unique_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/data/python/ops:unique",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "window_dataset_op_test",
    size = "medium",
    srcs = ["window_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
    ],
    deps = [
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/contrib/data/python/ops:grouping",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "writer_ops_test",
    size = "small",
    srcs = ["writer_ops_test.py"],
    deps = [
        "//tensorflow/contrib/data/python/ops:writers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

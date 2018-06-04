package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test", "py_test", "tf_py_test")

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
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "bucketing_test",
    size = "medium",
    srcs = ["bucketing_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:grouping",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "concatenate_dataset_op_test",
    size = "small",
    srcs = ["concatenate_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
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
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:batching",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dataset_serialization_test",
    srcs = [
        "dataset_serialization_test_base.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/data/python/ops:iterator_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:iterator_ops",
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
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:error_ops",
        "//tensorflow/contrib/data/python/ops:readers",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "filter_dataset_op_test",
    size = "small",
    srcs = ["filter_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "flat_map_dataset_op_test",
    size = "medium",
    srcs = ["flat_map_dataset_op_test.py"],
    additional_deps = [
        ":dataset_serialization_test",
        "//third_party/py/numpy",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
    ],
    grpc_enabled = True,
    tags = ["no_pip"],
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
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:interleave_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "directed_interleave_dataset_test",
    size = "medium",
    srcs = ["directed_interleave_dataset_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:interleave_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "get_single_element_test",
    size = "small",
    srcs = ["get_single_element_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/contrib/data/python/ops:get_single_element",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
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
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:error_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "prefetch_dataset_op_test",
    size = "small",
    srcs = ["prefetch_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/python:platform",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "range_dataset_op_test",
    size = "small",
    srcs = ["range_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:counter",
        "//tensorflow/contrib/data/python/ops:enumerate_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "reader_dataset_ops_test",
    size = "medium",
    srcs = ["reader_dataset_ops_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
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
    ],
)

py_test(
    name = "scan_dataset_op_test",
    size = "small",
    srcs = ["scan_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:scan_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sequence_dataset_op_test",
    size = "medium",
    srcs = ["sequence_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "serialization_integration_test",
    size = "small",
    srcs = ["serialization_integration_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/contrib/data/python/ops:iterator_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "shuffle_dataset_op_test",
    size = "medium",
    srcs = ["shuffle_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:shuffle_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sql_dataset_op_test",
    size = "small",
    srcs = ["sql_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:readers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "@org_sqlite//:python",
    ],
)

py_test(
    name = "stats_dataset_ops_test",
    size = "medium",
    srcs = ["stats_dataset_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:stats_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
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
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_test(
    name = "unique_dataset_op_test",
    size = "small",
    srcs = ["unique_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/contrib/data/python/ops:unique",
        "//tensorflow/contrib/stateless",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "zip_dataset_op_test",
    size = "small",
    srcs = ["zip_dataset_op_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":dataset_serialization_test",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
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
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
)

tf_py_test(
    name = "slide_dataset_op_test",
    size = "small",
    srcs = ["slide_dataset_op_test.py"],
    additional_deps = [
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/contrib/data/python/ops:sliding",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "writer_ops_test",
    size = "small",
    srcs = ["writer_ops_test.py"],
    additional_deps = [
        "//tensorflow/contrib/data/python/ops:writers",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:readers",
    ],
)

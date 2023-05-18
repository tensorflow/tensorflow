load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")
load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "feature_column_py",
    srcs = ["feature_column_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":feature_column_v2",
    ],
)

py_strict_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "feature_column_v2",
    srcs = [
        "feature_column_v2.py",
        "sequence_feature_column.py",
        "serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/experimental/ops:lookup_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

filegroup(
    name = "vocabulary_testdata",
    srcs = [
        "testdata/embedding.ckpt.data-00000-of-00001",
        "testdata/embedding.ckpt.index",
        "testdata/embedding.ckpt.meta",
        "testdata/unicode_vocabulary",
        "testdata/unicode_vocabulary.tfrecord.gz",
        "testdata/warriors_vocabulary.tfrecord.gz",
        "testdata/warriors_vocabulary.txt",
        "testdata/wire_vocabulary.tfrecord.gz",
        "testdata/wire_vocabulary.txt",
    ],
)

tf_py_strict_test(
    name = "feature_column_test",
    srcs = ["feature_column_test.py"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/206860622): Broken with numpy 1.20+
        "no_pip",
        "no_rocm",
        "no_windows",
    ],
    deps = [
        ":feature_column_test_main_lib",
    ],
)

py_strict_library(
    name = "feature_column_test_main_lib",
    srcs = ["feature_column_test.py"],
    data = [":vocabulary_testdata"],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":feature_column_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "feature_column_v2_test",
    srcs = ["feature_column_v2_test.py"],
    shard_count = 5,
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/206860622): Broken with numpy 1.20+
        "no_pip",
        "no_rocm",
        "no_windows",
    ],
    deps = [":feature_column_v2_test_main_lib"],
)

py_strict_library(
    name = "feature_column_v2_test_main_lib",
    srcs = ["feature_column_v2_test.py"],
    data = [":vocabulary_testdata"],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":feature_column_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/util:nest",
        "@six_archive//:six",
    ],
)

tf_py_strict_test(
    name = "sequence_feature_column_test",
    srcs = ["sequence_feature_column_test.py"],
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":feature_column_v2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "sequence_feature_column_integration_test",
    srcs = ["sequence_feature_column_integration_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_pip"],
    deps = [
        ":feature_column_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":feature_column_v2",
        "//tensorflow/python:client_testlib",
    ],
)

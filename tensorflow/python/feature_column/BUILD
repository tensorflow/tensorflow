load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

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
        ":sequence_feature_column",
        ":serialization",
    ],
)

py_strict_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/layers:layers_base",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:template",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:checkpoint_utils",
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
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_column",
        ":feature_column_v2_types",
        ":serialization",
        ":utils",
        "//tensorflow/python/data/experimental/ops:lookup_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:autotrackable",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/training:checkpoint_utils",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "serialization",
    srcs = [
        "serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":feature_column_v2_types",
        "//tensorflow/python/ops:init_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "sequence_feature_column",
    srcs = ["sequence_feature_column.py"],
    srcs_version = "PY3",
    deps = [
        ":feature_column_v2",
        ":serialization",
        ":utils",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
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
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
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
        ":feature_column_v2_types",
        ":serialization",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:partitioned_variables",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
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
        ":sequence_feature_column",
        ":serialization",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
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
        ":sequence_feature_column",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":feature_column_v2",
        ":serialization",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "feature_column_v2_types",
    srcs = ["feature_column_v2_types.py"],
    deps = ["//tensorflow/python/util:tf_export"],
)

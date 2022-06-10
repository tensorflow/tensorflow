load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    licenses = ["notice"],
)

pytype_strict_library(
    name = "trace_type",
    srcs = [
        "__init__.py",
        "trace_type_builder.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":default_types",
        ":util",
        "//tensorflow/python/types",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [],
)

py_strict_test(
    name = "trace_type_test",
    srcs = ["trace_type_test.py"],
    python_version = "PY3",
    # TODO(b/213375459): For continous benchmark monitoring
    visibility = ["//learning/brain/contrib/eager/python/examples:__pkg__"],
    deps = [
        ":default_types",
        ":trace_type",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "default_types",
    srcs = [
        "default_types.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/python/types"],
)

py_strict_test(
    name = "default_types_test",
    srcs = ["default_types_test.py"],
    python_version = "PY3",
    deps = [
        ":default_types",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "serialization",
    srcs = [
        "serialization.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = ["//tensorflow/core/function/trace_type:serialization_proto_py"],
)

py_strict_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    python_version = "PY3",
    deps = [
        ":serialization",
        "//tensorflow/core/function/trace_type:serialization_test_proto_py",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_proto_library(
    name = "serialization_proto",
    srcs = [
        "serialization.proto",
    ],
    cc_api_version = 2,
    visibility = ["//tensorflow:internal"],
)

tf_proto_library(
    name = "serialization_test_proto",
    srcs = [
        "serialization_test.proto",
    ],
    cc_api_version = 2,
    protodeps = [
        ":serialization_proto",
    ],
    visibility = ["//tensorflow:internal"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "serialization_py_pb2",
#     api_version = 2,
#     visibility = ["//tensorflow:internal"],
#     deps = [":serialization_proto"],
# )
#
# py_proto_library(
#     name = "serialization_test_py_pb2",
#     api_version = 2,
#     visibility = ["//tensorflow:internal"],
#     deps = [":serialization_test_proto"],
# )
# copybara:uncomment_end

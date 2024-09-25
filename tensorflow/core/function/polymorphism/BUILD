load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "type_dispatch",
    srcs = [
        "type_dispatch.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
    deps = [
        ":function_type",
    ],
)

py_strict_test(
    name = "type_dispatch_test",
    srcs = ["type_dispatch_test.py"],
    python_version = "PY3",
    deps = [
        ":function_type",
        ":type_dispatch",
        #internal proto upb dep
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types:trace",
    ],
)

pytype_strict_library(
    name = "function_cache",
    srcs = ["function_cache.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow/python/eager:__subpackages__",
    ],
    deps = [
        ":function_type",
        "//tensorflow/core/function/polymorphism:type_dispatch",
        "//tensorflow/core/function/trace_type",
    ],
)

py_strict_test(
    name = "function_cache_test",
    size = "medium",
    srcs = ["function_cache_test.py"],
    python_version = "PY3",
    # TODO(b/213375459): For continuous benchmark monitoring
    visibility = ["//learning/brain/contrib/eager/python/examples:__pkg__"],
    deps = [
        ":function_cache",
        #internal proto upb dep
        "//tensorflow/core/function/polymorphism:function_type",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types:trace",
    ],
)

pytype_strict_library(
    name = "function_type",
    srcs = [
        "function_type.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/function/polymorphism:function_type_proto_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/core/function/trace_type:serialization",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:trace",
        "@absl_py//absl/logging",
    ],
)

tf_proto_library(
    name = "function_type_proto",
    srcs = [
        "function_type.proto",
    ],
    protodeps = [
        "//tensorflow/core/function/trace_type:serialization_proto",
    ],
    visibility = ["//visibility:private"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "function_type_py_pb2",
#     visibility = ["//visibility:private"],
#     deps = [":function_type_proto"],
# )
# copybara:uncomment_end

py_strict_test(
    name = "function_type_test",
    srcs = ["function_type_test.py"],
    python_version = "PY3",
    deps = [
        ":function_type",
        #internal proto upb dep
        "//tensorflow/core/function/polymorphism:function_type_proto_py",
        "//tensorflow/core/function/trace_type",
        "//tensorflow/core/function/trace_type:serialization",
        "//tensorflow/python/framework:func_graph",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/types:trace",
        "@absl_py//absl/testing:parameterized",
    ],
)

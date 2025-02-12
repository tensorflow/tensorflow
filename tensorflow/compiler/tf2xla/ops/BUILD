load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:shape_inference",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_xla_ops",
    out = "gen_xla_ops.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [
        ":xla_ops",
    ],
)

tf_custom_op_library(
    name = "_xla_ops.so",
    srcs = [
        "xla_ops.cc",
    ],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:shape_inference",
    ],
)

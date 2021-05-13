# Description:
#   Provides ApiDef access and ApiDef validation for TensorFlow.
#
# The following targets can be used to access ApiDefs:
#   :base_api_def
#   :python_api_def
#   :java_api_def

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load(
    "//third_party/mkl:build_defs.bzl",
    "if_mkl",
)
load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

alias(
    name = "base_api_def",
    actual = "//tensorflow/core/api_def/base_api:base_api_def",
    visibility = ["//tensorflow:internal"],
)

alias(
    name = "python_api_def",
    actual = "//tensorflow/core/api_def/python_api:python_api_def",
    visibility = ["//tensorflow:internal"],
)

alias(
    name = "java_api_def",
    actual = "//tensorflow/core/api_def/java_api:java_api_def",
    visibility = ["//tensorflow:internal"],
)

cc_library(
    name = "excluded_ops_lib",
    srcs = ["excluded_ops.cc"],
    hdrs = ["excluded_ops.h"],
    copts = if_mkl(["-DINTEL_MKL=1"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"]),
)

cc_library(
    name = "update_api_def_lib",
    srcs = ["update_api_def.cc"],
    hdrs = ["update_api_def.h"],
    deps = [
        ":excluded_ops_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "update_api_def_test",
    srcs = ["update_api_def_test.cc"],
    deps = [
        ":update_api_def_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_binary(
    name = "update_api_def",
    srcs = [
        "update_api_def_main.cc",
    ],
    data = [
        ":base_api_def",
    ],
    deps = [
        ":update_api_def_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "api_test",
    srcs = ["api_test.cc"],
    data = [
        ":base_api_def",
        ":python_api_def",
    ],
    deps = [
        ":excluded_ops_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_test_internal",
        "//tensorflow/core:op_gen_lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:resource_loader",
    ],
)

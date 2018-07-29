# Description:
# TensorFlow SavedModel.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "if_ios",
    "if_mobile",
    "if_not_mobile",
    "tf_cc_test",
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "signature_constants",
    hdrs = ["signature_constants.h"],
)

cc_library(
    name = "tag_constants",
    hdrs = ["tag_constants.h"],
)

cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    deps = [
        ":constants",
    ] + if_not_mobile([
        # TODO(b/111634734): :lib and :protos_all contain dependencies that
        # cannot be built on mobile platforms. Instead, include the appropriate
        # tf_lib depending on the build platform.
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ]) + if_mobile([
        # Mobile-friendly SavedModel proto. See go/portable-proto for more info.
        "//tensorflow/core:saved_model_portable_proto",
    ]) + if_android([
        "//tensorflow/core:android_tensorflow_lib",
    ]) + if_ios([
        "//tensorflow/core:ios_tensorflow_lib",
    ]),
)

tf_cc_test(
    name = "reader_test",
    srcs = ["reader_test.cc"],
    data = [
        ":saved_model_half_plus_two",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":reader",
        ":tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "loader",
    hdrs = ["loader.h"],
    deps = [
        ":loader_lite",
    ] + if_not_mobile([
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
    ]) + if_android([
        "//tensorflow/core:android_tensorflow_lib",
    ]),
)

cc_library(
    name = "loader_lite",
    srcs = ["loader.cc"],
    hdrs = ["loader.h"],
    deps = [
        ":constants",
        ":reader",
    ] + if_not_mobile([
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/util/tensor_bundle:naming",
        # mobile not supported yet
    ]),
)

tf_cc_test(
    name = "loader_test",
    srcs = ["loader_test.cc"],
    data = [
        ":saved_model_half_plus_two",
    ],
    linkstatic = 1,
    deps = [
        ":constants",
        ":loader",
        ":signature_constants",
        ":tag_constants",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

# TODO(b/32673259): add a test to continuously validate these files.
filegroup(
    name = "saved_model_half_plus_two",
    srcs = glob([
        "testdata/half_plus_two_pbtxt/**",
        "testdata/half_plus_two_main_op/**",
        "testdata/half_plus_two/**",
    ]),
)

# A simple op. for testing and demonstrating the OpKernel interface.

load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_kernel_library")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/kernels/shim:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "simple_op",
    hdrs = ["simple_op.h"],
    deps = [
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "simple_tf_op",
    srcs = ["simple_tf_op.cc"],
    hdrs = ["simple_tf_op.h"],
    deps = [
        ":simple_op",
        "//tensorflow/core:framework",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_test(
    name = "simple_tf_op_test",
    srcs = ["simple_tf_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":simple_tf_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:tstring",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "simple_tflite_op",
    srcs = ["simple_tflite_op.cc"],
    hdrs = ["simple_tflite_op.h"],
    deps = [
        ":simple_op",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
    ],
)

cc_test(
    name = "simple_tflite_op_test",
    srcs = ["simple_tflite_op_test.cc"],
    deps = [
        ":simple_tflite_op",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "tmpl_op",
    hdrs = ["tmpl_op.h"],
    deps = [
        "//tensorflow/lite/kernels/shim:op_kernel",
        "//tensorflow/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "tmpl_tf_op",
    srcs = ["tmpl_tf_op.cc"],
    hdrs = ["tmpl_tf_op.h"],
    deps = [
        ":tmpl_op",
        "//tensorflow/core:framework",
        "//tensorflow/lite/kernels/shim:tf_op_shim",
    ],
)

tf_kernel_library(
    name = "tmpl_tflite_op",
    srcs = ["tmpl_tflite_op.cc"],
    hdrs = ["tmpl_tflite_op.h"],
    deps = [
        ":tmpl_op",
        "//tensorflow/lite:mutable_op_resolver",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels/shim:tflite_op_shim",
        "//tensorflow/lite/kernels/shim:tflite_op_wrapper",
    ],
)

tf_cc_test(
    name = "tmpl_tf_op_test",
    srcs = ["tmpl_tf_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":tmpl_tf_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "tmpl_tflite_op_test",
    srcs = ["tmpl_tflite_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":tmpl_tflite_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()

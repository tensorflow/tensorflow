load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_ios_per_kernel_test", "tflite_portable_test_suite")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "kernels",
    deps = [
        ":add",
        ":concat",
        ":conv",
        ":depthwise_conv",
        ":elementwise",
        ":fully_connected",
        ":max_unpooling",
        ":mean",
        ":mul",
        ":padding",
        ":pooling",
        ":prelu",
        ":relu",
        ":reshape",
        ":resize",
        ":slice",
        ":softmax",
        ":transpose_conv",
    ],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/types:variant",
    ],
)

objc_library(
    name = "add_test_lib",
    testonly = 1,
    srcs = ["add_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":add",
        ":test_util",
    ],
)

ios_unit_test(
    name = "add_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":add_test_lib"],
)

cc_library(
    name = "concat",
    srcs = ["concat.cc"],
    hdrs = ["concat.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
    ],
)

objc_library(
    name = "concat_test_lib",
    testonly = 1,
    srcs = ["concat_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":concat",
        ":test_util",
    ],
)

ios_unit_test(
    name = "concat_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":concat_test_lib"],
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "conv_test_lib",
    testonly = 1,
    srcs = ["conv_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":conv",
        ":test_util",
    ],
)

ios_unit_test(
    name = "conv_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":conv_test_lib"],
)

cc_library(
    name = "custom_registry",
    srcs = ["custom_registry.cc"],
    hdrs = ["custom_registry.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "depthwise_conv_test_lib",
    testonly = 1,
    srcs = ["depthwise_conv_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":depthwise_conv",
        ":test_util",
    ],
)

ios_unit_test(
    name = "depthwise_conv_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":depthwise_conv_test_lib"],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:environment",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "elementwise_test_lib",
    testonly = 1,
    srcs = ["elementwise_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":elementwise",
        ":test_util",
    ],
)

ios_unit_test(
    name = "elementwise_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":elementwise_test_lib"],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:environment",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "fully_connected_test_lib",
    testonly = 1,
    srcs = ["fully_connected_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":fully_connected",
        ":test_util",
    ],
)

ios_unit_test(
    name = "fully_connected_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":fully_connected_test_lib"],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "max_unpooling_test_lib",
    testonly = 1,
    srcs = ["max_unpooling_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":max_unpooling",
        ":test_util",
    ],
)

ios_unit_test(
    name = "max_unpooling_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":max_unpooling_test_lib"],
)

cc_library(
    name = "mean",
    srcs = ["mean.cc"],
    hdrs = ["mean.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/types:variant",
    ],
)

objc_library(
    name = "mean_test_lib",
    testonly = 1,
    srcs = ["mean_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":mean",
        ":test_util",
    ],
)

ios_unit_test(
    name = "mean_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":mean_test_lib"],
)

cc_library(
    name = "mul",
    srcs = ["mul.cc"],
    hdrs = ["mul.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

objc_library(
    name = "mul_test_lib",
    testonly = 1,
    srcs = ["mul_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":mul",
        ":test_util",
    ],
)

ios_unit_test(
    name = "mul_test",
    testonly = 1,
    minimum_os_version = "9.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":mul_test_lib"],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "padding_test_lib",
    testonly = 1,
    srcs = ["padding_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":padding",
        ":test_util",
    ],
)

ios_unit_test(
    name = "padding_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":padding_test_lib"],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "pooling_test_lib",
    testonly = 1,
    srcs = ["pooling_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":pooling",
        ":test_util",
    ],
)

ios_unit_test(
    name = "pooling_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":pooling_test_lib"],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "prelu_test_lib",
    testonly = 1,
    srcs = ["prelu_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":prelu",
        ":test_util",
    ],
)

ios_unit_test(
    name = "prelu_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":prelu_test_lib"],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "relu_test_lib",
    testonly = 1,
    srcs = ["relu_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":relu",
        ":test_util",
    ],
)

ios_unit_test(
    name = "relu_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":relu_test_lib"],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/types:variant",
    ],
)

objc_library(
    name = "resize_test_lib",
    testonly = 1,
    srcs = ["resize_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":resize",
        ":test_util",
    ],
)

ios_unit_test(
    name = "resize_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":resize_test_lib"],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        ":util",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "reshape_test_lib",
    testonly = 1,
    srcs = ["reshape_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":reshape",
        ":test_util",
    ],
)

ios_unit_test(
    name = "reshape_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":reshape_test_lib"],
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = ["slice.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "slice_test_lib",
    testonly = 1,
    srcs = ["slice_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":slice",
        ":test_util",
    ],
)

ios_unit_test(
    name = "slice_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":slice_test_lib"],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "softmax_test_lib",
    testonly = 1,
    srcs = ["softmax_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":softmax",
        ":test_util",
    ],
)

ios_unit_test(
    name = "softmax_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":softmax_test_lib"],
)

cc_library(
    name = "transpose_conv",
    srcs = ["transpose_conv.cc"],
    hdrs = ["transpose_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/metal:environment",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "transpose_conv_test_lib",
    testonly = 1,
    srcs = ["transpose_conv_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":test_util",
        ":transpose_conv",
    ],
)

ios_unit_test(
    name = "transpose_conv_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":transpose_conv_test_lib"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

objc_library(
    name = "test_util",
    testonly = 1,
    srcs = [
        "test_util.mm",
    ],
    hdrs = ["test_util.h"],
    sdk_frameworks = [
        "Metal",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:convert",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:api",
        "//tensorflow/lite/delegates/gpu/metal:common",
        "//tensorflow/lite/delegates/gpu/metal:compiled_model",
        "//tensorflow/lite/delegates/gpu/metal:inference_context",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@FP16",
        "@com_google_absl//absl/memory",
    ],
)

tflite_portable_test_suite()

tflite_ios_per_kernel_test()

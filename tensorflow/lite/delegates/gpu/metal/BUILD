load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite", "tflite_ios_per_kernel_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

DEFAULT_COPTS = [
    "-Wno-shorten-64-to-32",
]

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        ":compiled_model",
        ":compute_task_descriptor",
        ":environment",
        ":runtime_options",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/metal/kernels",
        "//tensorflow/lite/delegates/gpu/metal/kernels:custom_registry",
    ],
)

objc_library(
    name = "buffer_convert",
    srcs = ["buffer_convert.mm"],
    hdrs = ["buffer_convert.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = [
        "Metal",
    ],
    deps = [
        ":common",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:util",
    ],
)

objc_library(
    name = "common",
    srcs = ["common.mm"],
    hdrs = ["common.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = [
        "Metal",
        "UIKit",
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:status",
    ],
)

objc_library(
    name = "common_test_lib",
    testonly = 1,
    srcs = ["common_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":common",
        "//tensorflow/lite/delegates/gpu/metal/kernels:test_util",
    ],
)

ios_unit_test(
    name = "common_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":common_test_lib"],
)

cc_library(
    name = "compiled_model",
    srcs = ["compiled_model.cc"],
    hdrs = ["compiled_model.h"],
    deps = [
        ":compute_task_descriptor",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "compiled_model_test_lib",
    testonly = 1,
    srcs = ["compiled_model_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":compiled_model",
    ],
)

ios_unit_test(
    name = "compiled_model_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":compiled_model_test_lib"],
)

objc_library(
    name = "compute_task",
    srcs = ["compute_task.mm"],
    hdrs = ["compute_task.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":compute_task_descriptor",
        ":runtime_options",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
    ],
)

objc_library(
    name = "compute_task_descriptor",
    srcs = ["compute_task_descriptor.cc"],
    hdrs = ["compute_task_descriptor.h"],
    copts = DEFAULT_COPTS,
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/metal:runtime_options",
        "@FP16",
    ],
)

objc_library(
    name = "environment",
    srcs = ["environment.mm"],
    hdrs = ["environment.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = [
        "Metal",
        "UIKit",
    ],
    deps = [
        ":common",
    ],
)

objc_library(
    name = "environment_test_lib",
    testonly = 1,
    srcs = ["environment_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":environment",
        "//tensorflow/lite/delegates/gpu/metal/kernels:test_util",
    ],
)

ios_unit_test(
    name = "environment_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":environment_test_lib"],
)

objc_library(
    name = "inference_context",
    srcs = ["inference_context.mm"],
    hdrs = ["inference_context.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        ":compute_task",
        ":compute_task_descriptor",
        ":runtime_options",
        "//tensorflow/lite/delegates/gpu/common:memory_management",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "inference_context_test_lib",
    testonly = 1,
    srcs = ["inference_context_test.mm"],
    sdk_frameworks = ["XCTest"],
    deps = [
        ":inference_context",
        "//tensorflow/lite/delegates/gpu/metal/kernels:test_util",
    ],
)

ios_unit_test(
    name = "inference_context_test",
    testonly = 1,
    minimum_os_version = "10.0",
    tags = [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":inference_context_test_lib"],
)

cc_library(
    name = "runtime_options",
    hdrs = ["runtime_options.h"],
)

tflite_portable_test_suite()

tflite_ios_per_kernel_test()

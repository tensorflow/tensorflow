load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_unit_test")
load(
    "//tensorflow/lite:special_rules.bzl",
    "tflite_ios_lab_runner",
    "tflite_ios_per_kernel_test",
    "tflite_portable_test_suite",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("//tensorflow:tensorflow.bzl", "workspace_root")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

DEFAULT_COPTS = [
    "-Wno-shorten-64-to-32",
    "-std=c++17",
]

objc_library(
    name = "buffer",
    srcs = ["buffer.cc"],
    hdrs = ["buffer.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":gpu_object",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "buffer_test_lib",
    testonly = 1,
    srcs = ["buffer_test.mm"],
    sdk_frameworks = [
        "XCTest",
        "Metal",
    ],
    deps = [
        ":buffer",
        "//tensorflow/lite/delegates/gpu/common:types",
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
    ],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
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
    minimum_os_version = "11.4",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = tf_gpu_tests_tags() + [
        "notap",
        "tflite_not_portable_android",
    ],
    deps = [":common_test_lib"],
)

objc_library(
    name = "compute_task",
    srcs = ["compute_task.cc"],
    hdrs = ["compute_task.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":metal_arguments",
        ":metal_device",
        ":metal_spatial_tensor",
        "//tensorflow/lite/delegates/gpu/common:kernel_info",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_operation",
        "//tensorflow/lite/delegates/gpu/common/task:tuning_type",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "gpu_object",
    hdrs = ["gpu_object.h"],
    copts = DEFAULT_COPTS,
    sdk_frameworks = ["Metal"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:access_type",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_object_desc",
    ],
)

objc_library(
    name = "inference_context",
    srcs = ["inference_context.cc"],
    hdrs = ["inference_context.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    features = ["-layering_check"],
    sdk_frameworks = ["Metal"],
    deps = [
        ":compute_task",
        ":inference_context_cc_fbs",
        ":metal_device",
        ":metal_spatial_tensor",
        "//tensorflow/lite/delegates/gpu/common:gpu_model",
        "//tensorflow/lite/delegates/gpu/common:gpu_model_cc_fbs",
        "//tensorflow/lite/delegates/gpu/common:memory_management",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_hints",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/selectors:operation_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:special_selector",
        "//tensorflow/lite/delegates/gpu/common/selectors:subgraph",
        "//tensorflow/lite/delegates/gpu/common/task:profiling_info",
        "//tensorflow/lite/delegates/gpu/common/task:serialization_base",
        "//tensorflow/lite/delegates/gpu/common/task:storage_type_util",
        "//tensorflow/lite/delegates/gpu/common/task:tuning_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

flatbuffer_cc_library(
    name = "inference_context_cc_fbs",
    srcs = ["inference_context.fbs"],
    flatc_args = [
        "--scoped-enums",
        "-I " + workspace_root,
    ],
    includes = [
        "//tensorflow/lite/delegates/gpu/common:gpu_model_cc_fbs_includes",
        "//tensorflow/lite/delegates/gpu/common/task:serialization_base_cc_fbs_includes",
    ],
)

objc_library(
    name = "linear_storage",
    srcs = ["linear_storage.cc"],
    hdrs = ["linear_storage.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":gpu_object",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_linear_desc",
    ],
)

objc_library(
    name = "metal_arguments",
    srcs = ["metal_arguments.cc"],
    hdrs = ["metal_arguments.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":buffer",
        ":gpu_object",
        ":linear_storage",
        ":metal_device",
        ":metal_spatial_tensor",
        ":texture2d",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:arguments",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_object_desc",
        "//tensorflow/lite/delegates/gpu/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

objc_library(
    name = "metal_device",
    srcs = ["metal_device.cc"],
    hdrs = ["metal_device.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = ["//tensorflow/lite/delegates/gpu/common:gpu_info"],
)

objc_library(
    name = "metal_spatial_tensor",
    srcs = ["metal_spatial_tensor.cc"],
    hdrs = ["metal_spatial_tensor.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":gpu_object",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/task:buffer_desc",
        "//tensorflow/lite/delegates/gpu/common/task:gpu_tensor",
        "//tensorflow/lite/delegates/gpu/common/task:tensor_desc",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
    ],
)

objc_library(
    name = "metal_spatial_tensor_test_lib",
    testonly = 1,
    srcs = ["metal_spatial_tensor_test.mm"],
    sdk_frameworks = [
        "XCTest",
        "Metal",
    ],
    deps = [
        ":metal_spatial_tensor",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

objc_library(
    name = "texture2d",
    srcs = ["texture2d.cc"],
    hdrs = ["texture2d.h"],
    copts = DEFAULT_COPTS + [
        "-ObjC++",
    ],
    sdk_frameworks = ["Metal"],
    deps = [
        ":common",
        ":gpu_object",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/task:texture2d_desc",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "texture2d_test_lib",
    testonly = 1,
    srcs = ["texture2d_test.mm"],
    sdk_frameworks = [
        "XCTest",
        "Metal",
    ],
    deps = [
        ":texture2d",
        "//tensorflow/lite/delegates/gpu/common:types",
    ],
)

objc_library(
    name = "TestBinary",
    testonly = 1,
    srcs = ["host.m"],
    sdk_frameworks = ["UIKit"],
)

# Application for local testing in XCode
ios_application(
    name = "TestApplication",
    testonly = 1,
    bundle_id = "com.tensorflow.lite.delegates.gpu.metal.ComponentsTests",
    families = [
        "iphone",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "11.4",
    provisioning_profile = "//tensorflow/lite/delegates/gpu/metal:provisioning_profile.mobileprovision",
    tags = tf_gpu_tests_tags() + [
        "local",
        "notap",
    ],
    deps = [":TestBinary"],
)

objc_library(
    name = "common_tests_lib",
    testonly = 1,
    srcs = [
        "//tensorflow/lite/delegates/gpu/metal:buffer_test.mm",
        "//tensorflow/lite/delegates/gpu/metal:common_test.mm",
        "//tensorflow/lite/delegates/gpu/metal:metal_spatial_tensor_test.mm",
        "//tensorflow/lite/delegates/gpu/metal:texture2d_test.mm",
    ],
    hdrs = [
    ],
    sdk_frameworks = ["XCTest"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:types",
        "//tensorflow/lite/delegates/gpu/metal:buffer",
        "//tensorflow/lite/delegates/gpu/metal:common",
        "//tensorflow/lite/delegates/gpu/metal:inference_context",
        "//tensorflow/lite/delegates/gpu/metal:metal_spatial_tensor",
        "//tensorflow/lite/delegates/gpu/metal:texture2d",
        "//tensorflow/lite/delegates/gpu/metal/kernels:test_util",
        "@com_google_absl//absl/memory",
    ],
)

ios_unit_test(
    name = "ComponentsTests",
    minimum_os_version = "11.4",
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = tf_gpu_tests_tags() + ["notap"],
    test_host = ":TestApplication",
    deps = [
        ":common_tests_lib",
        "//tensorflow/lite/delegates/gpu/metal/kernels:kernel_tests_lib",
    ],
)

tflite_portable_test_suite()

tflite_ios_per_kernel_test()

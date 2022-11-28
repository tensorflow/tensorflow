load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "filegroup",
    "tf_cuda_cc_test",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    # features = ["-parse_headers"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "device_id",
    textual_hdrs = [
        "device_id.h",
        "device_id_manager.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/framework:device_id",
    ] + if_static([
        ":device_id_impl",
    ]),
)

cc_library(
    name = "device_id_impl",
    hdrs = [
        "device_id.h",
        "device_id_manager.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/tsl/framework:device_id_impl",
    ],
)

tf_cuda_library(
    name = "device_mem_allocator",
    srcs = [
        "device_id.h",
    ],
    hdrs = [
        "device_host_allocator.h",
        "device_mem_allocator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":device_id",
        "//tensorflow/compiler/xla/stream_executor:device_mem_allocator",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/platform:stream_executor",
    ],
)

filegroup(
    name = "device_runtime_headers",
    srcs = [
        "device_event_mgr.h",
        "device_host_allocator.h",
        "device_id.h",
        "device_id_manager.h",
        "device_mem_allocator.h",
        "//tensorflow/tsl/framework:device_runtime_headers",
    ],
)

cc_library(
    name = "device_event_mgr",
    srcs = [
        "device_event_mgr.cc",
    ],
    hdrs = [
        "device_event_mgr.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
    ],
)

cc_library(
    name = "device_event_mgr_hdrs",
    textual_hdrs = [
        "device_event_mgr.h",
    ],
)

cc_library(
    name = "device_utils",
    srcs = ["device_utils.cc"],
    hdrs = ["device_utils.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cuda_cc_test(
    name = "device_id_manager_test",
    size = "small",
    srcs = [
        "device_id_manager_test.cc",
    ],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":device_id",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_cuda_cc_test(
    name = "device_event_mgr_test",
    srcs = ["device_event_mgr_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":device_event_mgr",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/tsl/framework:device_id",
    ],
)

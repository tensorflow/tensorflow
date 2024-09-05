load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
    "tf_copts",
    "tf_cuda_library",
)
load(
    "//tensorflow:tensorflow.default.bzl",
    "filegroup",
    "tf_cuda_cc_test",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
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
        "@local_xla//xla/tsl/framework:device_id",
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
        "@local_xla//xla/tsl/framework:device_id_impl",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/platform:stream_executor",
        "@local_xla//xla/stream_executor/integrations:device_mem_allocator",
        "@local_xla//xla/tsl/framework:device_id",
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
        "@local_xla//xla/tsl/framework:device_runtime_headers",
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
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:stream_executor",
        "@com_google_absl//absl/container:flat_hash_map",
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
        "//tensorflow/core/platform:regexp",
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
        "@com_google_googletest//:gtest",
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
        "@local_xla//xla/tsl/framework:device_id",
    ] + if_cuda_or_rocm([
        "@local_xla//xla/stream_executor/gpu:gpu_init",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/platform:stream_executor",
    ]),
)

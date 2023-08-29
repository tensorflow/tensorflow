load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object", "tf_cc_test")
load(
    "//tensorflow/core/tpu:build_defs.bzl",
    "if_libtpu_tf_tensor",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "plugin_c_api_hdrs",
    hdrs = ["plugin_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_tensor_hdrs",
        "//tensorflow/compiler/xla/c:c_api_decl",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
    ],
)

cc_library(
    name = "example_plugin",
    testonly = 1,
    srcs = ["example_plugin.cc"],
    hdrs = ["example_plugin.h"],
    deps = [
        ":plugin_c_api_hdrs",
        "//tensorflow/core/platform:logging",
        "//tensorflow/tsl/platform:env",
        "@tf_runtime//:hostcontext_alwayslink",
    ],
)

tf_cc_test(
    name = "plugin_c_api_test",
    srcs = ["plugin_c_api_test.cc"],
    deps = [
        ":example_plugin",
        ":plugin_c_api_hdrs",
        "//tensorflow/c:c_api",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core/platform:status",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext_alwayslink",
    ],
)

tf_cc_shared_object(
    name = "test_next_pluggable_device_plugin.so",
    srcs = ["test_next_pluggable_device_plugin.cc"],
    visibility = ["//tensorflow/c:__subpackages__"],
    deps = [
        ":plugin_c_api_hdrs",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "outside_compilation_params",
    hdrs = ["outside_compilation_params.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_rendezvous_c_api",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api",
    hdrs = ["tf_rendezvous_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_tensor_hdrs",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api_defn",
    hdrs = ["tf_rendezvous_c_api_defn.h"],
    deps = [
        ":tf_rendezvous_c_api",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tf_rendezvous_c_api_conversions",
    srcs = ["tf_rendezvous_c_api_conversions.cc"],
    hdrs = ["tf_rendezvous_c_api_conversions.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":outside_compilation_params",
        ":tf_device_context_c_api_conversions",
        ":tf_rendezvous_c_api",
        ":tf_rendezvous_c_api_defn",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/tsl/framework:allocator",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + if_libtpu_tf_tensor(),
)

tf_cc_test(
    name = "tf_rendezvous_c_api_conversions_test",
    srcs = ["tf_rendezvous_c_api_conversions_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    deps = [
        ":tf_rendezvous_c_api",
        ":tf_rendezvous_c_api_conversions",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/tsl/framework:allocator",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_device_context_c_api",
    hdrs = ["tf_device_context_c_api.h"],
)

cc_library(
    name = "tf_device_context_c_api_conversions",
    srcs = ["tf_device_context_c_api_conversions.cc"],
    hdrs = ["tf_device_context_c_api_conversions.h"],
    deps = [
        ":tf_device_context_c_api",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework_internal",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + if_libtpu_tf_tensor(),
)

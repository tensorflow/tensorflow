load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_object", "tf_cc_test")

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
        ":tf_rendezvous_c_api",
        ":tf_rendezvous_c_api_defn",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/xla/stream_executor/tpu:proto_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/tsl/framework:allocator",
        "@com_google_absl//absl/log:check",
    ],
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
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/tsl/framework:allocator",
        "@com_google_googletest//:gtest_main",
    ],
)

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "c_api",
    srcs = ["c_api.cc"],
    hdrs = ["c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_pjrt_buffer_util",
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:variable_info",
        "//tensorflow/compiler/jit:variable_info_util",
        "//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/next_pluggable_device:plugin_resource",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

# Plugin should include this target to avoid linking the C API implementation.
cc_library(
    name = "c_api_hdrs",
    hdrs = ["c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "tensor_pjrt_buffer_util",
    srcs = ["tensor_pjrt_buffer_util.cc"],
    hdrs = ["tensor_pjrt_buffer_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "//tensorflow/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "tensor_pjrt_buffer_util_test",
    srcs = ["tensor_pjrt_buffer_util_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_pjrt_buffer_util",
        "//tensorflow/compiler/xla/pjrt:pjrt_api",
        "//tensorflow/compiler/xla/pjrt:pjrt_c_api_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_cpu",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

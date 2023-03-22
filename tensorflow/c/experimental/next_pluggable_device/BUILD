load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

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
        "//tensorflow/c:c_api_headers",
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
    ],
)

# Plugin should include this target to avoid linking the C API implementation.
cc_library(
    name = "c_api_hdrs",
    hdrs = ["c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

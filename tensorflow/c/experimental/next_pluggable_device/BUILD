load("//tensorflow:tensorflow.bzl", "tf_cc_test")
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
        ":tensor_pjrt_buffer_util",
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer",
        "//tensorflow/c:tf_status_internal",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:variable_info",
        "//tensorflow/compiler/jit:variable_info_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/next_pluggable_device:plugin_resource",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/pjrt:pjrt_c_api_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@local_xla//xla/pjrt/c:pjrt_c_api_helpers",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

# Plugin should include this target to avoid linking the C API implementation.
cc_library(
    name = "c_api_hdrs",
    hdrs = ["c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:c_api_macros_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_status_headers",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "tensor_pjrt_buffer_util",
    srcs = ["tensor_pjrt_buffer_util.cc"],
    hdrs = ["tensor_pjrt_buffer_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:pjrt_tensor_buffer_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:global_state",
        "//tensorflow/core/tfrt/common:pjrt_state",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/pjrt:pjrt_c_api_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "tensor_pjrt_buffer_util_test",
    srcs = ["tensor_pjrt_buffer_util_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_pjrt_buffer_util",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:casts",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/pjrt:pjrt_api",
        "@local_xla//xla/pjrt:pjrt_c_api_client",
        "@local_xla//xla/pjrt/c:pjrt_c_api_cpu",
        "@local_xla//xla/pjrt/c:pjrt_c_api_hdrs",
        "@local_xla//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

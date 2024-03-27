load(
    "@local_tsl//tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

# Authorized users go here.
package_group(
    name = "friends",
    packages = [
        # copybara:uncomment "//learning/brain/experimental/dtensor/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/google/monitoring/...",
        # copybara:uncomment "//learning/brain/google/xla/...",
        # copybara:uncomment "//learning/brain/tfrc/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/serving/model_servers/...",
        # copybara:uncomment "//platforms/xla/megascale/tensorflow/...",
        "//tensorflow/c/...",
        "//tensorflow/compiler/jit/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/core/common_runtime/...",
        "//tensorflow/core/common_runtime/next_pluggable_device/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/core/tpu/...",
        "//tensorflow/dtensor/...",
        "//third_party/tf_runtime_google/...",
    ],
)

cc_library(
    name = "global_state",
    srcs = [
        "global_state.cc",
    ],
    hdrs = [
        "global_state.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@local_xla//xla/pjrt:utils",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "async_value_tensor",
    srcs = [
        "async_value_tensor.cc",
    ],
    hdrs = [
        "async_value_tensor.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "@local_xla//xla/pjrt:pjrt_client",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "pjrt_state",
    srcs = [
        "pjrt_state.cc",
    ],
    hdrs = [
        "pjrt_state.h",
    ],
    visibility = [":friends"],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:local_device_state",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tf_pjrt_client",
        "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
    ],
)

# This target does not have depend on GPU and XLA compilation to have a small size.
# Uses this target if you do not need to create a PJRT client.
cc_library(
    name = "pjrt_util",
    srcs = [
        "pjrt_util.cc",
    ],
    hdrs = [
        "pjrt_util.h",
    ],
    visibility = [":friends"],
    deps = [
        ":global_state",
        ":pjrt_state",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

# Uses this target if you need to create a PJRT client.
# TODO(b/280671896) Combines pjrt_util and create_pjrt_client_util
cc_library(
    name = "create_pjrt_client_util",
    srcs = [
        "create_pjrt_client_util.cc",
    ],
    hdrs = [
        "create_pjrt_client_util.h",
    ],
    visibility = [":friends"],
    deps = [
        ":global_state",
        ":pjrt_state",
        "//tensorflow/core:framework_types_hdr",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cc_test(
    name = "pjrt_state_test",
    srcs = ["pjrt_state_test.cc"],
    deps = [
        ":global_state",
        ":pjrt_state",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/core/tfrt/common:pjrt_cpu_client_registration",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

tf_cc_test(
    name = "pjrt_util_test",
    srcs = ["pjrt_util_test.cc"],
    deps = [
        ":global_state",
        ":pjrt_state",
        ":pjrt_util",
        "//tensorflow/core:framework",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

tf_cuda_cc_test(
    name = "create_pjrt_client_util_test",
    srcs = ["create_pjrt_client_util_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":create_pjrt_client_util",
        ":global_state",
        ":pjrt_gpu_client_registration",
        ":pjrt_state",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@local_xla//xla/service:gpu_plugin",
    ],
)

cc_library(
    name = "pjrt_client_factory_options",
    hdrs = ["pjrt_client_factory_options.h"],
    visibility = [":friends"],
)

cc_library(
    name = "pjrt_client_factory_registry",
    srcs = ["pjrt_client_factory_registry.cc"],
    hdrs = ["pjrt_client_factory_registry.h"],
    deps = [
        ":pjrt_client_factory_options",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "@local_tsl//tsl/framework:device_type",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_cpu_client_registration",
    srcs = ["pjrt_cpu_client_registration.cc"],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        "//tensorflow/core:framework_types_hdr",
        "@local_xla//xla:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "pjrt_cpu_client_registration_test",
    srcs = ["pjrt_cpu_client_registration_test.cc"],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        ":pjrt_cpu_client_registration",
        "//tensorflow/core:framework_types_hdr",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pjrt_gpu_client_registration",
    srcs = ["pjrt_gpu_client_registration.cc"],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework_types_hdr",
        "@local_xla//xla:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/gpu:se_gpu_pjrt_client",
    ],
    alwayslink = True,
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "pjrt_gpu_client_registration_test",
    size = "small",
    srcs = ["pjrt_gpu_client_registration_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        ":pjrt_gpu_client_registration",
        "//tensorflow/core:framework_types_hdr",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/service:gpu_plugin",
    ],
)

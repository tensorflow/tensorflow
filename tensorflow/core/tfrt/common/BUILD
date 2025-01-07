load(
    "@local_xla//xla/tsl/platform:build_config_root.bzl",
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
        "//tensorflow/core:portable_gif_internal",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla/pjrt:pjrt_client",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "async_value_tensor_test",
    srcs = ["async_value_tensor_test.cc"],
    deps = [
        ":async_value_tensor",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/tsl/concurrency:async_value",
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
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/framework:resource_base",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:statusor",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:refcount",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:errors",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cc_test(
    name = "pjrt_state_test",
    srcs = ["pjrt_state_test.cc"],
    deps = [
        ":pjrt_cpu_client_registration",
        ":pjrt_state",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "pjrt_util_test",
    srcs = ["pjrt_util_test.cc"],
    deps = [
        ":pjrt_state",
        ":pjrt_util",
        "//tensorflow/core:framework",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
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
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:test_main",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/service:gpu_plugin",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
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
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/tsl/framework:device_type",
    ],
)

cc_library(
    name = "pjrt_cpu_client_registration",
    srcs = ["pjrt_cpu_client_registration.cc"],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        "//tensorflow/core:framework_types_hdr",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
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
        "@local_tsl//tsl/platform:statusor",
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
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "@local_xla//xla/pjrt/plugin/xla_gpu:xla_gpu_pjrt_client",
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
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/service:gpu_plugin",
    ],
)

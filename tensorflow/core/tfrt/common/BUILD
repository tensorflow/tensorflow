load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

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
        # copybara:uncomment "//learning/brain/google/xla/...",
        # copybara:uncomment "//learning/brain/tfrc/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//platforms/xla/megascale/tensorflow/...",
        "//tensorflow/c/...",
        "//tensorflow/compiler/jit/...",
        "//tensorflow/core/common_runtime/next_pluggable_device/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/core/tfrt/eager/backends/tpu/...",
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
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework",
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
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
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
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
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
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        ":pjrt_util",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tf_pjrt_client",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
    ],
)

tf_cc_test(
    name = "pjrt_state_test",
    srcs = ["pjrt_state_test.cc"],
    deps = [
        ":global_state",
        ":pjrt_state",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "pjrt_util_test",
    srcs = ["pjrt_util_test.cc"],
    deps = [
        ":global_state",
        ":pjrt_state",
        ":pjrt_util",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/core:framework",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cuda_cc_test(
    name = "create_pjrt_client_util_test",
    srcs = ["create_pjrt_client_util_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":create_pjrt_client_util",
        ":global_state",
        ":pjrt_gpu_client_registration",
        ":pjrt_state",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/core:framework",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "pjrt_client_factory_options",
    hdrs = ["pjrt_client_factory_options.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/compiler/xla/pjrt/gpu:gpu_helpers",
    ],
)

cc_library(
    name = "pjrt_client_factory_registry",
    srcs = ["pjrt_client_factory_registry.cc"],
    hdrs = ["pjrt_client_factory_registry.h"],
    deps = [
        ":pjrt_client_factory_options",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/tsl/framework:device_type",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_cpu_client_registration",
    srcs = ["pjrt_cpu_client_registration.cc"],
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/core:framework_types_hdr",
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
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt/gpu:se_gpu_pjrt_client",
        "//tensorflow/core:framework_types_hdr",
    ],
    alwayslink = True,
)

tf_cuda_cc_test(
    name = "pjrt_gpu_client_registration_test",
    size = "small",
    srcs = ["pjrt_gpu_client_registration_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":pjrt_client_factory_options",
        ":pjrt_client_factory_registry",
        ":pjrt_gpu_client_registration",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/core:framework_types_hdr",
        "@com_google_googletest//:gtest_main",
    ],
)

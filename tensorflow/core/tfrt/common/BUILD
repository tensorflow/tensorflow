load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_test")

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
    compatible_with = if_google([
        "//buildenv/target:libtpu",
        "//buildenv/target:gce",
    ]),
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
    compatible_with = if_google([
        "//buildenv/target:libtpu",
        "//buildenv/target:gce",
    ]),
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
    compatible_with = if_google([
        "//buildenv/target:libtpu",
        "//buildenv/target:gce",
    ]),
    visibility = [":friends"],
    deps = [
        ":global_state",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "pjrt_util",
    srcs = [
        "pjrt_util.cc",
    ],
    hdrs = [
        "pjrt_util.h",
    ],
    compatible_with = if_google([
        "//buildenv/target:libtpu",
        "//buildenv/target:gce",
    ]),
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
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:tfrt_cpu_pjrt_client",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

load("//tensorflow:tensorflow.bzl", "if_google")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/core/tfrt/...",
        # copybara:uncomment "//learning/brain/experimental/dtensor/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/brain/google/xla/...",
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
    compatible_with = if_google(["//buildenv/target:libtpu"]),
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@tf_runtime//:hostcontext",
    ],
)

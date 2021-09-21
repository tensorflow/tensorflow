load("//tensorflow:tensorflow.bzl", "if_google")

package(
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
        "//tensorflow/core/tfrt/...",
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

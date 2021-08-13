package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/core/tfrt/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/mobile/lite/delegates/tfmrt/...",
    ],
)

cc_library(
    name = "fallback_state",
    srcs = ["fallback_state.cc"],
    hdrs = ["fallback_state.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/tfrt/utils:statusor",
    ],
)

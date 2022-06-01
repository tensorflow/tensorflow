load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/tfrt/...",
    ],
)

tfrt_cc_library(
    name = "tf_jitrt_query_of_death",
    hdrs = ["tf_jitrt_query_of_death.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

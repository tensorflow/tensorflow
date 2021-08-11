# Temporary home for go/XLIR (XLA:GPU on TFRT) targets that depend on both XLA
# and TFRT. Currently, neither repository can depend on the other, but this will
# be fixed with b/188517768, at which point these targets should be moved to a
# more suitable home.

load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/xla/service/gpu/...",
    ],
)

tfrt_cc_library(
    name = "gpu_shared_context",
    srcs = ["gpu_shared_context.cc"],
    hdrs = ["gpu_shared_context.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@tf_runtime//:support",
    ],
)

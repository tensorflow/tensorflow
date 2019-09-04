package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/xla:friends"],
)

cc_library(
    name = "legalize_to_affine",
    srcs = ["legalize_to_affine.cc"],
    hdrs = ["legalize_to_affine.h"],
    deps = [
        "//tensorflow/compiler/mlir/xla:lhlo",
        "//tensorflow/compiler/xla:status",
        "@com_google_absl//absl/memory",
        "@local_config_mlir//:AffineOps",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
    ],
    alwayslink = 1,
)

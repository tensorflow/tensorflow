load("//tensorflow/compiler/mlir/tfr:build_defs.bzl", "gen_op_bindings")

package(
    default_visibility = [
        ":friends",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/...",
    ],
)

filegroup(
    name = "decomposition_lib",
    srcs = ["decomposition_lib.mlir"],
)

gen_op_bindings(name = "composite")

gen_op_bindings(name = "test")

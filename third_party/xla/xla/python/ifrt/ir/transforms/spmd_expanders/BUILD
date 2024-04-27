load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "spmd_expander",
    srcs = glob([
        "*spmd_expander.cc",
    ]),
    hdrs = glob([
        "*spmd_expander.h",
    ]),
    compatible_with = get_compatible_with_portable(),
    visibility = ["//xla/python/ifrt:friends"],
    deps = [
        "//xla/python/ifrt/ir",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

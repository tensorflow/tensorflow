load("//xla:lit.bzl", "enforce_glob", "lit_test_suite")
load("//xla/tsl:tsl.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

lit_test_suite(
    name = "all_tests",
    srcs = enforce_glob(
        [
            # go/keep-sorted start
            "chlo.mlir",
            "emit_mhlo.hlo",
            "emit_proto.mlir",
            "print_large_constants.mlir",
            "print_layouts.mlir",
            "simple.hlo",
            "simple.mlir",
            "stablehlo.mlir",
            "vhlo_input.mlir",
            # go/keep-sorted end
        ],
        include = [
            "*.mlir",
            "*.hlo",
        ],
    ),
    cfg = "//xla:lit.cfg.py",
    data = [
        "vhlo_input.mlir.bc",
        ":test_utilities",
    ],
    tools = [
        "//xla/hlo/tools:hlo-translate",
        "//xla/mlir_hlo:mlir-hlo-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
)

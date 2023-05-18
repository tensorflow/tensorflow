load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    data = [":test_utilities"],
    driver = "//tensorflow/compiler/xla:run_lit.sh",
    test_file_exts = ["mlir"],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":ifrt-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//mlir:run_lit.sh",
    ],
)

cc_binary(
    name = "ifrt-opt",
    srcs = ["ifrt-opt.cc"],
    deps = [
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
    ],
)

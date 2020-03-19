load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(licenses = ["notice"])

glob_lit_tests(
    data = [
        ":graph_config_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    tags_override = {
        "fadd_quant.mlir": ["no_oss"],  # TODO(b/150957738): to be fixed on oss.
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/aot:tfcompile",
        "//tensorflow/compiler/mlir:tf-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

# Bundle together all the graph files that are used by the tests.
filegroup(
    name = "graph_config_files",
    srcs = glob(
        ["**/*.pbtxt"],
    ),
)

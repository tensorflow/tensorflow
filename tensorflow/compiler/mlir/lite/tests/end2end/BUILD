load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

licenses(["notice"])

glob_lit_tests(
    data = [
        ":quant_stats_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "pbtxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_to_string",
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate",
        "@llvm-project//llvm:FileCheck",
    ],
)

# Bundle together all the quant stats files that are used by the tests.
filegroup(
    name = "quant_stats_files",
    srcs = glob(
        ["**/*.stats"],
    ),
)

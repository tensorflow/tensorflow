load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

licenses(["notice"])

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@local_config_mlir//:run_lit.sh",
    test_file_exts = ["pbtxt"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//tensorflow/compiler/mlir/lite:flatbuffer_to_string",
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate",
        "@llvm//:FileCheck",
    ],
)

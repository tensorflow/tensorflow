load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

glob_lit_tests(
    data = [
        ":quant_stats_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    size_override = {
        "quant_stats.pbtxt": "medium",
    },
    tags_override = {
        "add.pbtxt": ["no_rocm"],
        "conv_2d.pbtxt": ["no_rocm"],
        "fake_quant_per_channel.pbtxt": ["no_rocm"],
    },
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
        "@llvm-project//llvm:not",
    ],
)

# Bundle together all the quant stats files that are used by the tests.
filegroup(
    name = "quant_stats_files",
    srcs = glob(
        ["**/*.stats"],
    ),
)

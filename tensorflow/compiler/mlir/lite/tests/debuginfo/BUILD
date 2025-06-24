load("//tensorflow:strict.default.bzl", "py_strict_binary")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":debug_info_files",
        ":test_utilities",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = [
        "pbtxt",
        #        "py", TODO(b/150304798)
    ],
)

# Bundle together all the debug info files that are used by the tests.
filegroup(
    name = "debug_info_files",
    srcs = glob(
        ["**/*.debug"],
    ),
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":concrete_function_error",
        ":saved_model_error",
        "//tensorflow/compiler/mlir/lite:flatbuffer_to_string",
        "//tensorflow/compiler/mlir/lite:tf_tfl_translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)

py_strict_binary(
    name = "saved_model_error",
    srcs = ["saved_model_error.py"],
    main = "saved_model_error.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "concrete_function_error",
    srcs = ["concrete_function_error.py"],
    main = "concrete_function_error.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
    ],
)

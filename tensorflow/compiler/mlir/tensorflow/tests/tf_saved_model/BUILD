load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    licenses = ["notice"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:pywrap_mlir",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_library(
    name = "common_v1",
    srcs = ["common_v1.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:pywrap_mlir",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
    ],
)

test_files = glob(
    ["*.py"],
    exclude = [
        "common.py",
        "common_v1.py",
    ],
)

[
    py_binary(
        name = file[:-3],
        testonly = 1,
        srcs = [file],
        deps = [
            "//tensorflow/compiler/mlir/tensorflow/tests/tf_saved_model:common",
            "//tensorflow/compiler/mlir/tensorflow/tests/tf_saved_model:common_v1",
        ],
    )
    for file in test_files
]

glob_lit_tests(
    data = [":test_utilities"],
    default_tags = [
        "no_mac",  # TODO(b/191167848)
        "no_oss",  # TODO(b/190855110)
        "no_pip",
        "no_rocm",
    ],
    driver = "@llvm-project//mlir:run_lit.sh",
    exclude = [
        "common.py",
        "common_v1.py",
    ],
    per_test_extra_data = {
        file: [file[:-3]]
        for file in test_files
    },
    test_file_exts = ["py"],
)

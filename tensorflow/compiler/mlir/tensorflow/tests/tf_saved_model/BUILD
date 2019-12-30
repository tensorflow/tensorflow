package(
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow/compiler/mlir/tensorflow/tests/tf_saved_model:build_defs.bzl", "tf_saved_model_test")

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm-project//llvm:FileCheck",
    ],
)

# Drop trailing ".py" from all test file names.
all_test_basenames = [py[:-3] for py in glob(
    ["*.py"],
    exclude = ["common.py"],
)]

# Instantiate all the tests.
[tf_saved_model_test(
    name = name,
    data = [":test_utilities"],
) for name in all_test_basenames]

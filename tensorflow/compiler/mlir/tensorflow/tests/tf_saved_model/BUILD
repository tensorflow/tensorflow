load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/compiler/mlir/tensorflow/tests/tf_saved_model:build_defs.bzl", "tf_saved_model_test")

package(
    licenses = ["notice"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "common_v1",
    srcs = ["common_v1.py"],
    srcs_version = "PY3",
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
    exclude = [
        "common.py",
        "common_v1.py",
        "keras.py",  # TODO(b/190855110)
    ],
)]

# Instantiate all the tests.
[tf_saved_model_test(
    name = name,
    data = [":test_utilities"],
    tags = [
        "no_mac",  # b/191167848
        "no_pip",
    ],
) for name in all_test_basenames]

tf_saved_model_test(
    name = "keras",
    data = [":test_utilities"],
    tags = [
        "no_oss",  # TODO(b/190855110)
        "no_pip",
    ],
)

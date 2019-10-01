package(
    licenses = ["notice"],  # Apache 2.0
)

load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow/compiler/mlir/tensorflow/tests/tf_saved_model:build_defs.bzl", "tf_saved_model_test")

glob_lit_tests(
    data = [":test_utilities"],
    driver = "@local_config_mlir//:run_lit.sh",
    exclude = ["common.py"],
    test_file_exts = ["py"],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

test_data_files = [
    tf_saved_model_test("basic"),
    tf_saved_model_test("keras"),
    tf_saved_model_test("cyclic_object_graph"),
    tf_saved_model_test("dag_object_graph"),
    tf_saved_model_test("shapes_for_arguments"),
    tf_saved_model_test("shapes_for_variables"),
]

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "@llvm//:FileCheck",
    ] + test_data_files,
)

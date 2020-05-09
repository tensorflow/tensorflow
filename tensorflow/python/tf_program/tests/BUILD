load("//tensorflow:tensorflow.bzl", "tf_python_pybind_extension")

package(licenses = ["notice"])

tf_python_pybind_extension(
    name = "filecheck_wrapper",
    srcs = ["filecheck_wrapper.cc"],
    module_name = "filecheck_wrapper",
    deps = [
        "//tensorflow/python:pybind11_lib",
        "//tensorflow/python:pybind11_status",
        "@llvm-project//llvm:support",
        "@pybind11",
    ],
)

py_test(
    name = "mlir_gen_test",
    size = "small",
    testonly = True,
    srcs = ["mlir_gen_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_oss_py2",
        "no_pip",
    ],
    deps = [
        ":filecheck_wrapper",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/tf_program:mlir_gen",
        "//tensorflow/python/types",
    ],
)

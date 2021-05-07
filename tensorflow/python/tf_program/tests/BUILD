package(licenses = ["notice"])

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
        "//tensorflow/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/tf_program:mlir_gen",
        "//tensorflow/python/types",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "dlpack",
    srcs = ["dlpack.py"],
    deps = [
    ], 
    srcs_version = "PY3",
)

cuda_py_test(
    name = "dlpack_test",
    srcs = ["dlpack_test.py"],
    python_version = "PY3",
    deps = [
        ":dlpack",      
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ]
)
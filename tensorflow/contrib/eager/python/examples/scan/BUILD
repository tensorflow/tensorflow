licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

cuda_py_test(
    name = "scan_test",
    size = "small",
    srcs = ["scan_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "scan_graph_test",
    size = "small",
    srcs = ["scan_graph_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
    ],
)

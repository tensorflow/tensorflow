# Description:
# Some useful scripts that are bundled with TensorBoard.

package(default_visibility = ["//tensorflow/tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "generate_testdata",
    srcs = ["generate_testdata.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "execrooter",
    srcs = ["execrooter.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all_files",
    srcs = glob(["*"]),
    visibility = ["//tensorflow:__subpackages__"],
)

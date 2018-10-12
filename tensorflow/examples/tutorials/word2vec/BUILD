# Description:
# TensorFlow model for word2vec

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "word2vec_basic",
    srcs = [
        "word2vec_basic.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "no-internal-py3",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

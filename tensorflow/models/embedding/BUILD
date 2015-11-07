# Description:
# TensorFlow model for word2vec

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("/tensorflow/tensorflow", "tf_gen_op_wrapper_py")

py_binary(
    name = "word2vec",
    srcs = [
        "word2vec.py",
    ],
    deps = [
        ":gen_word2vec",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "word2vec_optimized",
    srcs = [
        "word2vec_optimized.py",
    ],
    deps = [
        ":gen_word2vec",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "word2vec_test",
    size = "small",
    srcs = ["word2vec_test.py"],
    deps = [
        ":word2vec",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "word2vec_optimized_test",
    size = "small",
    srcs = ["word2vec_optimized_test.py"],
    deps = [
        ":word2vec_optimized",
        "//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "word2vec_ops",
    srcs = [
        "word2vec_ops.cc",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "word2vec_kernels",
    srcs = [
        "word2vec_kernels.cc",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_word2vec",
    out = "gen_word2vec.py",
    deps = [":word2vec_ops"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

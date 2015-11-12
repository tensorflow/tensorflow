# Description:
# Example TensorFlow models for MNIST that achieves high accuracy

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "convolutional",
    srcs = [
        "convolutional.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = ["//tensorflow:tensorflow_py"],
)

py_test(
    name = "convolutional_test",
    size = "medium",
    srcs = [
        "convolutional.py",
    ],
    args = [
        "--self_test=True",
    ],
    main = "convolutional.py",
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
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

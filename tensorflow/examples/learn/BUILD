# Description:
# Examples of tf.learn usage

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "boston",
    srcs = ["boston.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "hdf5_classification",
    srcs = ["hdf5_classification.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "iris",
    srcs = ["iris.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "iris_custom_decay_dnn",
    srcs = ["iris_custom_decay_dnn.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "iris_custom_model",
    srcs = ["iris_custom_model.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "iris_run_config",
    srcs = ["iris_run_config.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "random_forest_mnist",
    srcs = ["random_forest_mnist.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:input_data",
        "//tensorflow/python:platform",
    ],
)

py_binary(
    name = "resnet",
    srcs = ["resnet.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_binary(
    name = "text_classification",
    srcs = ["text_classification.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/layers:layers_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "text_classification_character_cnn",
    srcs = ["text_classification_character_cnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "text_classification_character_rnn",
    srcs = ["text_classification_character_rnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "text_classification_cnn",
    srcs = ["text_classification_cnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "multiple_gpu",
    srcs = ["multiple_gpu.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

sh_test(
    name = "examples_test",
    size = "large",
    srcs = ["examples_test.sh"],
    data = [
        ":boston",
        ":iris",
        ":iris_custom_decay_dnn",
        ":iris_custom_model",
        ":iris_run_config",
        ":random_forest_mnist",
        ":resnet",
        ":text_classification",
        ":text_classification_character_cnn",
        ":text_classification_character_rnn",
        ":text_classification_cnn",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

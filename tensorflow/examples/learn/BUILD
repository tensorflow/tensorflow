# Description:
# Examples of tf.learn usage

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

# This target bundles many scripts into a single py_binary so they can be
# executed by saved_model_test without exploding the data dependencies.
py_binary(
    name = "run_script",
    srcs = [
        "export_mnist_cnn.py",
        "export_rnn_cell.py",
        "export_simple_text_embedding.py",
        "export_text_rnn_model.py",
        "run_script.py",
        "use_mnist_cnn.py",
        "use_model_in_sequential_keras.py",
        "use_rnn_cell.py",
        "use_text_embedding_in_dataset.py",
        "use_text_rnn_model.py",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":mnist_util",
        ":util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mnist_util",
    srcs = ["mnist_util.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "saved_model_test",
    srcs = [
        "saved_model_test.py",
    ],
    data = [
        ":run_script",
    ],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        # NOTE: Split SavedModelTest due to Forge input size limit.
        "no_cuda_on_cpu_tap",  # forge input size exceeded
        "noasan",  # forge input size exceeded
        "nomsan",  # forge input size exceeded
        "notsan",  # forge input size exceeded
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "integration_scripts",
    srcs = [
        "deploy_mnist_cnn.py",
        "export_mnist_cnn.py",
        "export_rnn_cell.py",
        "export_simple_text_embedding.py",
        "export_text_rnn_model.py",
        "integration_scripts.py",
        "use_mnist_cnn.py",
        "use_model_in_sequential_keras.py",
        "use_rnn_cell.py",
        "use_text_embedding_in_dataset.py",
        "use_text_rnn_model.py",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":distribution_strategy_utils",
        ":mnist_util",
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "mnist_util",
    srcs = ["mnist_util.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "distribution_strategy_utils",
    srcs = ["distribution_strategy_utils.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:strategy_combinations",
    ],
)

distribute_py_test(
    name = "saved_model_test",
    srcs = [
        "saved_model_test.py",
    ],
    shard_count = 4,
    tags = [
        "no_pip",  # b/131697937 and b/132196869
        "noasan",  # forge input size exceeded
        "nomsan",  # forge input size exceeded
        "notsan",  # forge input size exceeded
    ],
    tpu_tags = [
        "no_oss",  # Test infra collision (b/157754990)
    ],
    deps = [
        ":distribution_strategy_utils",
        ":integration_scripts",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python/distribute:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

# b/132234211: Target added to support internal test target that runs the test
# in an environment that has the extra dependencies required to test integration
# with non core tensorflow packages.
py_library(
    name = "saved_model_test_lib",
    srcs = [
        "saved_model_test.py",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [":integration_scripts"],
)

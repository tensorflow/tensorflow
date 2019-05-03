# Example TensorFlow models that use DistributionStrategy for training.

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "simple_estimator_example",
    srcs = ["simple_estimator_example.py"],
    python_version = "PY2",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "keras_model_with_estimator",
    srcs = [
        "keras_model_with_estimator.py",
    ],
    python_version = "PY2",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "keras_mnist",
    srcs = ["keras_mnist.py"],
    python_version = "PY2",
    deps = [":keras_mnist_lib"],
)

py_library(
    name = "keras_mnist_lib",
    srcs = [
        "keras_mnist.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "mnist_eager_multigpu",
    srcs = [
        "mnist_eager_multigpu.py",
    ],
    python_version = "PY2",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "mnist_tf1_tpu",
    srcs = [
        "mnist_tf1_tpu.py",
    ],
    python_version = "PY2",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

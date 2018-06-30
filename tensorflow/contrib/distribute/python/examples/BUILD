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
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "simple_tfkeras_example",
    srcs = [
        "simple_tfkeras_example.py",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

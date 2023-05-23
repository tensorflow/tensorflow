# Example Estimator model

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "mnist_tflite",
    srcs = [
        "dataset.py",
        "mnist_tflite.py",
    ],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

filegroup(
    name = "tensorboard_frontend",
    srcs = [
        "dist/index.html",
        "dist/tf-tensorboard.html",
        "//tensorflow/tensorboard/bower:bower",
    ] + glob(["lib/**/*"]),
)

py_library(
    name = "tensorboard_handler",
    srcs = ["tensorboard_handler.py"],
    deps = [
        ":float_wrapper",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
)

py_library(
    name = "float_wrapper",
    srcs = ["float_wrapper.py"],
)

py_test(
    name = "float_wrapper_test",
    size = "small",
    srcs = ["float_wrapper_test.py"],
    deps = [
        ":float_wrapper",
        "//tensorflow/python:platform_test",
    ],
)

py_binary(
    name = "tensorboard",
    srcs = ["tensorboard.py"],
    data = [":tensorboard_frontend"],
    deps = [
        ":tensorboard_handler",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
)

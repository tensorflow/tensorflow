# Description:
# TensorBoard, a dashboard for investigating TensorFlow

package(default_visibility = ["//tensorflow:internal"])

filegroup(
    name = "tensorboard_frontend",
    srcs = [
        "dist/index.html",
        "dist/tf-tensorboard.html",
        "//tensorflow/tensorboard/bower:bower",
        "TAG",
    ] + glob(["lib/**/*"]),
)

py_library(
    name = "tensorboard_handler",
    srcs = ["backend/tensorboard_handler.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":float_wrapper",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "tensorboard_handler_test",
    size = "small",
    srcs = ["backend/tensorboard_handler_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensorboard_handler",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "float_wrapper",
    srcs = ["backend/float_wrapper.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "float_wrapper_test",
    size = "small",
    srcs = ["backend/float_wrapper_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":float_wrapper",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "tensorboard_server",
    srcs = ["backend/tensorboard_server.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensorboard_handler",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
)

py_binary(
    name = "tensorboard",
    srcs = ["backend/tensorboard.py"],
    data = [":tensorboard_frontend"],
    srcs_version = "PY2AND3",
    deps = [
        ":tensorboard_server",
        "//tensorflow/python:platform",
    ],
)

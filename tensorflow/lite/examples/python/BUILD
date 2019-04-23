licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

py_binary(
    name = "label_image",
    srcs = ["label_image.py"],
    main = "label_image.py",
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/lite/python:lite",
    ],
)

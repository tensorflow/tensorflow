# TensorFlow API backwards compatibility tests.

package(
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

exports_files([
    "README.txt",
    "API_UPDATE_WARNING.txt",
])

py_test(
    name = "api_compatibility_test",
    size = "small",
    srcs = ["api_compatibility_test.py"],
    data = [
        "//tensorflow/tools/api/golden:api_golden",
        "//tensorflow/tools/api/tests:API_UPDATE_WARNING.txt",
        "//tensorflow/tools/api/tests:README.txt",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:lib",
        "//tensorflow/python:platform",
        "//tensorflow/tools/api/lib:python_object_to_proto_visitor",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

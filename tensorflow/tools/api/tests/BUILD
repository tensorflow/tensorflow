# TensorFlow API backwards compatibility tests.

package(
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

py_test(
    name = "api_compatibility_test",
    srcs = ["api_compatibility_test.py"],
    srcs_version = "PY2AND3",
    data = [
        "//tensorflow/tools/api/golden:api_golden",
    ],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/tools/api/lib:python_object_to_proto_visitor",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
        "@protobuf//:protobuf_python",
    ],
)

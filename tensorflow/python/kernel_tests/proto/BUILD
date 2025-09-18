# Tests of tf.io.*proto.

load("@rules_python//python:proto.bzl", "py_proto_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "if_oss", "tf_cc_shared_object")
load("//tensorflow:tensorflow.default.bzl", "stripped_cc_info", "tf_py_strict_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_additional_all_protos", "tf_proto_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_pywrap")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "test_example.proto",
])

tf_py_strict_test(
    name = "decode_proto_op_test",
    size = "small",
    srcs = ["decode_proto_op_test.py"],
    data = if_oss([":libtestexample.so"]),
    tags = [
        "no_pip",  # TODO(b/78026780)
        "no_windows",  # TODO(b/78028010)
    ],
    deps = [
        ":decode_proto_op_test_base",
        ":py_test_deps",
        "//tensorflow/python/ops:proto_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "encode_proto_op_test",
    size = "small",
    srcs = ["encode_proto_op_test.py"],
    data = if_oss([":libtestexample.so"]),
    tags = [
        "no_pip",  # TODO(b/78026780)
        "no_windows",  # TODO(b/78028010)
    ],
    deps = [
        ":encode_proto_op_test_base",
        ":py_test_deps",
        "//tensorflow/python/ops:proto_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "proto_op_test_base",
    testonly = 1,
    srcs = ["proto_op_test_base.py"],
    deps = [
        ":test_example_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "decode_proto_op_test_base",
    testonly = 1,
    srcs = ["decode_proto_op_test_base.py"],
    deps = [
        ":proto_op_test_base",
        ":test_example_proto_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "encode_proto_op_test_base",
    testonly = 1,
    srcs = ["encode_proto_op_test_base.py"],
    deps = [
        ":proto_op_test_base",
        ":test_example_proto_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:array_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(name = "py_test_deps")

tf_proto_library(
    name = "test_example_proto",
    srcs = ["test_example.proto"],
    protodeps = tf_additional_all_protos(),
)

tf_cc_shared_object(
    name = "libtestexample.so",
    linkstatic = 1,
    deps = if_pywrap(
        if_false = [
            ":test_example_proto_cc",
        ],
        if_true = [
            "//tensorflow/python:tensorflow_common_framework",
            ":test_example_proto_cc_stripped",
        ],
    ),
)

stripped_cc_info(
    name = "test_example_proto_cc_stripped",
    deps = [":test_example_proto_cc"],
)

py_strict_library(
    name = "descriptor_source_test_base",
    testonly = 1,
    srcs = ["descriptor_source_test_base.py"],
    deps = [
        ":proto_op_test_base",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@com_google_protobuf//:protobuf_python",
    ],
)

tf_py_strict_test(
    name = "descriptor_source_test",
    size = "small",
    srcs = ["descriptor_source_test.py"],
    tags = [
        "no_pip",
    ],
    deps = [
        ":descriptor_source_test_base",
        "//tensorflow/python/ops:proto_ops",
        "//tensorflow/python/platform:client_testlib",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "test_example_proto_py",
#     deps = [":test_example_proto"],
# )
# copybara:uncomment_end

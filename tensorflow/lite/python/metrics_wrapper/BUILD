load("//tensorflow:tensorflow.bzl", "pybind_extension")
load("//tensorflow:tensorflow.bzl", "pytype_strict_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable", "if_portable")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "metrics_wrapper_lib",
    srcs = if_portable(
        if_false = ["metrics_wrapper_nonportable.cc"],
        if_true = ["metrics_wrapper_portable.cc"],
    ),
    hdrs = ["metrics_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        "//third_party/python_runtime:headers",
    ] + if_portable(
        if_false = [
            "//learning/brain/google/monitoring:metrics_exporter",
        ],
        if_true = [],
    ),
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_metrics_wrapper",
    srcs = ["metrics_wrapper_pybind11.cc"],
    hdrs = ["metrics_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    module_name = "_pywrap_tensorflow_lite_metrics_wrapper",
    visibility = ["//visibility:private"],
    deps = [
        ":metrics_wrapper_lib",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@com_google_protobuf//:protobuf",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "metrics_wrapper",
    srcs = ["metrics_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_tensorflow_lite_metrics_wrapper",
        ":converter_error_data_proto_py",
        "//tensorflow/lite/python:wrap_toco",
    ],
)

py_test(
    name = "metrics_wrapper_test",
    srcs = ["metrics_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics_wrapper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_proto_library(
    name = "converter_error_data_proto",
    srcs = ["converter_error_data.proto"],
    cc_api_version = 2,
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "converter_error_data_proto_py",
#     api_version = 2,
#     deps = [":converter_error_data_proto"],
# )
# copybara:uncomment_end

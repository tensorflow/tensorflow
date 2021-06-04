load("//tensorflow:tensorflow.bzl", "pybind_extension")

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
    deps = [
        "//third_party/python_runtime:headers",
    ] + if_portable(
        if_false = ["//learning/brain/google/monitoring:metrics_exporter"],
        if_true = [],
    ),
)

pybind_extension(
    name = "_pywrap_tensorflow_lite_metrics_wrapper",
    srcs = ["metrics_wrapper_pybind11.cc"],
    hdrs = ["metrics_wrapper.h"],
    compatible_with = get_compatible_with_portable(),
    link_in_framework = True,
    module_name = "_pywrap_tensorflow_lite_metrics_wrapper",
    deps = [
        ":metrics_wrapper_lib",
        "//tensorflow/python:pybind11_lib",
        "//third_party/python_runtime:headers",
        "@pybind11",
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

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
        if_false = ["wrapper/metrics_wrapper_nonportable.cc"],
        if_true = ["wrapper/metrics_wrapper_portable.cc"],
    ),
    hdrs = ["wrapper/metrics_wrapper.h"],
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
    srcs = ["wrapper/metrics_wrapper_pybind11.cc"],
    hdrs = ["wrapper/metrics_wrapper.h"],
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
    srcs = ["wrapper/metrics_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        ":_pywrap_tensorflow_lite_metrics_wrapper",
        ":converter_error_data_proto_py",
        "//tensorflow/lite/python:wrap_toco",
    ],
)

py_test(
    name = "metrics_wrapper_test",
    srcs = ["wrapper/metrics_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics_wrapper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

pytype_strict_library(
    name = "metrics_interface",
    srcs = ["metrics_interface.py"],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = ["//visibility:private"],
)

genrule(
    name = "metrics_py_gen",
    srcs = if_portable(
        if_false = ["metrics_nonportable.py"],
        if_true = ["metrics_portable.py"],
    ),
    outs = ["metrics.py"],
    cmd = (
        "cat $(SRCS) > $(OUTS)"
    ),
    compatible_with = get_compatible_with_portable(),
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    compatible_with = get_compatible_with_portable(),
    srcs_version = "PY3",
    visibility = ["//tensorflow/lite:__subpackages__"],
    deps = if_portable(
        if_false = [
            ":converter_error_data_proto_py",
            ":metrics_wrapper",
            "//tensorflow/python/eager:monitoring",
        ],
        if_true = [],
    ) + [":metrics_interface"],
)

py_test(
    name = "metrics_test",
    srcs = if_portable(
        if_false = ["metrics_nonportable_test.py"],
        if_true = ["metrics_portable_test.py"],
    ),
    data = [
        "//tensorflow/lite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    main = if_portable(
        if_false = "metrics_nonportable_test.py",
        if_true = "metrics_portable_test.py",
    ),
    python_version = "PY3",
    deps = [
        ":converter_error_data_proto_py",
        ":metrics",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
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

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "if_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tensorflow_lite_metrics_wrapper.pyi",
    ],
    visibility = [
        "__subpackages__",
        "//tensorflow/tools/pip_package:__subpackages__",
    ],
    deps = [
        ":metrics_wrapper_lib",
        "//tensorflow/python/lib/core:pybind11_lib",
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
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/compiler/mlir/lite/python:wrap_converter",
    ],
)

py_strict_test(
    name = "metrics_wrapper_test",
    srcs = ["wrapper/metrics_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics_wrapper",
        #internal proto upb dep
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:convert",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
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
            "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
            ":metrics_wrapper",
            "//tensorflow/python/eager:monitoring",
        ],
        if_true = [],
    ) + [":metrics_interface"],
)

py_strict_test(
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
        ":metrics",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/lite/python:convert",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/framework:convert_to_constants",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:importer",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/trackable:autotrackable",
    ],
)

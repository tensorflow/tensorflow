# TensorFlow ops for audio front-end processing.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_opts_nortti_if_android",
)
load("//tensorflow:tensorflow.default.bzl", "tf_custom_op_py_strict_library", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "audio_microfrontend",
    srcs = ["audio_microfrontend.cc"],
    hdrs = ["audio_microfrontend.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:reference",
        "@flatbuffers",
    ],
)

cc_library(
    name = "audio_microfrontend_op_lib",
    srcs = ["ops/audio_microfrontend_op.cc"],
    copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_android() + [
        "-Wno-narrowing",
        "-Wno-sign-compare",
        "-Wno-overloaded-virtual",
    ] + select({
        "//tensorflow:android": [
            # Selective registration uses constexprs with recursive
            # string comparisons; that can lead to compiler errors, so
            # we increase the constexpr recursion depth.
            "-fconstexpr-depth=1024",
            "-Oz",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib_lite",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "audio_microfrontend_test",
    size = "small",
    srcs = ["audio_microfrontend_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":audio_microfrontend",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tf_custom_op_library(
    name = "python/ops/_audio_microfrontend_op.so",
    srcs = [
        "ops/audio_microfrontend_op.cc",
    ],
    deps = [
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["audio_microfrontend_op"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
    ],
)

tf_gen_op_wrapper_py(
    name = "audio_microfrontend_op",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":audio_microfrontend_op_op_lib"],
)

tf_custom_op_py_strict_library(
    name = "audio_microfrontend_py",
    srcs = [
        "python/ops/audio_microfrontend_op.py",
    ],
    dso = [":python/ops/_audio_microfrontend_op.so"],
    kernels = [
        ":audio_microfrontend_op_op_lib",
    ],
    deps = [
        ":audio_microfrontend_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:load_library",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
    ],
)

tf_py_strict_test(
    name = "audio_microfrontend_op_test",
    size = "small",
    srcs = ["python/kernel_tests/audio_microfrontend_op_test.py"],
    deps = [
        ":audio_microfrontend_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:ops",
    ],
)

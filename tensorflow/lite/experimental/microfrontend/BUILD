# TensorFlow ops for audio front-end processing.

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_opts_nortti_if_android",
    "tf_py_test",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "audio_microfrontend",
    srcs = ["audio_microfrontend.cc"],
    hdrs = ["audio_microfrontend.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/experimental/microfrontend/lib:frontend",
        "//tensorflow/lite/kernels:kernel_util",
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
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
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
    deps = [":audio_microfrontend_op_op_lib"],
)

tf_custom_op_py_library(
    name = "audio_microfrontend_py",
    srcs = [
        "python/ops/audio_microfrontend_op.py",
    ],
    dso = [":python/ops/_audio_microfrontend_op.so"],
    kernels = [
        ":audio_microfrontend_op_op_lib",
    ],
    srcs_version = "PY3",
    deps = [
        ":audio_microfrontend_op",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "audio_microfrontend_op_test",
    size = "small",
    srcs = ["python/kernel_tests/audio_microfrontend_op_test.py"],
    tags = ["no_pip"],
    deps = [
        ":audio_microfrontend_py",
        "//tensorflow:tensorflow_py",
    ],
)

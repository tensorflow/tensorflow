package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow/contrib/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

py_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    data = [
        "//tensorflow/contrib/lite/schema:schema.fbs",
        "@flatbuffers//:flatc",
    ],
    srcs_version = "PY2AND3",
)

tf_cc_binary(
    name = "generate_op_registrations",
    srcs = ["gen_op_registration_main.cc"],
    deps = [
        "//tensorflow/contrib/lite/tools:gen_op_registration",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_binary(
    name = "benchmark_model",
    srcs = ["benchmark_model.cc"],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-landroid",
            "-lm",
            "-z defs",
            "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":mutable_op_resolver",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/core:android_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
        ],
    }),
)

cc_library(
    name = "gen_op_registration",
    srcs = ["gen_op_registration.cc"],
    hdrs = ["gen_op_registration.h"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:string",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "gen_op_registration_test",
    srcs = ["gen_op_registration_test.cc"],
    data = [
        "//tensorflow/contrib/lite:testdata/0_subgraphs.bin",
        "//tensorflow/contrib/lite:testdata/2_subgraphs.bin",
        "//tensorflow/contrib/lite:testdata/empty_model.bin",
        "//tensorflow/contrib/lite:testdata/test_model.bin",
        "//tensorflow/contrib/lite:testdata/test_model_broken.bin",
    ],
    deps = [
        ":gen_op_registration",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "mutable_op_resolver",
    srcs = ["mutable_op_resolver.cc"],
    hdrs = ["mutable_op_resolver.h"],
    deps = ["//tensorflow/contrib/lite:framework"],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    deps = [
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_test(
    name = "verifier_test",
    size = "small",
    srcs = ["verifier_test.cc"],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":mutable_op_resolver",
        ":verifier",
        "//tensorflow/contrib/lite:framework",
        "//tensorflow/contrib/lite:schema_fbs_version",
        "//tensorflow/contrib/lite:string_util",
        "//tensorflow/contrib/lite/schema:schema_fbs",
        "//tensorflow/contrib/lite/testing:util",
        "//tensorflow/core:framework_lite",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()

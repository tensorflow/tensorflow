package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_protos_all")

VECTORIZER_DEPS = [
    ":vectorizer_registry",
    "//tensorflow/cc:ops",
    "//tensorflow/core/grappler/optimizers/data:graph_utils",
    "//tensorflow/core:core_cpu",
    "//tensorflow/cc:scope_internal",
    "//tensorflow/cc:math_ops",
    "//tensorflow/cc:array_ops",
    "//tensorflow/cc:const_op",
] + tf_protos_all()

cc_library(
    name = "wrapped_tensor",
    hdrs = ["wrapped_tensor.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "vectorizer",
    hdrs = ["vectorizer.h"],
    deps = [
        ":wrapped_tensor",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
    ] + tf_protos_all(),
)

cc_library(
    name = "vectorizer_registry",
    srcs = ["vectorizer_registry.cc"],
    hdrs = ["vectorizer_registry.h"],
    deps = [
        ":vectorizer",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "vectorizer_registry_test",
    srcs = ["vectorizer_registry_test.cc"],
    deps = [
        ":vectorizer_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ] + tf_protos_all(),
)

cc_library(
    name = "add_vectorizer",
    srcs = [
        "add_vectorizer.cc",
    ],
    deps = VECTORIZER_DEPS,
    alwayslink = 1,
)

cc_library(
    name = "cast_vectorizer",
    srcs = ["cast_vectorizer.cc"],
    deps = VECTORIZER_DEPS,
    alwayslink = 1,
)

cc_library(
    name = "decode_csv_vectorizer",
    srcs = ["decode_csv_vectorizer.cc"],
    deps = VECTORIZER_DEPS,
    alwayslink = 1,
)

cc_library(
    name = "parse_single_example_vectorizer",
    srcs = ["parse_single_example_vectorizer.cc"],
    deps = VECTORIZER_DEPS,
    alwayslink = 1,
)

cc_library(
    name = "reshape_vectorizer",
    srcs = ["reshape_vectorizer.cc"],
    deps = VECTORIZER_DEPS,
    alwayslink = 1,
)

cc_library(
    name = "unpack_vectorizer",
    srcs = ["unpack_vectorizer.cc"],
    deps = VECTORIZER_DEPS,
    alwayslink = 1,
)

cc_library(
    name = "vectorization",
    hdrs = ["vectorizer_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":add_vectorizer",
        ":cast_vectorizer",
        ":decode_csv_vectorizer",
        ":parse_single_example_vectorizer",
        ":reshape_vectorizer",
        ":unpack_vectorizer",
        ":vectorizer",
        ":vectorizer_registry",
    ],
)

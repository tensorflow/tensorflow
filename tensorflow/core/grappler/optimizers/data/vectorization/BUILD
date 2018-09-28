package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_protos_all")

VECTORIZER_DEPS = [
    ":vectorizer_registry",
    "//tensorflow/core/grappler/optimizers/data:graph_utils",
] + tf_protos_all()

cc_library(
    name = "vectorizer",
    hdrs = ["vectorizer.h"],
    deps = [
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

cc_library(
    name = "cast_vectorizer",
    srcs = ["cast_vectorizer.cc"],
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
        ":cast_vectorizer",
        ":unpack_vectorizer",
        ":vectorizer",
        ":vectorizer_registry",
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

"""Fuzzing template for TensorFlow ops."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")

def tf_ops_fuzz_target_lib(name):
    cc_library(
        name = name + "_fuzz_lib",
        srcs = [name + "_fuzz.cc"],
        deps = [
            "//tensorflow/core/kernels/fuzzing:fuzz_session",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/cc:ops",
            "//tensorflow/cc:scope",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//third_party/absl/log",
        ],
        tags = [
            "manual",
            "no_windows",
        ],
        alwayslink = 1,
        visibility = ["//visibility:public"],
    )

def tf_oss_fuzz_corpus(name):
    native.filegroup(
        name = name + "_corpus",
        srcs = native.glob(["corpus/" + name + "/*"]),
    )

def tf_oss_fuzz_dict(name):
    native.filegroup(
        name = name + "_dict",
        srcs = native.glob(["dictionaries/" + name + ".dict"]),
    )

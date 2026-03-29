"""Fuzzing template for TensorFlow ops."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//tensorflow/security/fuzzing:tf_fuzzing.bzl", "tf_cc_fuzz_test")

def tf_ops_fuzz_target_lib(name):
    cc_library(
        name = name + "_fuzz_lib",
        srcs = [name + "_fuzz.cc"],
        deps = [
            "//tensorflow/core/kernels/fuzzing:fuzz_session",
            "//tensorflow/cc:cc_ops",
            "//tensorflow/core/framework:types_proto_cc",
        ],
        tags = [
            "manual",
            "no_windows",
        ],
        alwayslink = 1,
    )

    tf_cc_fuzz_test(
        name = name + "_fuzz",
        deps = [
            ":" + name + "_fuzz_lib",
        ],
        tags = [
            "no_windows",
        ],
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

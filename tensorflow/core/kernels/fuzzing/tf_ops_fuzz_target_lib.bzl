"""Fuzzing template for TensorFlow ops."""

def tf_ops_fuzz_target_lib(name):
    native.cc_library(
        name = name + "_fuzz_lib",
        srcs = [name + "_fuzz.cc"],
        deps = [
            "//tensorflow/core/kernels/fuzzing:fuzz_session",
            "//tensorflow/cc:cc_ops",
        ],
        tags = ["no_windows"],
        alwayslink = 1,
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

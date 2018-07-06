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

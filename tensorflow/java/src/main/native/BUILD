# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Java API using the TensorFlow C library.

package(default_visibility = ["//tensorflow/java:__pkg__"])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")

tf_cuda_library(
    name = "native",
    srcs = glob(["*.cc"]) + [
        ":jni.h",
        ":jni_md.h",
    ],
    hdrs = glob(["*.h"]),
    includes = ["."],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:ops",
    ],
    alwayslink = 1,
)

# Silly rules to make
# #include <jni.h>
# in the source headers work
# (in combination with the "includes" attribute of the tf_cuda_library rule
# above).
#
# Inspired from:
# https://github.com/bazelbuild/bazel/blob/f99a0543f8d97339d32075c7176b79f35be84606/src/main/native/BUILD
# but hopefully there is a simpler alternative to this.
#
# TODO(ashankar): This should not be necessary for Android builds as the
# toolchain makes <jni.h> available. Perhaps remove ":jni.h" and ":jni_md.h"
# from "srcs" and make these genrules a no-op when building for Android?
genrule(
    name = "copy_jni_h",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_h",
    srcs = select({
        "//tensorflow:darwin": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

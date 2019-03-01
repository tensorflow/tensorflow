package(default_visibility = ["//tensorflow/lite:__subpackages__"])

licenses(["notice"])  # Apache 2.0

# Helper target for exposing JNI headers across multiple platforms.
cc_library(
    name = "jni",
    hdrs = select({
        # The Android toolchain makes "jni.h" available in the include path.
        # For non-Android toolchains, generate jni.h and jni_md.h.
        "//tensorflow:android": [],
        "//conditions:default": [
            ":jni.h",
            ":jni_md.h",
        ],
    }),
    includes = select({
        "//tensorflow:android": [],
        "//conditions:default": ["."],
    }),
)

# Silly rules to make
# #include <jni.h>
# in the source headers work
# (in combination with the "includes" attribute of the tf_cuda_library rule
# above. Not needed when using the Android toolchain).
#
# Inspired from:
# https://github.com/bazelbuild/bazel/blob/f99a0543f8d97339d32075c7176b79f35be84606/src/main/native/BUILD
# but hopefully there is a simpler alternative to this.
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

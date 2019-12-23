# For workaround https://github.com/bazelbuild/bazel/issues/8772 with Bazel >= 0.29.1
# TensorFlow still targets Java 1.7 (See JAVACOPTS in tensorflow/java/build_defs.bzl)
# which doesn't support "-parameters" flag. Starting from Java 11 (default since Bazel
# 0.29.1), a warning message will be thrown if "-parameters" is passed. If "-Werror" also exists,
# the compiling action will fail. To workaround this, we override the misc value of
# the default java toolchain to remove "-parameters" flag.
load("@bazel_tools//tools/jdk:default_java_toolchain.bzl", "default_java_toolchain")

licenses(["notice"])

default_java_toolchain(
    name = "tf_java_toolchain",
    misc = [
        "-XDskipDuplicateBridges=true",
        "-g",
    ],
    visibility = ["//visibility:public"],
)

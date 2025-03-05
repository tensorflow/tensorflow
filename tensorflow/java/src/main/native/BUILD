# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Java API using the TensorFlow C library.
#
# In the opensource world, more machinery is needed
# to build libraries with JNI (see BUILD.bazel).
# Since that gets complicated, the native code here is
# placed in its own package (and BUILD file) instead of
# in third_party/tensorflow/java/BUILD.

load("//tensorflow:tensorflow.bzl", "tf_copts", "tf_cuda_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/java:__pkg__",
        # TODO(ashankar): Temporary hack for the Java API and
        # //third_party/tensorflow/tools/android/inference_interface:android_tensorflow_inference_jni
        # to co-exist in a single shared library. However, the hope is that
        # //third_party/tensorflow/tools/android/inference_interface:android_tensorflow_jni can be
        # removed once the Java API provides feature parity with it.
        "//tensorflow/tools/android/inference_interface:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "native_srcs",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    visibility = ["//visibility:public"],
)

tf_cuda_library(
    name = "native",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
    copts = tf_copts(),
    features = select({
        "//tensorflow:android": ["-layering_check"],
        "//conditions:default": [],
    }),
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/c:c_api",
            "//tensorflow/c/eager:c_api",
            "//tensorflow/core:all_kernels",
            "//tensorflow/core:direct_session",
            "//tensorflow/core:ops",
            "//third_party/java/jdk:jni",
        ],
    }),
    alwayslink = 1,
)

# Description:
#   JNI-based Java inference interface for TensorFlow.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_android",
    "tf_cc_binary",  # @unused
    "tf_copts",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "jni/version_script.lds",
])

filegroup(
    name = "android_tensorflow_inference_jni_srcs",
    srcs = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "android_tensorflow_inference_jni",
    srcs = if_android([":android_tensorflow_inference_jni_srcs"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:portable_tensorflow_lib_lite",
        "//tensorflow/java/src/main/native",
    ],
    alwayslink = 1,
)

# JAR with Java bindings to TF.
android_library(
    name = "android_tensorflow_inference_java",
    srcs = glob(["java/**/*.java"]) + ["//tensorflow/java:java_sources"],
    javacopts = [
        "-XepOpt:CheckReturnValue:CheckAllConstructors=false",  # b/226969262
    ],
    tags = [
        "manual",
        "notap",
    ],
)

# Build the native .so.
# bazel build //tensorflow/tools/android/inference_interface:libtensorflow_inference.so \
#   --crosstool_top=//external:android/crosstool \
#   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#   --cpu=armeabi-v7a
LINKER_SCRIPT = "//tensorflow/tools/android/inference_interface:jni/version_script.lds"

# This fails to buiild if converted to tf_cc_binary.
cc_binary(
    name = "libtensorflow_inference.so",
    copts = tf_copts() + [
        "-ffunction-sections",
        "-fdata-sections",
    ],
    linkopts = if_android([
        "-landroid",
        "-latomic",
        "-ldl",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--gc-sections",
        "-Wl,--version-script,$(location {})".format(LINKER_SCRIPT),
    ]),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":android_tensorflow_inference_jni",
        "//tensorflow/core:portable_tensorflow_lib",
        LINKER_SCRIPT,
    ],
)

cc_library(
    name = "android_tensorflow_inference_native",
    srcs = if_android([":libtensorflow_inference.so"]),
)

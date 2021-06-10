# Description:
# Java Native Interface (JNI) library for testing the TensorFlow Lite Java API.

load(
    "//tensorflow/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite",
    "jni_binary_with_tflite",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "native",
    testonly = 1,
    srcs = [
        "interpreter_test_jni.cc",
        "supported_features_jni.cc",
        "test_init_jni.cc",
    ],
    tflite_deps = [
        "//tensorflow/lite/core/shims:c_shims_test_util",
        "//tensorflow/lite/java/src/main/native",
        "//tensorflow/lite/java/src/main/native:native_framework_only",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/nnapi/java/src/main/native",
        "//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

jni_binary_with_tflite(
    name = "libtensorflowlite_test_jni.so",
    testonly = 1,
    linkscript = "//tensorflow/lite/java:tflite_version_script.lds",
    tflite_deps = [":native"],
)

# Dummy native library which doesn't actually contain the TFLite implementation.
jni_binary_with_tflite(
    name = "libtensorflowlite_jni.so",
    testonly = 1,
)

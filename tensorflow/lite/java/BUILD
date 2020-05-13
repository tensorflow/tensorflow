# Description:
# TensorFlow Lite Java API.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "src/testdata/add.bin",
    "src/testdata/add_unknown_dimensions.bin",
    "src/testdata/grace_hopper_224.jpg",
])

JAVA_SRCS = glob([
    "src/main/java/org/tensorflow/lite/*.java",
    "src/main/java/org/tensorflow/lite/annotations/*.java",
]) + ["//tensorflow/lite/delegates/nnapi/java/src/main/java/org/tensorflow/lite/nnapi:nnapi_delegate_src"]

# Building tensorflow-lite.aar including 4 variants of .so
# To build an aar for release, run below command:
# bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
#   tensorflow/lite/java:tensorflow-lite
aar_with_jni(
    name = "tensorflow-lite",
    android_library = ":tensorflowlite",
    headers = [
        "//tensorflow/lite:builtin_ops.h",
        "//tensorflow/lite/c:c_api.h",
        "//tensorflow/lite/c:c_api_experimental.h",
        "//tensorflow/lite/c:common.h",
    ],
)

# EXPERIMENTAL: AAR target for using TensorFlow ops with TFLite. Note that this
# .aar contains *only* the Flex delegate for using select tf ops; clients must
# also include the core `tensorflow-lite` runtime.
aar_with_jni(
    name = "tensorflow-lite-select-tf-ops",
    android_library = ":tensorflowlite_flex",
)

# EXPERIMENTAL: AAR target for GPU acceleration. Note that this .aar contains
# *only* the GPU delegate; clients must also include the core `tensorflow-lite`
# runtime.
aar_with_jni(
    name = "tensorflow-lite-gpu",
    android_library = ":tensorflowlite_gpu",
    headers = [
        "//tensorflow/lite/delegates/gpu:delegate.h",
    ],
)

android_library(
    name = "tensorflowlite",
    srcs = JAVA_SRCS,
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        ":tensorflowlite_native",
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Android target that supports TensorFlow op execution with TFLite.
# Note that this library contains *only* the Flex delegate and its Java wrapper for using
# select TF ops; clients must also include the core `tensorflowlite` runtime.
android_library(
    name = "tensorflowlite_flex",
    srcs = ["//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        ":tensorflowlite_java",
        ":tensorflowlite_native_flex",
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Android target target for GPU acceleration. Note that this
# library contains *only* the GPU delegate and its Java wrapper; clients must
# also include the core `tensorflowlite` runtime.
android_library(
    name = "tensorflowlite_gpu",
    srcs = ["//tensorflow/lite/delegates/gpu/java/src/main/java/org/tensorflow/lite/gpu:gpu_delegate"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        ":tensorflowlite_java",
        ":tensorflowlite_native_gpu",
        "@org_checkerframework_qual",
    ],
)

android_library(
    name = "tensorflowlite_java",
    srcs = JAVA_SRCS,
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "tensorflowlite_javalib",
    srcs = JAVA_SRCS,
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "tensorflowlitelib",
    srcs = JAVA_SRCS,
    javacopts = JAVACOPTS,
    deps = [
        ":libtensorflowlite_jni.so",
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Java target that supports TensorFlow op execution with TFLite.
java_library(
    name = "tensorflowlitelib_flex",
    srcs = ["//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    deps = [
        ":libtensorflowlite_flex_jni.so",
        ":tensorflowlitelib",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "TensorFlowLiteTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/TensorFlowLiteTest.java"],
    javacopts = JAVACOPTS,
    tags = [
        "no_mac",  # TODO(b/122888913): libtensorflowlite_test_jni broke on mac.
        "v1only",
    ],
    test_class = "org.tensorflow.lite.TensorFlowLiteTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "TensorFlowLiteNoNativeLibTest",
    size = "small",
    srcs = JAVA_SRCS + ["src/test/java/org/tensorflow/lite/TensorFlowLiteNoNativeLibTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.TensorFlowLiteNoNativeLibTest",
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_truth",
        "@junit",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "TensorFlowLiteInvalidNativeLibTest",
    size = "small",
    srcs = JAVA_SRCS + ["src/test/java/org/tensorflow/lite/TensorFlowLiteInvalidNativeLibTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.tensorflow.lite.TensorFlowLiteInvalidNativeLibTest",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/lite/java/src/test/native:libtensorflowlite_jni.so",
        "@com_google_truth",
        "@junit",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "DataTypeTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/DataTypeTest.java"],
    javacopts = JAVACOPTS,
    tags = [
        "no_mac",  # TODO(b/122888913): libtensorflowlite_test_jni broke on mac.
    ],
    test_class = "org.tensorflow.lite.DataTypeTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "NativeInterpreterWrapperTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/NativeInterpreterWrapperTest.java"],
    data = [
        # The files named as <data_type>.bin reshape the incoming tensor from (2, 8, 8, 3) to (2, 4, 4, 12).
        "src/testdata/add.bin",
        "src/testdata/int32.bin",
        "src/testdata/int64.bin",
        "src/testdata/invalid_model.bin",
        "src/testdata/string.bin",
        # Takes a scalar string and reshapes to a rank-1, single element string.
        "src/testdata/string_scalar.bin",
        "src/testdata/uint8.bin",
        "src/testdata/with_custom_op.lite",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_mac",  # TODO(b/122888913): libtensorflowlite_test_jni broke on mac.
    ],
    test_class = "org.tensorflow.lite.NativeInterpreterWrapperTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

# TODO: generate large models at runtime, instead of storing them.
java_test(
    name = "InterpreterTest",
    size = "small",
    srcs = [
        "src/test/java/org/tensorflow/lite/InterpreterTest.java",
        "src/test/java/org/tensorflow/lite/TestUtils.java",
    ],
    data = [
        "src/testdata/add.bin",
        "src/testdata/add_unknown_dimensions.bin",
        "//tensorflow/lite:testdata/dynamic_shapes.bin",
        "//tensorflow/lite:testdata/multi_add.bin",
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_mac",  # TODO(b/122888913): libtensorflowlite_test_jni broke on mac.
    ],
    test_class = "org.tensorflow.lite.InterpreterTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "//tensorflow/lite/java/src/test/native:libtensorflowlite_test_jni.so",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "NnApiDelegateTest",
    size = "small",
    srcs = [
        "src/test/java/org/tensorflow/lite/TestUtils.java",
        "src/test/java/org/tensorflow/lite/nnapi/NnApiDelegateTest.java",
    ],
    data = [
        "src/testdata/add.bin",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_mac"],
    test_class = "org.tensorflow.lite.nnapi.NnApiDelegateTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "InterpreterFlexTest",
    size = "small",
    srcs = [
        "src/test/java/org/tensorflow/lite/InterpreterFlexTest.java",
        "src/test/java/org/tensorflow/lite/TestUtils.java",
    ],
    data = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_cuda_on_cpu_tap",  # CUDA + flex is not officially supported.
        "no_gpu",  # GPU + flex is not officially supported.
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
        # TODO(b/121204962): Re-enable test after fixing memory leaks.
        "noasan",
    ],
    test_class = "org.tensorflow.lite.InterpreterFlexTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        ":tensorflowlitelib_flex",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "TensorTest",
    size = "small",
    srcs = ["src/test/java/org/tensorflow/lite/TensorTest.java"],
    data = [
        "src/testdata/add.bin",
        "src/testdata/int32.bin",
        "src/testdata/int64.bin",
        "src/testdata/quantized.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_mac",  # TODO(b/122888913): libtensorflowlite_test_jni broke on mac.
    ],
    test_class = "org.tensorflow.lite.TensorTest",
    visibility = ["//visibility:private"],
    deps = [
        ":tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

filegroup(
    name = "portable_tests",
    srcs = [
        "src/test/java/org/tensorflow/lite/InterpreterMobileNetTest.java",
        "src/test/java/org/tensorflow/lite/InterpreterTest.java",
        "src/test/java/org/tensorflow/lite/TensorFlowLiteTest.java",
        "src/test/java/org/tensorflow/lite/nnapi/NnApiDelegateTest.java",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "portable_flex_tests",
    srcs = [
        "src/test/java/org/tensorflow/lite/InterpreterFlexTest.java",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "portable_gpu_tests",
    srcs = [
        "src/test/java/org/tensorflow/lite/gpu/GpuDelegateTest.java",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libtensorflowlite_jni",
    srcs = select({
        "//conditions:default": [":libtensorflowlite_jni.so"],
    }),
    tags = [
        "no_mac",  # TODO(b/122888913): libtensorflowlite_test_jni broke on mac.
    ],
)

cc_library(
    name = "tensorflowlite_native",
    srcs = ["libtensorflowlite_jni.so"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorflowlite_native_flex",
    srcs = ["libtensorflowlite_flex_jni.so"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "tensorflowlite_native_gpu",
    srcs = ["libtensorflowlite_gpu_jni.so"],
    visibility = ["//visibility:private"],
)

tflite_jni_binary(
    name = "libtensorflowlite_jni.so",
    linkscript = ":tflite_version_script.lds",
    deps = [
        # Note that we explicitly include the C API here for convenience, as it
        # allows bundling of the C lib w/ AAR distribution.
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
        "//tensorflow/lite/delegates/nnapi/java/src/main/native",
        "//tensorflow/lite/java/src/main/native",
    ],
)

# EXPERIMENTAL: Native target that supports TensorFlow op execution with TFLite.
tflite_jni_binary(
    name = "libtensorflowlite_flex_jni.so",
    deps = [
        "//tensorflow/lite/delegates/flex/java/src/main/native",
    ],
)

# EXPERIMENTAL: Native target that supports GPU acceleration.
tflite_jni_binary(
    name = "libtensorflowlite_gpu_jni.so",
    linkscript = ":gpu_version_script.lds",
    deps = [
        "//tensorflow/lite/delegates/gpu/java/src/main/native",
    ],
)

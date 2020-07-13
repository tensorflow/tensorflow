load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_jni_library")

# Following targets are using for testing selective-built flex delegate
# in Java. Please don't use them for other purposes.

package(
    default_visibility = [
        "//tensorflow/lite/android:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tflite_flex_jni_library(
    name = "test",
    models = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
)

java_library(
    name = "test_tensorflowlitelib_flex",
    testonly = 1,
    srcs = ["//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":libtensorflowlite_flex_jni.so",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "SelectiveBuiltInterpreterFlexTest",
    size = "small",
    srcs = [
        "//tensorflow/lite/java:portable_flex_tests",
        "//tensorflow/lite/java:portable_test_utils",
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
        "notsan",  # TODO(b/158651814) Re-enable after fixing racing condition.
    ],
    test_class = "org.tensorflow.lite.InterpreterFlexTest",
    visibility = ["//visibility:private"],
    deps = [
        ":test_tensorflowlitelib_flex",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

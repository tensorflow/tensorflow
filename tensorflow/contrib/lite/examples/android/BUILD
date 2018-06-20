# Description:
#   TensorFlow camera demo app for Android.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# Build the demo native demo lib from the original directory to reduce code
# reuse. Note that the Java counterparts (ObjectTracker.java and
# ImageUtils.java) are still duplicated.
cc_library(
    name = "tensorflow_native_libs",
    srcs = [
        "//tensorflow/examples/android:libtensorflow_demo.so",
    ],
    tags = [
        "manual",
        "notap",
    ],
)

android_binary(
    name = "tflite_demo",
    srcs = glob([
        "app/src/main/java/**/*.java",
    ]),
    # Package assets from assets dir as well as all model targets.
    # Remove undesired models (and corresponding Activities in source)
    # to reduce APK size.
    assets = [
        "//tensorflow/contrib/lite/examples/android/app/src/main/assets:labels_mobilenet_quant_v1_224.txt",
        "@tflite_mobilenet//:mobilenet_quant_v1_224.tflite",
        "@tflite_conv_actions_frozen//:conv_actions_frozen.tflite",
        "//tensorflow/contrib/lite/examples/android/app/src/main/assets:conv_actions_labels.txt",
        "@tflite_mobilenet_ssd//:mobilenet_ssd.tflite",
        "//tensorflow/contrib/lite/examples/android/app/src/main/assets:box_priors.txt",
        "//tensorflow/contrib/lite/examples/android/app/src/main/assets:coco_labels_list.txt",
    ],
    assets_dir = "",
    custom_package = "org.tensorflow.lite.demo",
    inline_constants = 1,
    manifest = "app/src/main/AndroidManifest.xml",
    nocompress_extensions = [
        ".tflite",
    ],
    resource_files = glob(["app/src/main/res/**"]),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":tensorflow_native_libs",
        "//tensorflow/contrib/lite/java:tensorflowlite",
    ],
)

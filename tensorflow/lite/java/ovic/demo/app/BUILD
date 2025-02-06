# Sample app for OVIC benchmarking.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

android_binary(
    name = "ovic_benchmarker_binary",
    srcs = [
        "OvicBenchmarkerActivity.java",
    ],
    assets = [
        "//tensorflow/lite/java/ovic/src/testdata:coco_labels.txt",
        "//tensorflow/lite/java/ovic/src/testdata:labels.txt",
        "//tensorflow/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    assets_dir = "",
    custom_package = "ovic.demo.app",
    manifest = "AndroidManifest.xml",
    multidex = "legacy",
    nocompress_extensions = [
        ".lite",
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    tags = ["manual"],
    deps = [
        # copybara:uncomment "//third_party/java/android/android_sdk_linux/extras/android/compatibility/multidex",
        "//tensorflow/lite/java:tensorflowlite",
        "//tensorflow/lite/java/ovic:ovicbenchmarkerlib",
        "//tensorflow/lite/java/ovic:ovicdetectionbenchmarkerlib",
    ],
)

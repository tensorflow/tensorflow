load("@build_bazel_rules_android//android:rules.bzl", "android_binary")

# Sample app for OVIC benchmarking.
licenses(["notice"])  # Apache 2.0

android_binary(
    name = "ovic_benchmarker_binary",
    srcs = [
        "OvicBenchmarker.java",
        "OvicBenchmarkerActivity.java",
    ],
    assets = [
        "//tensorflow/contrib/lite/java/ovic/src/testdata:ovic_testdata",
        "//tensorflow/contrib/lite/java/ovic/src/testdata:labels.txt",
    ],
    assets_dir = "",
    custom_package = "ovic.demo.app",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = [
        ".lite",
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    tags = ["manual"],
    deps = [
        "//tensorflow/contrib/lite/java:tensorflowlite",
        "//tensorflow/contrib/lite/java/ovic:ovicbenchmarkerlib",
        "@androidsdk//com.android.support:support-v13-25.2.0",
        "@androidsdk//com.android.support:support-v4-25.2.0",
    ],
)

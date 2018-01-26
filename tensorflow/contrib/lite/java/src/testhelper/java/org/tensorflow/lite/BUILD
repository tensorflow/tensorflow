# Description:
# Internal helper function to test TF Lite API.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

android_library(
    name = "testhelper",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//tensorflow/contrib/lite/java:tensorflowlite_java",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

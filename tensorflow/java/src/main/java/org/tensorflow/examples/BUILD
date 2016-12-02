# Description:
# TensorFlow Java examples.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

java_binary(
    name = "example",
    srcs = ["Example.java"],
    main_class = "org.tensorflow.examples.Example",
    deps = ["//tensorflow/java:tensorflow"],
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

# Description:
# TensorFlow Java examples.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

java_binary(
    name = "label_image",
    srcs = ["LabelImage.java"],
    main_class = "org.tensorflow.examples.LabelImage",
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

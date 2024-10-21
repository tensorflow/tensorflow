# Description:
# TensorFlow Java examples.

load("@rules_java//java:defs.bzl", "java_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

java_binary(
    name = "label_image",
    srcs = ["LabelImage.java"],
    main_class = "org.tensorflow.examples.LabelImage",
    deps = ["//tensorflow/java:tensorflow"],
)

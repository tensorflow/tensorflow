# Description:
# TensorFlow Java examples.

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

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
            # Exclude any .class files that might have been generated
            # by manual invocations of javac as described in
            # tensorflow/java/README.md
            "**/*.class",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

# Description:
#   TensorFlow C++ inference example for labeling images.

load("//tensorflow:strict.default.bzl", "py_strict_binary")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

exports_files(["data/grace_hopper.jpg"])

tf_cc_binary(
    name = "label_image",
    srcs = [
        "main.cc",
    ],
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-landroid",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",
        ],
        "//conditions:default": ["-lm"],
    }),
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
            # cc:android_tensorflow_image_op is for including jpeg/gif/png
            # decoder to enable real-image evaluation on Android
            "//tensorflow/core/kernels/image:android_tensorflow_image_op",
        ],
        "//conditions:default": [
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:tensorflow",
        ],
    }) + [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "@local_xla//xla/tsl/util:command_line_flags",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:types",
        # cc:cc_ops is used to include image ops (for label_image)
        # Jpg, gif, and png related code won't be included
        "//tensorflow/cc:cc_ops",
    ],
)

py_strict_binary(
    name = "label_image_py",
    srcs = ["label_image.py"],
    main = "label_image.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

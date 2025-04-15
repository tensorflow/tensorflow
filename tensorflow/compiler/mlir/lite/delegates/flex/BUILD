load(
    "//tensorflow:tensorflow.bzl",
    "if_mobile",
    "if_not_mobile",
    "tf_features_nolayering_check_if_ios",
)
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:special_rules.bzl", "internal_visibility_allowlist")

default_visibility = [
    "//tensorflow/compiler/mlir/lite:__subpackages__",
    "//tensorflow/lite/android:__subpackages__",
    "//tensorflow/lite/toco/tflite:__subpackages__",
]

#
# This is a TF Lite delegate that is powered by TensorFlow's Eager.
#
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

cc_library(
    name = "allowlisted_flex_ops_lib",
    srcs = [
        "allowlisted_flex_ops.cc",
    ],
    hdrs = [
        "allowlisted_flex_ops.h",
        "allowlisted_flex_ops_internal.h",
    ],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    visibility = internal_visibility_allowlist(),
    deps = if_mobile([
        "//tensorflow/core:portable_tensorflow_lib_lite",
    ]) + if_not_mobile([
        "//tensorflow/core:framework",
    ]),
)

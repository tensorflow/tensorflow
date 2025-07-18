"""Provides project and wheel version data for TensorFlow."""

load(
    "//tensorflow:tf_version.default.bzl",
    "SEMANTIC_VERSION_SUFFIX",
    "VERSION_SUFFIX",
)

# These constants are used by the targets //third_party/tensorflow/core/public:release_version,
# //third_party/tensorflow:tensorflow_bzl and //third_party/tensorflow/tools/pip_package:setup_py.
TF_VERSION = "2.21.0"
MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = TF_VERSION.split(".")
TF_WHEEL_VERSION_SUFFIX = VERSION_SUFFIX
TF_SEMANTIC_VERSION_SUFFIX = SEMANTIC_VERSION_SUFFIX

"""Default (OSS) TensorFlow wheel version suffix data."""

load(
    "@tf_wheel_version_suffix//:wheel_version_suffix.bzl",
    "SEMANTIC_WHEEL_VERSION_SUFFIX",
    "WHEEL_VERSION_SUFFIX",
)

VERSION_SUFFIX = WHEEL_VERSION_SUFFIX
SEMANTIC_VERSION_SUFFIX = SEMANTIC_WHEEL_VERSION_SUFFIX

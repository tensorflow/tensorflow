"""Default (OSS) build versions of TSL general-purpose build extensions."""

load(
    "//tensorflow/tsl:tsl.bzl",
    _filegroup = "filegroup",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_not_mobile_or_arm_or_lgpl_restricted = "if_not_mobile_or_arm_or_lgpl_restricted",
)

get_compatible_with_portable = _get_compatible_with_portable
filegroup = _filegroup
if_not_mobile_or_arm_or_lgpl_restricted = _if_not_mobile_or_arm_or_lgpl_restricted

"""Default (OSS) build versions of TSL general-purpose build extensions."""

load(
    "//xla/tsl:tsl.bzl",
    _filegroup = "filegroup",
    _get_compatible_with_libtpu_portable = "get_compatible_with_libtpu_portable",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_not_mobile_or_arm_or_macos_or_lgpl_restricted = "if_not_mobile_or_arm_or_macos_or_lgpl_restricted",
    _internal_hlo_deps = "internal_hlo_deps",
    _tsl_extra_config_settings = "tsl_extra_config_settings",
    _tsl_extra_config_settings_targets = "tsl_extra_config_settings_targets",
    _tsl_google_bzl_deps = "tsl_google_bzl_deps",
    _tsl_grpc_cc_dependencies = "tsl_grpc_cc_dependencies",
    _tsl_pybind_extension = "tsl_pybind_extension",
)

get_compatible_with_portable = _get_compatible_with_portable
get_compatible_with_libtpu_portable = _get_compatible_with_libtpu_portable
filegroup = _filegroup
if_not_mobile_or_arm_or_macos_or_lgpl_restricted = _if_not_mobile_or_arm_or_macos_or_lgpl_restricted
internal_hlo_deps = _internal_hlo_deps
tsl_grpc_cc_dependencies = _tsl_grpc_cc_dependencies
tsl_pybind_extension = _tsl_pybind_extension
tsl_google_bzl_deps = _tsl_google_bzl_deps
tsl_extra_config_settings = _tsl_extra_config_settings
tsl_extra_config_settings_targets = _tsl_extra_config_settings_targets

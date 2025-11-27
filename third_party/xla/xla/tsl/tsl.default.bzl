"""Default (OSS) build versions of TSL general-purpose build extensions."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load(
    "//xla/tsl:tsl.bzl",
    _filegroup = "filegroup",
    _get_compatible_with_libtpu_portable = "get_compatible_with_libtpu_portable",
    _get_compatible_with_portable = "get_compatible_with_portable",
    _if_not_mobile_or_arm_or_macos_or_lgpl_restricted = "if_not_mobile_or_arm_or_macos_or_lgpl_restricted",
    _tsl_extra_config_settings = "tsl_extra_config_settings",
    _tsl_extra_config_settings_targets = "tsl_extra_config_settings_targets",
    _tsl_google_bzl_deps = "tsl_google_bzl_deps",
    _tsl_pybind_extension = "tsl_pybind_extension",
)

visibility(DEFAULT_LOAD_VISIBILITY)

get_compatible_with_portable = _get_compatible_with_portable
get_compatible_with_libtpu_portable = _get_compatible_with_libtpu_portable
filegroup = _filegroup
if_not_mobile_or_arm_or_macos_or_lgpl_restricted = _if_not_mobile_or_arm_or_macos_or_lgpl_restricted
tsl_pybind_extension = _tsl_pybind_extension
tsl_google_bzl_deps = _tsl_google_bzl_deps
tsl_extra_config_settings = _tsl_extra_config_settings
tsl_extra_config_settings_targets = _tsl_extra_config_settings_targets

# These configs are used to determine whether we should use CUDA/NVSHMEM tools and libs in
# cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hermetic CUDA tools."""
    return select({
        "@local_config_cuda//cuda:cuda_tools": if_true,
        "//conditions:default": if_false,
    })

selects.config_setting_group(
    name = "subprocess_compilation_enabled_with_cuda_tools",
    match_all = [
        ":subprocess_compilation_support_enabled",
        "@local_config_cuda//cuda:cuda_tools",
    ],
)

def if_cuda_tools_subprocess_compilation(if_both, if_cuda_no_subprocess, if_none = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing three-fold: with hermetic CUDA tools and subprocess compilation, with hermetic CUDA tools but no subprocess compilation, and neither (no CUDA tools, no subprocess compilation)."""
    return select({
        ":subprocess_compilation_enabled_with_cuda_tools": if_both,
        "@local_config_cuda//cuda:cuda_tools": if_cuda_no_subprocess,
        "//conditions:default": if_none,
    })

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({
        "@local_config_cuda//cuda:cuda_tools_and_libs": if_true,
        "//conditions:default": if_false,
    })

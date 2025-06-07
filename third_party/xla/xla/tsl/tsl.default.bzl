"""Default (OSS) build versions of TSL general-purpose build extensions."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
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

visibility(DEFAULT_LOAD_VISIBILITY)

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

# These configs are used to determine whether we should use CUDA/NVSHMEM tools and libs in
# cc_libraries.
# They are intended for the OSS builds only.
def if_cuda_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hermetic CUDA tools."""
    return select({
        "@local_config_cuda//cuda:cuda_tools": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic CUDA libraries."""
    return select({
        "@local_config_cuda//cuda:cuda_tools_and_libs": if_true,
        "//conditions:default": if_false,
    })

def if_nvshmem_tools(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we're building with hermetic NVSHMEM tools."""
    return select({
        "@local_config_nvshmem//:nvshmem_tools": if_true,
        "//conditions:default": if_false,
    })

def if_nvshmem_libs(if_true, if_false = []):  # buildifier: disable=unused-variable
    """Shorthand for select()'ing on whether we need to include hermetic NVSHMEM libraries."""
    return select({
        "@local_config_nvshmem//:nvshmem_tools_and_libs": if_true,
        "//conditions:default": if_false,
    })

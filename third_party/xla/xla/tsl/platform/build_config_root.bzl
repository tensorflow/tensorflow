"""Provides a redirection point for platform specific implementations of starlark utilities."""

load(
    "//xla/tsl:package_groups.bzl",
    "DEFAULT_LOAD_VISIBILITY",
    "LEGACY_TSL_PLATFORM_BUILD_CONFIG_ROOT_USERS",
)
load(
    "//xla/tsl/platform/default:build_config_root.bzl",
    _if_llvm_aarch32_available = "if_llvm_aarch32_available",
    _if_llvm_aarch64_available = "if_llvm_aarch64_available",
    _if_llvm_arm_available = "if_llvm_arm_available",
    _if_llvm_powerpc_available = "if_llvm_powerpc_available",
    _if_llvm_system_z_available = "if_llvm_system_z_available",
    _if_llvm_x86_available = "if_llvm_x86_available",
    _if_pywrap = "if_pywrap",
    _if_static = "if_static",
    _if_static_and_not_mobile = "if_static_and_not_mobile",
    _tf_additional_grpc_deps_py = "tf_additional_grpc_deps_py",
    _tf_additional_license_deps = "tf_additional_license_deps",
    _tf_additional_profiler_deps = "tf_additional_profiler_deps",
    _tf_additional_tpu_ops_deps = "tf_additional_tpu_ops_deps",
    _tf_additional_xla_deps_py = "tf_additional_xla_deps_py",
    _tf_cuda_tests_tags = "tf_cuda_tests_tags",
    _tf_exec_properties = "tf_exec_properties",
    _tf_gpu_tests_tags = "tf_gpu_tests_tags",
)

visibility(DEFAULT_LOAD_VISIBILITY + LEGACY_TSL_PLATFORM_BUILD_CONFIG_ROOT_USERS)

if_llvm_aarch32_available = _if_llvm_aarch32_available
if_llvm_aarch64_available = _if_llvm_aarch64_available
if_llvm_arm_available = _if_llvm_arm_available
if_llvm_powerpc_available = _if_llvm_powerpc_available
if_llvm_system_z_available = _if_llvm_system_z_available
if_llvm_x86_available = _if_llvm_x86_available
if_static = _if_static
if_pywrap = _if_pywrap
if_static_and_not_mobile = _if_static_and_not_mobile
tf_additional_grpc_deps_py = _tf_additional_grpc_deps_py
tf_additional_license_deps = _tf_additional_license_deps
tf_additional_profiler_deps = _tf_additional_profiler_deps
tf_additional_tpu_ops_deps = _tf_additional_tpu_ops_deps
tf_additional_xla_deps_py = _tf_additional_xla_deps_py
tf_cuda_tests_tags = _tf_cuda_tests_tags
tf_exec_properties = _tf_exec_properties
tf_gpu_tests_tags = _tf_gpu_tests_tags

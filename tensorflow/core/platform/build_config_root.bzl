"""Provides a redirection point for platform specific implementations of starlark utilities."""

load(
    "//tensorflow/core/platform/default:build_config_root.bzl",
    _if_dynamic_kernels = "if_dynamic_kernels",
    _if_static = "if_static",
    _if_static_and_not_mobile = "if_static_and_not_mobile",
    _register_extension_info = "register_extension_info",
    _tf_additional_grpc_deps_py = "tf_additional_grpc_deps_py",
    _tf_additional_license_deps = "tf_additional_license_deps",
    _tf_additional_plugin_deps = "tf_additional_plugin_deps",
    _tf_additional_xla_deps_py = "tf_additional_xla_deps_py",
    _tf_cuda_tests_tags = "tf_cuda_tests_tags",
    _tf_exec_compatible_with = "tf_exec_compatible_with",
    _tf_gpu_tests_tags = "tf_gpu_tests_tags",
    _tf_sycl_tests_tags = "tf_sycl_tests_tags",
)

if_dynamic_kernels = _if_dynamic_kernels
if_static = _if_static
if_static_and_not_mobile = _if_static_and_not_mobile
register_extension_info = _register_extension_info
tf_additional_grpc_deps_py = _tf_additional_grpc_deps_py
tf_additional_license_deps = _tf_additional_license_deps
tf_additional_plugin_deps = _tf_additional_plugin_deps
tf_additional_xla_deps_py = _tf_additional_xla_deps_py
tf_cuda_tests_tags = _tf_cuda_tests_tags
tf_exec_compatible_with = _tf_exec_compatible_with
tf_gpu_tests_tags = _tf_gpu_tests_tags
tf_sycl_tests_tags = _tf_sycl_tests_tags

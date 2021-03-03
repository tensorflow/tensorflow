load(
    "@bazel_skylib//rules:common_settings.bzl",
    "bool_flag",
    "string_flag",
)

package(default_visibility = ["//visibility:public"])

# Build flag to enable CUDA support.
#
# Enable with '--@local_config_cuda//:enable_cuda', or indirectly with
# ./configure or '--config=cuda'.
bool_flag(
    name = "enable_cuda",
    build_setting_default = False,
)

# Config setting whether CUDA support has been requested.
#
# Enable path: ./configure > --config=cuda (.tf_configure.bazelrc)
#     > --//tensorflow:enable_cuda (.bazelrc) > :is_cuda_enabled
config_setting(
    name = "is_cuda_enabled",
    flag_values = {":enable_cuda": "True"},
)

# Build flag to select CUDA compiler.
#
# Set with '--@local_config_cuda//:cuda_compiler=...', or indirectly with
# ./configure, '--config=cuda' or '--config=cuda_clang'.
string_flag(
    name = "cuda_compiler",
    build_setting_default = "nvcc",
    values = [
        "clang",
        "nvcc",
    ],
)

# Config setting whether CUDA device code should be compiled with clang.
config_setting(
    name = "is_cuda_compiler_clang",
    flag_values = {":cuda_compiler": "clang"},
)

# Config setting whether CUDA device code should be compiled with nvcc.
config_setting(
    name = "is_cuda_compiler_nvcc",
    flag_values = {":cuda_compiler": "nvcc"},
)

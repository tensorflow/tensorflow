load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@local_config_cuda//cuda:build_defs.bzl", "enable_cuda_flag")

package(default_visibility = ["//visibility:public"])

# Build flag to enable CUDA support.
#
# Enable with '--@local_config_cuda//:enable_cuda', or indirectly with
# ./configure or '--config=cuda'.
enable_cuda_flag(
    name = "enable_cuda",
    build_setting_default = False,
    enable_override = select({
        ":define_using_cuda_nvcc": True,
        "//conditions:default": False,
    }),
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

# Config setting to keep `--define=using_cuda_nvcc=true` working.
# TODO(b/174244321): Remove when downstream projects have been fixed, along
# with the enable_cuda_flag rule in cuda:build_defs.bzl.tpl.
config_setting(
    name = "define_using_cuda_nvcc",
    define_values = {"using_cuda_nvcc": "true"},
    visibility = ["//visibility:private"],
)

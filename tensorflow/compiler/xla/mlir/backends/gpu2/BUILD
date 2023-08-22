load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

# Add `--//third_party/tensorflow/compiler/xla/mlir/backends/gpu2:enable` to build command to
# enable experimental backend compiler and runtime.
bool_flag(
    name = "enable",
    build_setting_default = False,
)

config_setting(
    name = "enabled",
    flag_values = {":enable": "True"},
)

# copybara:uncomment_begin(not supported in OSS build)
#
# build_test(
#     name = "xla-gpu2-opt_build_test",
#     targets = [":xla-gpu2-opt"],
# )
#
# xla_cc_binary(
#     name = "xla-gpu2-opt",
#     srcs = ["xla-gpu2-opt.cc"],
#     deps = [
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:MlirOptLib",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/transforms:passes",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
#         "//tensorflow/tsl/platform:platform_port",
#     ],
# )
#
# copybara:uncomment_end

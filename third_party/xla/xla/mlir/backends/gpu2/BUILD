load("//xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
#         "//xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//xla/mlir/backends/gpu2/transforms:passes",
#         "//xla/mlir_hlo:lhlo",
#         "//xla/mlir_hlo:lhlo_gpu",
#         "@local_tsl//tsl/platform:platform_port",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:MlirOptLib",
#         "@llvm-project//mlir:Transforms",
#     ],
# )
#
# copybara:uncomment_end

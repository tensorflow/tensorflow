load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config.bzl", "tf_platform_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/xla:friends"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# # Add `--define=xla_gpu_with_openxla_runtime=1` to build command to enable experimental OpenXLA/IREE
# # backend for XLA:GPU executables.
# config_setting(
#     name = "with_openxla_runtime",
#     values = {
#         "define": "xla_gpu_with_openxla_runtime=1",
#     },
# )
#
# cc_library(
#     name = "compiler",
#     srcs = select({
#         ":with_openxla_runtime": ["compiler.cc"],
#         "//conditions:default": [],
#     }),
#     hdrs = select({
#         ":with_openxla_runtime": ["compiler.h"],
#         "//conditions:default": [],
#     }),
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     data = select({
#         ":with_openxla_runtime": ["//third_party/iree/lib:libIREECompiler.so"],
#         "//conditions:default": [],
#     }),
#     deps = [
#         "@com_google_absl//absl/base",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:util",
#         "//tensorflow/tsl/platform",
#     ] + tf_platform_deps(
#         "compiler",
#         platform_dir = "//tensorflow/compiler/xla/service/gpu/openxla/",
#     ) + select({
#         ":with_openxla_runtime": [
#             "//third_party/iree/compiler/bindings/c:headers",
#             "//third_party/iree/compiler/bindings/c:loader",
#             "//third_party/iree/llvm-external-projects/iree-dialects:IREEInputDialect",
#         ],
#         "//conditions:default": [],
#     }),
# )
#
# cc_library(
#     name = "executable",
#     srcs = select({
#         ":with_openxla_runtime": ["executable.cc"],
#         "//conditions:default": [],
#     }),
#     hdrs = ["executable.h"],
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     defines = select({
#         ":with_openxla_runtime": [],
#         "//conditions:default": ["XLA_DISABLE_OPENXLA_RUNTIME=1"],
#     }),
#     deps = [
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/strings",
#         "@llvm-project//mlir:IR",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla:util",
#         "//tensorflow/compiler/xla:xla_proto_cc",
#         "//tensorflow/compiler/xla/service:buffer_assignment",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
#         "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
#     ] + select({
#         ":with_openxla_runtime": [
#             ":compiler",
#             "//third_party/iree/runtime/src/iree/base",
#             "//third_party/iree/runtime/src/iree/hal",
#             "//third_party/iree/runtime/src/iree/hal/drivers/cuda",
#             "//third_party/iree/runtime/src/iree/modules/hal",
#             "//third_party/iree/runtime/src/iree/modules/hal:types",
#             "//third_party/iree/runtime/src/iree/vm",
#             "//third_party/iree/runtime/src/iree/vm/bytecode:module",
#         ],
#         "//conditions:default": [],
#     }),
# )
#
# copybara:uncomment_end_and_comment_begin
cc_library(
    name = "executable",
    hdrs = ["executable.h"],
    defines = ["XLA_DISABLE_OPENXLA_RUNTIME=1"],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
        "@llvm-project//mlir:IR",
    ],
)
# copybara:comment_end

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
# # Add `--define=xla_gpu_bundle_lib_iree_compiler=1` to build command to bundle `libIREECompiler.so`
# # with XLA:GPU by default. Otherwise use `XLA_OPENXLA_IREE_COMPILER_LIB` environment variable to
# # load custom compiler library.
# config_setting(
#     name = "bundle_lib_iree_compiler",
#     values = {
#         "define": "xla_gpu_bundle_lib_iree_compiler=1",
#     },
# )
#
# cc_library(
#     name = "compiler",
#     srcs = ["compiler.cc"],
#     hdrs = ["compiler.h"],
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     data = select({
#         ":bundle_lib_iree_compiler": ["//third_party/iree/lib:libIREECompiler.so"],
#         "//conditions:default": [],
#     }),
#     deps = [
#         "@com_google_absl//absl/base",
#         "//third_party/iree/compiler/bindings/c:headers",
#         "//third_party/iree/compiler/bindings/c:loader",
#         "//third_party/iree/llvm-external-projects/iree-dialects:IREEInputDialect",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:Support",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:util",
#         "//tensorflow/tsl/platform",
#     ] + tf_platform_deps(
#         "compiler",
#         platform_dir = "//tensorflow/compiler/xla/service/gpu/openxla/",
#     ),
# )
#
# cc_library(
#     name = "executable",
#     srcs = ["executable.cc"],
#     hdrs = ["executable.h"],
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":compiler",
#         "@com_google_absl//absl/log",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/strings",
#         "//third_party/iree/runtime/src/iree/base",
#         "//third_party/iree/runtime/src/iree/hal",
#         "//third_party/iree/runtime/src/iree/hal/drivers/cuda",
#         "//third_party/iree/runtime/src/iree/modules/hal",
#         "//third_party/iree/runtime/src/iree/modules/hal:types",
#         "//third_party/iree/runtime/src/iree/vm",
#         "//third_party/iree/runtime/src/iree/vm/bytecode:module",
#         "@llvm-project//mlir:IR",
#         "//tensorflow/compiler/xla:status",
#         "//tensorflow/compiler/xla:statusor",
#         "//tensorflow/compiler/xla:util",
#         "//tensorflow/compiler/xla:xla_proto_cc",
#         "//tensorflow/compiler/xla/service:buffer_assignment",
#         "//tensorflow/compiler/xla/service:executable",
#         "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
#         "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
#     ],
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

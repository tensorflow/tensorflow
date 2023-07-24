load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# gentbl_cc_library(
#     name = "passes_inc_gen",
#     compatible_with = get_compatible_with_portable(),
#     tbl_outs = [
#         (
#             ["-gen-pass-decls"],
#             "passes.h.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "passes.td",
#     deps = ["@llvm-project//mlir:PassBaseTdFiles"],
# )
#
# cc_library(
#     name = "passes",
#     srcs = [
#         "convert_to_openxla.cc",
#         "passes.cc",
#     ],
#     hdrs = ["passes.h"],
#     # TODO(ezhulenev): Override cc_library()'s default compatibility because IREE targets are not
#     # compatible with `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":passes_inc_gen",
#         "//third_party/iree/llvm-external-projects/iree-dialects:IREEInputDialect",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Pass",
#         "@llvm-project//mlir:SCFDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/openxla/conversion:convert_compiled_ops",
#         "//tensorflow/compiler/xla/mlir/backends/openxla/conversion:convert_memref_ops",
#         "//tensorflow/compiler/xla/mlir/backends/openxla/conversion:convert_while_op",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#     ],
# )
#
# copybara:uncomment_end_and_comment_begin
cc_library(
    name = "passes",
    hdrs = ["passes.h"],
    defines = ["XLA_DISABLE_OPENXLA_COMPILER=1"],
)
# copybara:comment_end

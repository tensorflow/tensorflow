load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla/mlir/backends/gpu2:build_config.bzl", "if_gpu2")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla:internal"],
    licenses = ["notice"],
)

# copybara:uncomment_begin(not supported in OSS build)
#
# cc_library(
#     name = "convert_compiled_ops",
#     srcs = if_gpu2(["convert_compiled_ops.cc"]),
#     hdrs = if_gpu2(["convert_compiled_ops.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         ":xla_gpu_api",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:BufferizationDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#         "//tensorflow/compiler/xla/service/gpu:gpu_executable",
#         "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
#         "//tensorflow/compiler/xla/service/gpu:thunk",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "convert_graph_region_op",
#     srcs = if_gpu2(["convert_graph_region_op.cc"]),
#     hdrs = if_gpu2(["convert_graph_region_op.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         ":xla_gpu_api",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:BufferizationDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#     ],
# )
#
# cc_library(
#     name = "convert_library_ops",
#     srcs = if_gpu2(["convert_library_ops.cc"]),
#     hdrs = if_gpu2(["convert_library_ops.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         ":xla_gpu_api",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
#         "//tensorflow/compiler/xla/translate/mhlo_to_hlo:location_exporter",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "convert_memref_ops",
#     srcs = if_gpu2(["convert_memref_ops.cc"]),
#     hdrs = if_gpu2(["convert_memref_ops.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "convert_case_op",
#     srcs = if_gpu2(["convert_case_op.cc"]),
#     hdrs = if_gpu2(["convert_case_op.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         ":xla_gpu_api",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:BufferizationDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:SCFDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "convert_while_op",
#     srcs = if_gpu2(["convert_while_op.cc"]),
#     hdrs = if_gpu2(["convert_while_op.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         ":xla_gpu_api",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:BufferizationDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:SCFDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:TensorDialect",
#         "@llvm-project//mlir:Transforms",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "de_bufferization",
#     srcs = ["de_bufferization.cc"],
#     hdrs = ["de_bufferization.h"],
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:BufferizationDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "//tensorflow/compiler/xla/mlir_hlo:lhlo",
#     ],
# )
#
# cc_library(
#     name = "xla_gpu_api",
#     srcs = if_gpu2(["xla_gpu_api.cc"]),
#     hdrs = if_gpu2(["xla_gpu_api.h"]),
#     deps = [
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "//tensorflow/compiler/xla/mlir/backends/gpu2/ir:xla_gpu",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# copybara:uncomment_end

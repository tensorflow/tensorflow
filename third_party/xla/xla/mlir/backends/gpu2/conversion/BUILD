load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/mlir/backends/gpu2:build_config.bzl", "if_gpu2")

package(
    default_visibility = ["//visibility:public"],
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
#         "//xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//xla/mlir_hlo:lhlo",
#         "//xla/service/gpu:gpu_executable",
#         "//xla/service/gpu:launch_dimensions",
#         "//xla/service/gpu:thunk",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:BufferizationDialect",
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
#     name = "convert_library_ops",
#     srcs = if_gpu2(["convert_library_ops.cc"]),
#     hdrs = if_gpu2(["convert_library_ops.h"]),
#     # TODO(ezhulenev): Override cc_library()'s internal default value of ["//buildenv/target:gce"]
#     # because IREE targets are not compatible with the `non_prod` constraint.
#     compatible_with = [],
#     deps = [
#         ":de_bufferization",
#         ":xla_gpu_api",
#         "//xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//xla/mlir_hlo:lhlo_gpu",
#         "//xla/translate/mhlo_to_hlo:location_exporter",
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
#         "//xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//xla/mlir_hlo:lhlo",
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
#         "//xla/mlir/backends/gpu2/ir:xla_gpu",
#         "//xla/mlir_hlo:lhlo",
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
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "de_bufferization",
#     srcs = if_gpu2(["de_bufferization.cc"]),
#     hdrs = if_gpu2(["de_bufferization.h"]),
#     deps = [
#         "//xla/mlir_hlo:lhlo",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:BufferizationDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#         "@llvm-project//mlir:Transforms",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# cc_library(
#     name = "xla_gpu_api",
#     srcs = if_gpu2(["xla_gpu_api.cc"]),
#     hdrs = if_gpu2(["xla_gpu_api.h"]),
#     deps = [
#         "//xla/mlir/backends/gpu2/ir:xla_gpu",
#         "@llvm-project//llvm:Support",
#         "@llvm-project//mlir:ArithDialect",
#         "@llvm-project//mlir:FuncDialect",
#         "@llvm-project//mlir:IR",
#         "@llvm-project//mlir:MemRefDialect",
#         "@llvm-project//mlir:Support",
#     ] + if_gpu2(["//third_party/iree/llvm-external-projects/iree-dialects/google_internal:IREEInputDialect"]),
# )
#
# copybara:uncomment_end

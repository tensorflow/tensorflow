#
# Copyright 2020 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set(LLVM_TARGET_DEFINITIONS mhlo_passes.td)
mlir_tablegen(mhlo_passes.h.inc -gen-pass-decls -name AllMhlo)
add_public_tablegen_target(MLIRMhloPassIncGen)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS lower_complex/lower_complex_patterns.td)
mlir_tablegen(lower_complex/generated_lower_complex.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloLowerComplexIncGen)

set(LLVM_TARGET_DEFINITIONS legalize_to_standard/legalize_to_standard_patterns.td)
mlir_tablegen(legalize_to_standard/generated_legalize_to_standard.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloLegalizeToStandardIncGen)

set(LLVM_TARGET_DEFINITIONS chlo_legalize_to_hlo/chlo_legalize_to_hlo_patterns.td)
mlir_tablegen(chlo_legalize_to_hlo/generated_chlo_legalize_to_hlo.inc -gen-rewriters)
add_public_tablegen_target(MLIRChloLegalizeToHloIncGen)



add_mlir_library(MhloPasses
  broadcast_propagation/broadcast_propagation.cc
  collapse_elementwise_map/collapse_elementwise_map.cc
  constraint_fusion/constraint_fusion_pass.cc
  convert_to_signless/convert_to_signless_pass.cc
  expand_hlo_tuples/expand_hlo_tuples.cc
  expand_ops_simplifier/expand_ops_simplifier.cc
  group_reduction_dimensions/group_reduction_dimensions.cc
  legalize_broadcast_to_broadcast_in_dim/legalize_broadcast_to_broadcast_in_dim.cc
  legalize_create_token_to_after_all/legalize_create_token_to_after_all.cc
  legalize_cross_replica_sum_to_all_reduce/legalize_cross_replica_sum_to_all_reduce.cc
  legalize_dot_general_to_dot/legalize_dot_general_to_dot.cc
  legalize_dot_to_dot_general/legalize_dot_to_dot_general.cc
  legalize_einsum_to_dot_general/legalize_einsum_to_dot_general.cc
  legalize_gather_to_torch_index_select/legalize_gather_to_torch_index_select.cc
  legalize_shape_computations/legalize_shape_computations.cc
  legalize_torch_index_select_to_gather/legalize_torch_index_select_to_gather.cc
  legalize_trigonometric_to_approximation/legalize_trigonometric_to_approximation.cc
  lower_complex/lower_complex.cc
  lower_complex/lower_complex_patterns.td
  lower_general_dot/lower_general_dot.cc
  materialize_broadcasts/materialize_broadcasts.cc
  materialize_broadcasts/materialize_broadcasts_pass.cc
  merge_assuming_ops/merge_assuming_ops.cc
  mhlo_canonicalize_dot/mhlo_canonicalize_dot.cc
  mhlo_canonicalize_gather/mhlo_canonicalize_gather.cc
  mhlo_canonicalize_reduction/mhlo_canonicalize_reduction.cc
  mhlo_canonicalize_scatter/mhlo_canonicalize_scatter.cc
  mhlo_flatten_tuple/mhlo_flatten_tuple.cc
  prepare_for_export/prepare_for_export.cc
  optimize_mhlo/optimize_mhlo.cc
  optimize_mhlo/optimize_mhlo_pass.cc
  restrict_max_rank/restrict_max_rank.cc
  shape_legalize_to_hlo/shape_legalize_to_hlo.cc
  shape_reification/shape_reification_pass.cc
  shape_simplification/shape_simplification.cc
  sink_constants_to_control_flow/sink_constants_to_control_flow.cc
  symbolic_shape_optimization/symbolic_shape_optimization.cc
  test_infer_shaped_type/test_infer_shaped_type_pass.cc
  unfuse_batch_norm/unfuse_batch_norm.cc
  unfuse_batch_norm/unfuse_batch_norm_pass.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloLowerComplexIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ChloOps
  MhloAnalysis
  MhloDialect
  MhloScatterUtils
  MhloTypeConversion
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMhloUtils
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransformUtils
  StablehloBroadcastUtils
)

add_mlir_library(MhloQuantToIntConversion
  mhlo_quant_legalize_to_int/mhlo_quant_legalize_to_int.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRIR
  MLIRPass
  MLIRMathDialect
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_library(MhloToMemrefConversion
  hlo_legalize_to_memref/hlo_legalize_to_memref.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRIR
  MLIRPass
  MLIRMathDialect
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_library(MhloToArithmeticConversion
  hlo_legalize_to_arithmetic/hlo_legalize_to_arithmetic.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRIR
  MLIRPass
  MLIRMathDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRTransformUtils
)

add_mlir_library(MhloToStandard
  legalize_control_flow/legalize_control_flow.cc
  legalize_sort/legalize_sort.cc
  legalize_to_standard/legalize_to_standard.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloLegalizeToStandardIncGen
  MLIRMhloPassIncGen
  MhloTypeConversion

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
)

add_mlir_library(ChloPasses
  chlo_legalize_to_hlo/chlo_legalize_to_hlo_pass.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRChloLegalizeToHloIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ChloOps
  HloToLinalgUtils
  MLIRComplexDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)

add_mlir_library(MhloToLinalg
  legalize_to_linalg/legalize_to_linalg.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  HloToLinalgUtils
  MhloDialect
  MhloRngUtils
  MhloToArithmeticConversion
  MhloTypeConversion
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRIR
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)

add_mlir_library(MhloShapeOpsToStandard
  hlo_legalize_shape_ops_to_standard/hlo_legalize_shape_ops_to_standard.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRComplexDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)

add_mlir_library(MhloToStablehlo
  hlo_legalize_to_stablehlo/hlo_legalize_to_stablehlo.cc
  hlo_legalize_to_stablehlo/hlo_legalize_to_stablehlo_pass.cc

  DEPENDS
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  StablehloOps
)

add_mlir_library(StablehloToMhlo
  stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo.cc
  stablehlo_legalize_to_hlo/stablehlo_legalize_to_hlo_pass.cc

  DEPENDS
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRAsmParser
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  StablehloOps
)

add_library(AllMhloPasses INTERFACE)
target_link_libraries(AllMhloPasses INTERFACE
  ChloPasses
  MhloPasses
  MhloQuantToIntConversion
  MhloToArithmeticConversion
  MhloToMemrefConversion
  MhloToStandard
  HloToLinalgUtils
  MhloToLinalg
  MhloShapeOpsToStandard
  MhloToStablehlo
  StablehloToMhlo
)

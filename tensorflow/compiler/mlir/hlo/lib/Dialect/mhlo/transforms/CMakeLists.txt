#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS lower_complex_patterns.td)
mlir_tablegen(generated_lower_complex.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloLowerComplexIncGen)

set(LLVM_TARGET_DEFINITIONS legalize_to_standard_patterns.td)
mlir_tablegen(generated_legalize_to_standard.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloLegalizeToStandardIncGen)

set(LLVM_TARGET_DEFINITIONS chlo_legalize_to_hlo_patterns.td)
mlir_tablegen(generated_chlo_legalize_to_hlo.inc -gen-rewriters)
add_public_tablegen_target(MLIRChloLegalizeToHloIncGen)


add_mlir_library(ChloPasses
  chlo_legalize_to_hlo.cc
  chlo_legalize_to_hlo_pass.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRChloLegalizeToHloIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  ChloDialect
  MLIRIR
  MLIRPass
)

add_mlir_library(MhloPasses
  broadcast_propagation.cc
  legalize_einsum_to_dot_general.cc
  legalize_gather_to_torch_index_select.cc
  legalize_trigonometric_to_approximation.cc
  lower_complex.cc
  lower_complex_patterns.td
  lower_general_dot.cc
  materialize_broadcasts.cc
  materialize_broadcasts_pass.cc
  mhlo_canonicalize_reduction.cc
  mhlo_fusion.cc
  mhlo_mark_shape_calc.cc
  optimize_mhlo.cc
  optimize_mhlo_pass.cc
  rank_specialization.cc
  sink_constants_to_control_flow.cc
  test_infer_shaped_type_pass.cc
  unfuse_batch_norm.cc
  unfuse_batch_norm_pass.cc

  DEPENDS
  MLIRDiscRalPassIncGen
  MLIRhlo_opsIncGen
  MLIRMhloLowerComplexIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMhloUtils
  MLIRPass
  MLIRTransformUtils
)

add_mlir_library(MhloToLhloConversion
  hlo_legalize_to_lhlo.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRlhlo_opsIncGen
  MLIRMhloPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  LmhloDialect
  MLIRIR
  MLIRPass
  MLIRMath
)

add_mlir_library(MhloTypeConversion
  type_conversion.cc

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_library(MhloToMemrefConversion
  hlo_legalize_to_memref.cc

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
  MLIRMath
)

add_mlir_library(MhloToStandard
  legalize_control_flow.cc
  legalize_to_standard.cc
  mhlo_control_flow_to_scf.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRlhlo_opsIncGen
  MLIRMhloLegalizeToStandardIncGen
  MhloTypeConversion

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTensor
)

add_mlir_library(MhloLhloToLinalg
  legalize_to_linalg.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRlhlo_opsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MhloDialect
  MhloTypeConversion
  MLIRComplex
  MLIRIR
  MLIRPass
)

add_mlir_library(LmhloPasses
  fusion_utils.cc
  input_inline_fusion_pass.cc
  legalize_tensor_load_op.cc
  lhlo_elemental_utils.cc
  lhlo_fuse_linalg.cc
  lhlo_fusion.cc
  lhlo_fusion_inliner.cc
  lhlo_legalize_roots_to_loops.cc
  lhlo_legalize_to_affine.cc
  lhlo_legalize_to_gpu.cc
  lhlo_legalize_to_parallel_loops.cc

  DEPENDS
  MLIRlhlo_opsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LmhloDialect
  MLIRMhloUtils
  MLIRIR
  MLIRPass
)

add_mlir_library(DiscRalPasses
  ral_inject_execution_context.cc
  ral_lower_to_library_call.cc

  DEPENDS
  MLIRdisc_ral_opsIncGen
  MLIRDiscRalPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  DiscRalDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
)

add_mlir_library(DiscRalToLLVM
  ral_legalize_to_llvm.cc

  DEPENDS
  MLIRdisc_ral_opsIncGen
  MLIRDiscRalPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  DiscRalDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
)

add_library(AllMhloPasses INTERFACE)
target_link_libraries(AllMhloPasses INTERFACE
  ChloPasses
  DiscRalPasses
  DiscRalToLLVM
  MhloPasses
  MhloToLhloConversion
  MhloToMemrefConversion
  MhloToStandard
  MhloLhloToLinalg
  LmhloPasses
)

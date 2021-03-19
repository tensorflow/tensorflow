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

set(LLVM_TARGET_DEFINITIONS hlo_patterns.td)
mlir_tablegen(hlo_patterns.cc.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloRewriterIncGen)

set(LLVM_TARGET_DEFINITIONS mhlo_canonicalize.td)
mlir_tablegen(mhlo_canonicalize.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloCanonicalizeIncGen)

add_mlir_dialect_library(ChloDialect
  chlo_ops.cc

  DEPENDS
  MLIRchlo_opsIncGen
)
target_link_libraries(ChloDialect PUBLIC MLIRIR)

add_mlir_library(MhloInferFusibilityOpInterface
  infer_fusibility_op_interface.cc

  DEPENDS
  MLIRinfer_fusibility_op_interfaceIncGen
)

add_mlir_library(HloOpsCommon
  hlo_ops_common.cc
)

add_mlir_dialect_library(MhloDialect
  hlo_ops.cc
  hlo_ops_base_structs.cc
  hlo_ops_base_enums.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloCanonicalizeIncGen
  MLIRMhloRewriterIncGen
  MLIRinfer_fusibility_op_interfaceIncGen
)
target_link_libraries(MhloDialect
  PUBLIC
  MLIRIR
  MhloInferFusibilityOpInterface
  MLIRMhloUtils
  HloOpsCommon
)


add_mlir_dialect_library(LmhloDialect
  lhlo_ops.cc
  lhlo_ops_structs.cc

  DEPENDS
  MLIRlhlo_opsIncGen
)
target_link_libraries(LmhloDialect
  PUBLIC
  MLIRIR
  HloOpsCommon
)

add_mlir_dialect_library(LmhloGPUDialect
  lhlo_gpu_ops.cc
  lhlo_gpu_ops_structs.cc
  lhlo_gpu_ops_enums.cc

  DEPENDS
  MLIRlhlo_gpu_opsIncGen
)
target_link_libraries(LmhloGPUDialect PUBLIC MLIRIR)


add_mlir_dialect_library(MhloRegisterDialects
  init.cc
DEPENDS
  MLIRchlo_opsIncGen
  MLIRhlo_opsIncGen
  MLIRlhlo_opsIncGen
  MLIRlhlo_gpu_opsIncGen
)
target_link_libraries(MhloRegisterDialects
  PUBLIC
  ChloDialect
  MhloDialect
  LmhloDialect
  LmhloGPUDialect
)

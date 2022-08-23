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

add_mlir_library(MLIRHloBase
  base.cc

  DEPENDS
  MLIRHloBaseIncGen

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSupport
)

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

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRHloBase
  MLIRComplexDialect
  MLIRIR
  MLIRMhloUtils
)

add_mlir_library(HloOpsCommon
  hlo_ops_common.cc

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_dialect_library(MhloDialect
  hlo_ops.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloCanonicalizeIncGen
  MLIRMhloRewriterIncGen
)
target_link_libraries(MhloDialect
  PUBLIC
  MLIRComplexDialect
  MLIRHloBase
  MLIRIR
  MLIRMhloUtils
  MLIRQuantDialect
  HloOpsCommon
)
target_include_directories(MhloDialect
  PUBLIC
  $<BUILD_INTERFACE:${MLIR_HLO_GEN_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${MLIR_HLO_MAIN_INCLUDE_DIR}>
)

add_mlir_dialect_library(MhloRegisterDialects
  init.cc
DEPENDS
  MLIRchlo_opsIncGen
  MLIRhlo_opsIncGen
)
target_link_libraries(MhloRegisterDialects
  PUBLIC
  ChloDialect
  MhloDialect
)

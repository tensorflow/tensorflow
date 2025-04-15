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
# Need a separate function because of the .cc vs .cpp used in the one provided by MLIR
set(LLVM_TARGET_DEFINITIONS hlo_ops.td)
mlir_tablegen(hlo_ops.h.inc -gen-op-decls)
mlir_tablegen(hlo_ops.cc.inc -gen-op-defs)
mlir_tablegen(hlo_ops_enums.h.inc -gen-enum-decls)
mlir_tablegen(hlo_ops_enums.cc.inc -gen-enum-defs)
mlir_tablegen(hlo_ops_attrs.h.inc -gen-attrdef-decls)
mlir_tablegen(hlo_ops_attrs.cc.inc -gen-attrdef-defs)
mlir_tablegen(hlo_ops_typedefs.h.inc -gen-typedef-decls --typedefs-dialect=mhlo)
mlir_tablegen(hlo_ops_typedefs.cc.inc -gen-typedef-defs --typedefs-dialect=mhlo)
add_public_tablegen_target(MLIRhlo_opsIncGen)
add_dependencies(mlir-headers MLIRhlo_opsIncGen)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS hlo_patterns.td)
mlir_tablegen(hlo_patterns.cc.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloRewriterIncGen)

set(LLVM_TARGET_DEFINITIONS mhlo_canonicalize.td)
mlir_tablegen(mhlo_canonicalize.inc -gen-rewriters)
add_public_tablegen_target(MLIRMhloCanonicalizeIncGen)

add_mlir_library(HloOpsCommon
  hlo_ops_common.cc

  LINK_LIBS PUBLIC
  MLIRIR
)

add_mlir_dialect_library(MhloDialect
  hlo_ops.cc
  mhlo_bytecode.cc

  DEPENDS
  MLIRhlo_opsIncGen
  MLIRMhloCanonicalizeIncGen
  MLIRMhloRewriterIncGen
)
target_link_libraries(MhloDialect
  PUBLIC
  MLIRComplexDialect
  MLIRIR
  MLIRMhloUtils
  MLIRQuantDialect
  MLIRSparseTensorDialect
  HloOpsCommon
  StablehloAssemblyFormat
  StablehloBase
  StablehloTypeInference
)
target_include_directories(MhloDialect
  PUBLIC
  $<BUILD_INTERFACE:${MLIR_HLO_SOURCE_DIR}>
  $<BUILD_INTERFACE:${MLIR_HLO_GEN_INCLUDE_DIR}>
)

add_mlir_dialect_library(MhloRegisterDialects
  init.cc
DEPENDS
  MLIRhlo_opsIncGen
)
target_link_libraries(MhloRegisterDialects
  PUBLIC
  MhloDialect
)

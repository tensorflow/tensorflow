# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

set(LLVM_TARGET_DEFINITIONS gml_st_ops.td)
mlir_tablegen(gml_st_ops.h.inc -gen-op-decls)
mlir_tablegen(gml_st_ops.cc.inc -gen-op-defs)
mlir_tablegen(gml_st_dialect.h.inc -gen-dialect-decls)
mlir_tablegen(gml_st_dialect.cc.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRgml_st_opsIncGen)
add_dependencies(mlir-headers MLIRgml_st_opsIncGen)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_dialect_library(GmlStDialect
  gml_st_ops.cc

  DEPENDS
  MLIRgml_st_opsIncGen

  LINK_LIBS PUBLIC
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRVectorDialect
)

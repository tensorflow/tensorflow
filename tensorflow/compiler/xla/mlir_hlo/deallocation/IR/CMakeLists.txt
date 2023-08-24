# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

set(LLVM_TARGET_DEFINITIONS deallocation_ops.td)
mlir_tablegen(deallocation_ops.h.inc -gen-op-decls)
mlir_tablegen(deallocation_ops.cc.inc -gen-op-defs)
mlir_tablegen(deallocation_dialect.h.inc -gen-dialect-decls)
mlir_tablegen(deallocation_dialect.cc.inc -gen-dialect-defs)
mlir_tablegen(deallocation_typedefs.h.inc -gen-typedef-decls)
mlir_tablegen(deallocation_typedefs.cc.inc -gen-typedef-defs)

add_public_tablegen_target(MLIRdeallocation_opsIncGen)
add_dependencies(mlir-headers MLIRdeallocation_opsIncGen)

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_dialect_library(DeallocationDialect
  deallocation_ops.cc

  DEPENDS
  MLIRdeallocation_opsIncGen

  LINK_LIBS PUBLIC
  MLIRDeallocationUtils
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
)

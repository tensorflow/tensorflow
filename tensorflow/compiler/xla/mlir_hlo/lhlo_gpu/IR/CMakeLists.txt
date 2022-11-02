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

set(LLVM_TARGET_DEFINITIONS lhlo_gpu_ops.td)
mlir_tablegen(lhlo_gpu_ops.h.inc -gen-op-decls)
mlir_tablegen(lhlo_gpu_ops.cc.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS lhlo_gpu_ops_enums.td)
mlir_tablegen(lhlo_gpu_ops_enums.h.inc -gen-enum-decls)
mlir_tablegen(lhlo_gpu_ops_enums.cc.inc -gen-enum-defs)
mlir_tablegen(lhlo_gpu_ops_attrdefs.h.inc -gen-attrdef-decls)
mlir_tablegen(lhlo_gpu_ops_attrdefs.cc.inc -gen-attrdef-defs)
mlir_tablegen(lhlo_gpu_ops_dialect.h.inc -gen-dialect-decls)
mlir_tablegen(lhlo_gpu_ops_dialect.cc.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRlhlo_gpu_opsIncGen)
add_dependencies(mlir-headers MLIRlhlo_gpu_opsIncGen)


include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_dialect_library(LmhloGPUDialect
  lhlo_gpu_ops.cc

  DEPENDS
  MLIRlhlo_gpu_opsIncGen

  LINK_LIBS PUBLIC
  MhloDialect
  MLIRIR
  HloOpsCommon
)

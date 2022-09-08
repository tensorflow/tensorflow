# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 The StableHLO Authors.
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

set(LLVM_TARGET_DEFINITIONS Base.td)
mlir_tablegen(BaseAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(BaseAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(StablehloBaseIncGen)
add_dependencies(mlir-headers StablehloBaseIncGen)

add_mlir_library(StablehloBase
  PARTIAL_SOURCES_INTENDED
  Base.cpp

  DEPENDS
  StablehloBaseIncGen

  LINK_LIBS PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSupport
)

add_mlir_library(StablehloBroadcastUtils
  PARTIAL_SOURCES_INTENDED
  BroadcastUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRShapeDialect
  MLIRSupport
)

set(LLVM_TARGET_DEFINITIONS ChloOps.td)
mlir_tablegen(ChloOps.h.inc -gen-op-decls)
mlir_tablegen(ChloOps.cpp.inc -gen-op-defs)
mlir_tablegen(ChloEnums.h.inc -gen-enum-decls)
mlir_tablegen(ChloEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ChloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(ChloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(ChloOpsIncGen)
add_dependencies(mlir-headers ChloOpsIncGen)

add_mlir_dialect_library(ChloOps
  PARTIAL_SOURCES_INTENDED
  ChloOps.cpp

  DEPENDS
  ChloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloBase
  StablehloBroadcastUtils
  MLIRArithmeticDialect
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
)

add_mlir_dialect_library(StablehloRegister
  PARTIAL_SOURCES_INTENDED
  Register.cpp

  DEPENDS
  ChloOpsIncGen
  StablehloOpsIncGen

  LINK_LIBS PUBLIC
  ChloOps
  StablehloOps
)

set(LLVM_TARGET_DEFINITIONS StablehloOps.td)
mlir_tablegen(StablehloOps.h.inc -gen-op-decls)
mlir_tablegen(StablehloOps.cpp.inc -gen-op-defs)
mlir_tablegen(StablehloEnums.h.inc -gen-enum-decls)
mlir_tablegen(StablehloEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(StablehloAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(StablehloAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(StablehloOpsIncGen)
add_dependencies(mlir-headers StablehloOpsIncGen)

add_mlir_dialect_library(StablehloOps
  PARTIAL_SOURCES_INTENDED
  StablehloOps.cpp

  DEPENDS
  StablehloOpsIncGen

  LINK_LIBS PUBLIC
  StablehloBase
  MLIRArithmeticDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSparseTensorDialect
  MLIRSupport
  MLIRTensorDialect
)

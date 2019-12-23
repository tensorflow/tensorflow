//===- ROCDLDialect.h - MLIR ROCDL IR dialect -------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the ROCDL dialect in MLIR, containing ROCDL operations
// and ROCDL specific extensions to the LLVM type system.
//
// Unfortunately there does not exists a formal definition of ROCDL IR that be
// pointed to here. However the following links contain more information about
// ROCDL (ROCm-Device-Library)
//
// https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/master/doc/OCML.md
// https://github.com/RadeonOpenCompute/ROCm-Device-Libs/blob/master/doc/OCKL.md
// https://llvm.org/docs/AMDGPUUsage.html
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_ROCDLDIALECT_H_
#define MLIR_DIALECT_LLVMIR_ROCDLDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace ROCDL {

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.h.inc"

class ROCDLDialect : public Dialect {
public:
  explicit ROCDLDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "rocdl"; }
};

} // namespace ROCDL
} // namespace mlir

#endif /* MLIR_DIALECT_LLVMIR_ROCDLDIALECT_H_ */

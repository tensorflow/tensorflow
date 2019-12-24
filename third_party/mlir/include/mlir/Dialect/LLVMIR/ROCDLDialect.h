//===- ROCDLDialect.h - MLIR ROCDL IR dialect -------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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

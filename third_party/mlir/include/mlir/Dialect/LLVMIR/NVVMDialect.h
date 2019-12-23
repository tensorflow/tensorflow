//===- NVVMDialect.h - MLIR NVVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NVVM IR dialect in MLIR, containing NVVM operations and
// NVVM specific extensions to the LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_NVVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_NVVMDIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
namespace mlir {
namespace NVVM {

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMOps.h.inc"

class NVVMDialect : public Dialect {
public:
  explicit NVVMDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "nvvm"; }
};

} // namespace NVVM
} // namespace mlir

#endif /* MLIR_DIALECT_LLVMIR_NVVMDIALECT_H_ */

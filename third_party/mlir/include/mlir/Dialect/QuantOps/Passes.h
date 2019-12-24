//===- Passes.h - Quantization Passes ------ --------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the passes owned by the quantization dialect. As
// things mature, it is expected that passes specific to certain frontend or
// backend dialects will move to those dialects directly. For now, they are
// incubated here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANTOPS_PASSES_H
#define MLIR_DIALECT_QUANTOPS_PASSES_H

#include <memory>

namespace mlir {
class FuncOp;
template <typename T> class OpPassBase;

namespace quant {

/// Creates a pass that converts quantization simulation operations (i.e.
/// FakeQuant and those like it) to casts into/out of supported QuantizedTypes.
std::unique_ptr<OpPassBase<FuncOp>> createConvertSimulatedQuantPass();

/// Creates a pass that converts constants followed by a qbarrier to a
/// constant whose value is quantized. This is typically one of the last
/// passes done when lowering to express actual quantized arithmetic in a
/// low level representation. Because it modifies the constant, it is
/// destructive and cannot be undone.
std::unique_ptr<OpPassBase<FuncOp>> createConvertConstPass();

} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANTOPS_PASSES_H

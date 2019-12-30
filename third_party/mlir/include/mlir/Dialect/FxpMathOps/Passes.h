//===- Passes.h - Fixed point math passes -----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the passes owned by the FxpMathOps dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FXPMATHOPS_PASSES_H
#define MLIR_DIALECT_FXPMATHOPS_PASSES_H

namespace mlir {
class FuncOp;
template <typename T> class OpPassBase;

namespace fxpmath {

/// Creates a pass that lowers uniform-quantized real math ops to integer
/// arithmetic. This will leave unrecognized real math ops as-is and is
/// typically followed by a pass that lowers any unrecognized ops to a pure
/// floating point form.
OpPassBase<FuncOp> *createLowerUniformRealMathPass();

/// Creates a pass that lowers uniform-quantized qcast/dcast ops to equivalent
/// operations that perform quantize/dequantize.
OpPassBase<FuncOp> *createLowerUniformCastsPass();

} // namespace fxpmath
} // namespace mlir

#endif // MLIR_DIALECT_FXPMATHOPS_PASSES_H

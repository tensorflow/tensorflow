//===- Passes.h - Fixed point math passes -----------------------*- C++ -*-===//
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

//===- Passes.h - Quantization Passes ------ --------------------*- C++ -*-===//
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

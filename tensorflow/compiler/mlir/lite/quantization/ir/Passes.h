/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//
// This file defines all of the passes owned by the quantization dialect. As
// things mature, it is expected that passes specific to certain frontend or
// backend dialects will move to those dialects directly. For now, they are
// incubated here.
//
//===----------------------------------------------------------------------===//

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_IR_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_IR_PASSES_H_

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func

namespace quantfork {

/// Creates a pass that converts quantization simulation operations (i.e.
/// FakeQuant and those like it) to casts into/out of supported QuantizedTypes.
std::unique_ptr<OperationPass<func::FuncOp>> createConvertSimulatedQuantPass();

/// Creates a pass that converts constants followed by a qbarrier to a
/// constant whose value is quantized. This is typically one of the last
/// passes done when lowering to express actual quantized arithmetic in a
/// low level representation. Because it modifies the constant, it is
/// destructive and cannot be undone.
std::unique_ptr<OperationPass<func::FuncOp>> createConvertConstPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/mlir/lite/quantization/ir/Passes.h.inc"

}  // namespace quantfork
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_IR_PASSES_H_

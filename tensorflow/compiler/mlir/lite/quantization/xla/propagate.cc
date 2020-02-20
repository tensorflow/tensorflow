/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies quantization propagation on xla_hlo dialect.
#include <iterator>
#include <string>

#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> disable_per_channel(
    "xla-disable-per-channel", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether disable per-channel quantized weights."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// The quantization propagation Pass.
//
namespace mlir {
namespace xla_hlo {

namespace {

// Applies the quantization propagation on the input function. During the
// propagation, two facts are respected:
// - The quantization type (params) of the ops in the function
// - The quantization spec for the ops
// The propagation results should assign quantization types to all the tensors
// and the two restrictions are respected.
struct PropagateQuantPass : public FunctionPass<PropagateQuantPass> {
  explicit PropagateQuantPass() = default;
  PropagateQuantPass(const PropagateQuantPass &) {}

  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/lite/quantization/xla/op_quant_spec.inc"

void PropagateQuantPass::runOnFunction() {
  FuncOp func = getFunction();
  // XLA only support uint8/uint16 quantization for now.
  ApplyQuantizationParamsPropagation(func, /*is_signed*/ false,
                                     disable_per_channel, GetOpQuantSpec);
}

}  // namespace

// Creates an instance of the xla_hlo dialect quantization propagation pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePropagateQuantPass() {
  return std::make_unique<PropagateQuantPass>();
}

static PassRegistration<PropagateQuantPass> pass(
    "xla-hlo-propagate-quant", "Propagate quantization information");

}  // namespace xla_hlo
}  // namespace mlir

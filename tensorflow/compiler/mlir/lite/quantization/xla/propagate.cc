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
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_context.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/quantization/xla/cpu_device_target.h"

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
struct PropagateQuantPass
    : public PassWrapper<PropagateQuantPass, FunctionPass> {
  explicit PropagateQuantPass() = default;
  PropagateQuantPass(const PropagateQuantPass &) {}

  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/lite/quantization/xla/op_quant_spec.inc"

void PropagateQuantPass::runOnFunction() {
  FuncOp func = getFunction();
  // TODO(fengliuai): deprecate this old code generation path.
  // XLA only support uint8/uint16 quantization for now.
  ApplyQuantizationParamsPropagation(func, /*is_signed*/ false,
                                     disable_per_channel, GetOpQuantSpec);

  CpuDeviceTarget spec(&getContext());
  quant::QuantizeContext ctx(func, spec);

  std::vector<quant::QuantizeRegionOp> work_list = ctx.GetAllOps();
  bool changed = false;
  while (!work_list.empty()) {
    quant::QuantizeRegionOp op = work_list.back();
    work_list.pop_back();

    llvm::SmallVector<Operation *, 4> new_items;
    if (failed(ctx.Handle(op, &new_items, &changed))) {
      // The IR is still valid, thus we shouldn't fail.
      signalPassFailure();
    }
    for (auto item : new_items) {
      if (auto reg = llvm::dyn_cast_or_null<quant::QuantizeRegionOp>(item))
        work_list.push_back(reg);
    }
  }

  if (!changed) return;

  if (failed(ctx.Finalize())) {
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the xla_hlo dialect quantization propagation pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePropagateQuantPass() {
  return std::make_unique<PropagateQuantPass>();
}

static PassRegistration<PropagateQuantPass> pass(
    "xla-hlo-propagate-quant", "Propagate quantization information");

}  // namespace xla_hlo
}  // namespace mlir

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

#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_dialect.h"
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTLAUNCHFUNCTOCUDAGRAPHPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::LaunchFuncOp;

class ConvertLaunchFuncToCudaGraphPass
    : public impl::ConvertLaunchFuncToCudaGraphPassBase<
          ConvertLaunchFuncToCudaGraphPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, runtime::RuntimeDialect>();
  }
};

//===----------------------------------------------------------------------===//

// A sequence of launch func operation to be outlined into cuda graph
// constructor.
struct LaunchFuncSequence {
  llvm::SmallVector<gpu::LaunchFuncOp> ops;
};

// Collect sequences of LaunchFuncOp operations that can be outlined into
// Cuda Graph functions.
//
// TODO(ezhulenev): Do not collect launch func sequences if they are already
// inside a graph capture function.
static llvm::SmallVector<LaunchFuncSequence> CollectLaunchFuncSequences(
    ModuleOp module) {
  llvm::SmallVector<LaunchFuncSequence> seqs;
  llvm::DenseSet<LaunchFuncOp> outlined;

  module.walk([&](LaunchFuncOp op) {
    // This launch operation is a part of already collected sequence.
    if (outlined.contains(op)) return;

    // Find the first LaunchFuncOp in a sequence.
    Operation* first = op;
    while (Operation* prev = first->getPrevNode()) {
      if (!isa<LaunchFuncOp>(prev)) break;
      first = prev;
    }

    // Find the last LaunchFuncOp in a sequence.
    Operation* last = op;
    while (Operation* next = last->getNextNode()) {
      if (!isa<LaunchFuncOp>(next)) break;
      last = next;
    }

    // Skip sequences consisting of a single operation.
    if (first == last) return;

    // Collect all launch func ops.
    LaunchFuncSequence& seq = seqs.emplace_back();

    auto r = llvm::make_range(Block::iterator(first), ++Block::iterator(last));
    llvm::transform(r, std::back_inserter(seq.ops), [&](Operation& op) {
      auto launch = cast<LaunchFuncOp>(op);
      outlined.insert(launch);
      return launch;
    });
  });

  return seqs;
}

//===----------------------------------------------------------------------===//

using xla::runtime::CustomCallDeclarations;

// Given a sequence of LaunchFuncOp operations outline them into a function,
// and replace with an XLA Gpu runtime function call.
static void Outline(CustomCallDeclarations& custom_calls,
                    LaunchFuncSequence& seq) {
  SymbolTable& sym_table = custom_calls.sym_table();
  MLIRContext* ctx = sym_table.getOp()->getContext();

  // Create a fused location out of LaunchFuncOp operations.
  llvm::SmallVector<Location> locations;
  for (auto& op : seq.ops) locations.push_back(op.getLoc());
  ImplicitLocOpBuilder b(FusedLoc::get(ctx, locations), sym_table.getOp());

  // Collect all arguments used by the launch func operations.
  llvm::SetVector<Value> args;
  for (LaunchFuncOp op : seq.ops)
    args.insert(op.operand_begin(), op.operand_end());

  llvm::SmallVector<Type> args_types;
  for (Value arg : args) args_types.push_back(arg.getType());

  // Create a function in the compiled module.
  auto func_type = FunctionType::get(ctx, args_types, TypeRange());
  auto func = b.create<func::FuncOp>("xla.gpu.cuda.graph.capture", func_type);

  // Add graph building function to the module.
  sym_table.insert(func);

  // Export graph builder function to runtime.
  b.setInsertionPoint(func);
  b.create<runtime::ExportOp>(func);

  // Create a custom call declaration corresponding to the outlined graph
  // capture function.
  func::FuncOp graph_launch = custom_calls.GetOrCreate(
      b, "xla.gpu.cuda.graph.launch", args_types, TypeRange());

  // Call the cuda graph launch custom call.
  b.setInsertionPoint(seq.ops.front());
  auto call = b.create<func::CallOp>(graph_launch.getName(), TypeRange(),
                                     args.getArrayRef());
  call->setAttr(b.getStringAttr("capture"), FlatSymbolRefAttr::get(func));

  // At this point we successfully added new functions to the module, so we can
  // move LaunchFuncOp operations from their original location to the graph
  // capture function.

  // Move all launch func operations into the function body.
  Block* body = func.addEntryBlock();
  for (LaunchFuncOp op : seq.ops) op->moveBefore(body, body->end());

  // Replace uses of original values with block arguments.
  for (auto p : llvm::zip(args, func.getArguments()))
    replaceAllUsesInRegionWith(std::get<0>(p), std::get<1>(p), func.getBody());

  // Add a return operation to the graph capture function.
  b.setInsertionPointToEnd(body);
  b.create<func::ReturnOp>(ValueRange());
}

//===----------------------------------------------------------------------===//

void ConvertLaunchFuncToCudaGraphPass::runOnOperation() {
  SymbolTable sym_table(getOperation());
  CustomCallDeclarations custom_calls(std::move(sym_table));

  for (auto& seq : CollectLaunchFuncSequences(getOperation())) {
    Outline(custom_calls, seq);
  }
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertLaunchFuncToCudaGraphPass() {
  return std::make_unique<ConvertLaunchFuncToCudaGraphPass>();
}

}  // namespace gpu
}  // namespace xla

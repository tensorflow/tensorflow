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
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Dominance.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_dialect.h"
#include "tensorflow/compiler/xla/mlir/runtime/ir/rt_ops.h"
#include "tensorflow/compiler/xla/mlir/runtime/utils/custom_calls.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo_gpu/IR/lhlo_gpu_ops.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_OUTLINECUDAGRAPHSPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

using mlir::gpu::LaunchFuncOp;

class OutlineCudaGraphsPass
    : public impl::OutlineCudaGraphsPassBase<OutlineCudaGraphsPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, runtime::RuntimeDialect>();
  }
};

//===----------------------------------------------------------------------===//

struct OpCapturePattern {
  // CUDA-graph-compatible operations can be either moved or cloned into the
  // graph capture function. Most of the operations should be moved, as they
  // have side effects, however small constants and pure operations like
  // `memref.view` can be safely cloned into the graph region. We rely on later
  // dead code elimination to erase them from the "main" function if they are
  // not used by any other operations.
  enum class Capture { kMove, kClone };

  virtual ~OpCapturePattern() = default;
  virtual FailureOr<Capture> match(Operation* op) = 0;
};

using OpCapturePatternSet = std::vector<std::unique_ptr<OpCapturePattern>>;

// A sequence of operations to be outlined into cuda graph capture function.
using CaptureSequence =
    llvm::SmallVector<std::pair<Operation*, OpCapturePattern::Capture>>;

//===----------------------------------------------------------------------===//

template <OpCapturePattern::Capture capture, typename T, typename... Ts>
struct OpCapture : public OpCapturePattern {
  FailureOr<OpCapturePattern::Capture> match(Operation* op) final {
    if (isa<T, Ts...>(op)) return capture;
    return failure();
  }
};

static constexpr auto kMove = OpCapturePattern::Capture::kMove;
static constexpr auto kClone = OpCapturePattern::Capture::kClone;

template <typename T, typename... Ts>
using MoveOp = OpCapture<kMove, T, Ts...>;
template <typename T, typename... Ts>
using CloneOp = OpCapture<kClone, T, Ts...>;

// Capture gpu operations by moving them intp graph capture function.
struct LaunchFuncOpCapture : public MoveOp<LaunchFuncOp> {};
struct ConvOpCapture : public MoveOp<lmhlo_gpu::ConvForwardFusedOp> {};

// Capture pure operations by cloning them into graph capture function.
struct ConstantOpCapture : public CloneOp<arith::ConstantOp> {};
struct ViewOpCapture : public CloneOp<memref::ViewOp> {};

//===----------------------------------------------------------------------===//

// Collect sequences of operations that can be outlined into Cuda Graphs.
static std::vector<CaptureSequence> CollectCaptureSequences(
    DominanceInfo& dominance, ModuleOp module, OpCapturePatternSet& patterns) {
  std::vector<CaptureSequence> seqs;

  // Match given operation with all capture patterns.
  auto match = [&](Operation* op) -> FailureOr<OpCapturePattern::Capture> {
    for (auto& pattern : patterns) {
      if (auto matched = pattern->match(op); succeeded(matched)) return matched;
    }
    return failure();
  };

  // Find graph-compatible sequences of operations in every block.
  module.walk([&](Block* block) {
    CaptureSequence* seq = &seqs.emplace_back();

    for (Operation& op : *block) {
      FailureOr<OpCapturePattern::Capture> matched = match(&op);
      // Append matched operation to the current sequence. We only append
      // operations that must be moved into the graph capture function (ops with
      // side effects), and add cloneable operations later.
      if (succeeded(matched) && *matched == kMove)
        seq->emplace_back(&op, *matched);

      // Skip unsupported operation and start a new sequence.
      if (failed(matched) && !seq->empty()) seq = &seqs.emplace_back();
    }

    // Remove the last sequence if it's empty.
    if (seq->empty()) seqs.pop_back();
  });

  // Remove cloneable operations accidentally captured by the sequence of ops,
  // e.g. we can have `memref.view` between two kernel launch operations that
  // is not used by operations in the captured sequence.
  for (CaptureSequence& seq : seqs) {
    llvm::DenseSet<Operation*> moveable_ops;
    for (auto& [op, capture] : seq)
      if (capture == kMove) moveable_ops.insert(op);

    llvm::erase_if(seq, [&](auto& pair) {
      return pair.second == kClone &&
             llvm::none_of(pair.first->getUsers(), [&](Operation* user) {
               return moveable_ops.contains(user);
             });
    });
  }

  // Try to extend discovered sequences of ops following operands use-def chains
  // and pulling cloneable operations defining operands into the graph capture
  // sequence. In practice we just clone `arith.constant` and `memref.view`
  // operations into the graph capture function, to make it cheaper to compute
  // the hash of the arguments at run time.
  for (CaptureSequence& seq : seqs) {
    llvm::DenseSet<Operation*> seq_ops;  // operations already in `seq`
    llvm::SmallVector<Operation*> worklist;

    // Add operations that define `op` arguments to the worklist.
    auto populate_worklist = [&](Operation* op) {
      for (Value arg : op->getOperands())
        if (Operation* op = arg.getDefiningOp()) worklist.push_back(op);
    };

    for (auto& [op, _] : seq) {
      seq_ops.insert(op);
      populate_worklist(op);
    }

    // Find cloneable ops and group them by block where they are defined.
    llvm::DenseMap<Block*, llvm::SmallVector<Operation*>> cloneable;

    // Traverse use-def chains to collect all cloneable operations.
    while (!worklist.empty()) {
      Operation* op = worklist.pop_back_val();
      if (seq_ops.contains(op)) continue;

      // Check if operation can be cloned into graph capture function.
      if (auto matched = match(op);
          succeeded(matched) && *matched == OpCapturePattern::Capture::kClone) {
        cloneable[op->getBlock()].push_back(op);
        seq_ops.insert(op);
        populate_worklist(op);
      }
    }

    // Traverse blocks according to their dominance to avoid used-before-defined
    // invalid SSA region construction in graph capture function.
    llvm::SmallVector<Block*> blocks;
    for (auto& [block, _] : cloneable) blocks.push_back(block);
    llvm::sort(blocks, [&](Block* a, Block* b) {
      return dominance.properlyDominates(a, b);
    });

    for (Block* block : llvm::reverse(blocks)) {
      // Sort operations according to their original position in the block.
      llvm::sort(cloneable[block], [](Operation* a, Operation* b) {
        return a->isBeforeInBlock(b);
      });

      // Prepend all cloneable operations to the discovered ops sequence.
      auto cloned = llvm::map_range(cloneable[block], [](Operation* op) {
        return std::make_pair(op, OpCapturePattern::Capture::kClone);
      });
      seq.insert(seq.begin(), cloned.begin(), cloned.end());
    }
  }

  return seqs;
}

//===----------------------------------------------------------------------===//

using xla::runtime::CustomCallDeclarations;

static std::vector<Value> GetGraphCaptureFuncArgs(const CaptureSequence& seq) {
  llvm::SetVector<Value> args;

  // Values defined by operations in the capture sequence.
  llvm::DenseSet<Value> defined_by_seq;
  for (auto& [op, _] : seq)
    defined_by_seq.insert(op->result_begin(), op->result_end());

  // Add arguments defined outside of the capture sequence.
  for (auto& [op, _] : seq) {
    auto external_args = llvm::make_filter_range(
        op->getOperands(),
        [&](Value arg) { return !defined_by_seq.contains(arg); });
    args.insert(external_args.begin(), external_args.end());
  }

  return args.takeVector();
}

// Given a sequence of operations, outline them into a graph capture function
// and replace them with an XLA Gpu runtime function call.
static LogicalResult Outline(unsigned ordinal,
                             CustomCallDeclarations& custom_calls,
                             CaptureSequence& seq) {
  // Only operations that have to be moved into the graph capture function
  // represent Gpu computations.
  unsigned num_move_captures = llvm::count_if(seq, [](auto capture) {
    return capture.second == OpCapturePattern::Capture::kMove;
  });
  if (num_move_captures < 2) return failure();

  SymbolTable& sym_table = custom_calls.sym_table();
  MLIRContext* ctx = sym_table.getOp()->getContext();

  // Create a fused location out of LaunchFuncOp operations.
  llvm::SmallVector<Location> locations;
  for (auto& op : seq) locations.push_back(op.first->getLoc());
  ImplicitLocOpBuilder b(FusedLoc::get(ctx, locations), sym_table.getOp());

  // Arguments of the graph capture function.
  std::vector<Value> args = GetGraphCaptureFuncArgs(seq);

  // Create a function in the compiled module.
  auto func = b.create<func::FuncOp>(
      "xla.gpu.cuda.graph.capture",
      FunctionType::get(ctx, TypeRange(ValueRange(args)), TypeRange()));

  // Add graph capture function to the module.
  sym_table.insert(func);

  // Export graph capture function to the runtime.
  b.setInsertionPoint(func);
  b.create<runtime::ExportOp>(func, ordinal);

  // Create a custom call declaration corresponding to the outlined graph
  // capture function.
  func::FuncOp graph_launch = custom_calls.GetOrCreate(
      b, "xla.gpu.cuda.graph.launch", TypeRange(ValueRange(args)), TypeRange());

  // Call the cuda graph launch custom call right before the first moved op.
  auto insertion_point = llvm::find_if(seq, [](auto capture) {
    return capture.second == OpCapturePattern::Capture::kMove;
  });
  b.setInsertionPoint(insertion_point->first);

  auto call = b.create<func::CallOp>(graph_launch.getName(), TypeRange(), args);
  call->setAttr(b.getStringAttr("capture"), FlatSymbolRefAttr::get(func));

  // At this point we successfully added new functions to the module, so we can
  // move or clone captured operations from their original location to the graph
  // capture function.
  Block* body = func.addEntryBlock();

  // We'll need to replace operands of cloned/moved operations inside the graph
  // capture function.
  llvm::SmallVector<std::pair<Value, Value>> mappings;  // {from, to} mappings
  for (auto mapping : llvm::zip(args, func.getArguments()))
    mappings.emplace_back(std::get<0>(mapping), std::get<1>(mapping));

  // Move or clone operations into the graph capture function.
  for (auto& [op, capture] : seq) {
    if (capture == OpCapturePattern::Capture::kMove)
      op->moveBefore(body, body->end());

    if (capture == OpCapturePattern::Capture::kClone) {
      Operation* clone = op->clone();
      OpBuilder::atBlockEnd(body).insert(clone);

      for (auto mapping : llvm::zip(op->getResults(), clone->getResults()))
        mappings.emplace_back(std::get<0>(mapping), std::get<1>(mapping));
    }
  }

  // Update def-use chains inside the graph capture function.
  for (auto mapping : mappings) {
    replaceAllUsesInRegionWith(mapping.first, mapping.second, func.getBody());
  }

  // Add a return operation to the graph capture function.
  b.setInsertionPointToEnd(body);
  b.create<func::ReturnOp>(ValueRange());

  return success();
}

//===----------------------------------------------------------------------===//

void OutlineCudaGraphsPass::runOnOperation() {
  SymbolTable sym_table(getOperation());
  CustomCallDeclarations custom_calls(std::move(sym_table));

  OpCapturePatternSet patterns;
  patterns.emplace_back(new LaunchFuncOpCapture());
  patterns.emplace_back(new ConvOpCapture());
  patterns.emplace_back(new ConstantOpCapture());
  patterns.emplace_back(new ViewOpCapture());

  unsigned ordinal = 1;  // entry point will be exported with ordinal 0
  for (auto& seq : CollectCaptureSequences(getAnalysis<DominanceInfo>(),
                                           getOperation(), patterns)) {
    if (succeeded(Outline(ordinal, custom_calls, seq))) ordinal++;
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createOutlineCudaGraphsPass() {
  return std::make_unique<OutlineCudaGraphsPass>();
}

}  // namespace gpu
}  // namespace xla

/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"  // IWYU pragma: keep
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"

#define GEN_PASS_DECL_CREATEGRAPHREGIONS
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h.inc"

#define GEN_PASS_DEF_CREATEGRAPHREGIONS
#include "tensorflow/compiler/xla/mlir/backends/gpu2/transforms/passes.h.inc"

namespace xla::gpu {
namespace {
using namespace mlir;  // NOLINT

//===----------------------------------------------------------------------===//
// OpCapturePattern
//===----------------------------------------------------------------------===//

struct OpCapturePattern {
  enum class Capture {
    // Operation is supported and will be moved into graph region.
    kMove,
    // Operation is not directly supported by the graph region, however it will
    // not break it, but instead it will be moved to the parent block right
    // before the graph region operation. For example all `memref.view`
    // operations are in this category. XLA:GPU graph region implicitly captures
    // all SSA values defined above, and later when we'll be finalizing graph
    // dispathes captured values will become function call arguments.
    kOutline,
  };

  virtual ~OpCapturePattern() = default;
  virtual FailureOr<Capture> match(Operation* op) = 0;
};

using OpCapturePatternSet = std::vector<std::unique_ptr<OpCapturePattern>>;

template <OpCapturePattern::Capture capture, typename T, typename... Ts>
struct OpCapture : public OpCapturePattern {
  FailureOr<OpCapturePattern::Capture> match(Operation* op) final {
    if (isa<T, Ts...>(op)) return capture;
    return failure();
  }
};

constexpr auto kMove = OpCapturePattern::Capture::kMove;
constexpr auto kOutline = OpCapturePattern::Capture::kOutline;

template <typename T, typename... Ts>
using MoveOp = OpCapture<kMove, T, Ts...>;
template <typename T, typename... Ts>
using OutlineOp = OpCapture<kOutline, T, Ts...>;

//===----------------------------------------------------------------------===//
// Configure ops supported by XLA:GPU graph runtime
//===----------------------------------------------------------------------===//

// Move compiled operations into the graph regions.
struct FusionOpCapture : public MoveOp<lmhlo::FusionOp> {};
struct SortOpCapture : public MoveOp<lmhlo::SortOp> {};

// Outline auxiliary operations out of the graph region.
struct MemrefViewOpCapture : public OutlineOp<memref::ViewOp> {};
struct MemrefCastOpCapture : public OutlineOp<memref::ReinterpretCastOp> {};
struct ArithConstOpCapture : public OutlineOp<arith::ConstantOp> {};

//===----------------------------------------------------------------------===//

// A sequence of operations prepared for constructing a graph region operation.
struct GraphRegion {
  explicit GraphRegion(Block* block) : block(block) {}
  Block* block;
  llvm::SmallVector<std::pair<Operation*, OpCapturePattern::Capture>> ops;
};

// Collect sequences of operations in the module that can be outlined into
// XLA:GPU graph regions.
llvm::SmallVector<GraphRegion> collectGraphRegions(
    ModuleOp module, OpCapturePatternSet& patterns) {
  llvm::SmallVector<GraphRegion> graph_regions;

  // Match given operation with all capture patterns.
  auto match = [&](Operation* op) -> FailureOr<OpCapturePattern::Capture> {
    for (auto& pattern : patterns) {
      if (auto matched = pattern->match(op); succeeded(matched)) return matched;
    }
    return failure();
  };

  // Find graph-compatible sequences of operations in every block.
  module.walk([&](Block* block) {
    GraphRegion* graph_region = &graph_regions.emplace_back(block);

    for (Operation& op : *block) {
      FailureOr<OpCapturePattern::Capture> matched = match(&op);
      if (succeeded(matched)) {
        graph_region->ops.emplace_back(&op, *matched);
      } else if (!graph_region->ops.empty()) {
        graph_region = &graph_regions.emplace_back(block);
      }
    }

    // Remove the last graph region if it's empty.
    if (graph_region->ops.empty()) graph_regions.pop_back();
  });

  return graph_regions;
}

LogicalResult buildGraphRegionOp(GraphRegion& graph_region) {
  // Skip graph regions without any load-bearing ops.
  size_t num_moved_ops = llvm::count_if(
      graph_region.ops, [](auto& op) { return op.second == kMove; });
  if (num_moved_ops == 0) return success();

  // Create a fused location out of moved-in operations
  llvm::SmallVector<Location> locations;
  for (auto& op : graph_region.ops) {
    if (op.second == kOutline) continue;
    locations.push_back(op.first->getLoc());
  }

  MLIRContext* ctx = graph_region.block->getParentOp()->getContext();
  ImplicitLocOpBuilder b(FusedLoc::get(ctx, locations), ctx);
  b.setInsertionPointAfter(graph_region.ops.back().first);

  // Move operations with `kMove` capture into the graph region body.
  auto op = b.create<GraphRegionOp>();
  Block* body = &op.getBody().emplaceBlock();

  for (auto& op : graph_region.ops) {
    if (op.second == kOutline) continue;
    op.first->moveBefore(body, body->end());
  }

  return success();
}

//===----------------------------------------------------------------------===//

class CreateGraphRegionsPass
    : public ::impl::CreateGraphRegionsBase<CreateGraphRegionsPass> {
 public:
  void runOnOperation() override {
    OpCapturePatternSet patterns;

    // TODO(ezhulenev): Make patterns configurable.
    patterns.emplace_back(new FusionOpCapture());
    patterns.emplace_back(new SortOpCapture());

    patterns.emplace_back(new MemrefViewOpCapture());
    patterns.emplace_back(new MemrefCastOpCapture());
    patterns.emplace_back(new ArithConstOpCapture());

    for (auto& graph_region : collectGraphRegions(getOperation(), patterns)) {
      if (failed(buildGraphRegionOp(graph_region))) return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createCreateGraphRegionsPass() {
  return std::make_unique<CreateGraphRegionsPass>();
}

}  // namespace xla::gpu

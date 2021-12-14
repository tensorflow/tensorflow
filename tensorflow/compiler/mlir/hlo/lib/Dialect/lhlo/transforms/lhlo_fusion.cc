/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir-hlo/Dialect/lhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/lhlo/transforms/fusion_utils.h"
#include "mlir-hlo/utils/cycle_detector.h"
#include "mlir/Dialect/Shape/IR/Shape.h"      // TF:llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"              // TF:llvm-project
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"               // TF:local_config_mlir
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project

// This pass has similar functionality of the fusion pass in XLA stack.
// However, unlike XLA, it targets the fully dynamic shape scenario.
// Currently, it implements the kLoop and kInput fusion templates.
// During conversion, it tries to greedily find kLoop/kInput fusion
// patterns.
//
// Similar to XLA, this pass supports fusion pattern having multiple outputs
// if all the shape of outputs are consistent. Following are some examples.
//
//        kLoop                          kInput
// +----+  +----+  +----+    +----+    +----+    +----+
// |elem|  |elem|  |elem|    |elem<----+elem+---->elem+----+
// +-+--+  +-+--+  +-+--+    +-+--+    +----+    +-+--+    |
//   |       |       |         |                   |       |
//   |               |         |                   |       |
// +-v--+    |     +-v--+   +--v---+            +--v---+   |
// |elem+<---+----<+elem|   |reduce|            |reduce|   |
// +-+--+          +-+--+   +--+---+            +--+---+   |
//   |               |         |                   |       |
//   |               |         |                   |       |
//   v               v         v                   v       v
//
// To this end, we also add an simple shape constraint analysis phase.
// For kLoop fusion template, it requires all the outputs of the fused
// pattern have the same shape. However, we don't know the actual value
// of the shape at the compile time in the dynamic shape world.
// Fortunately, we could still infer the relationship among different ops
// according to their shape constraint traits. Currently, We only consider
// shape equality propagation for elementwise ops (assuming that implicit
// shape broadcast is forbidden). The above process could be built on the
// shape dialect once it is ready.
//
// TODO(disc): This file implements fusion on buffer level, re-visit this after
// more shape inference/constraint infras are ready in mhlo level.
// TODO(disc): Not using fusibility interface a.t.m, re-visit this if necessary.

namespace mlir {
namespace lmhlo {
namespace {

struct FusionOptions {
  // Maximum allowed number of arguments per fused kernel. Here arguments
  // include both ready-only buffers and writable buffers.
  int max_num_arguments_per_kernel;
};

// A fusion planner that can propose a fusion plan for a block of ops.
// The fusion plan is consisted of a group of fusion patterns.
//
// Currently all proposed patterns followed xla kLoop/kInput like fusion
// templates while are adapted to the fully dynamic shape world.
//
// kLoop fusion template satisfies:
//   - all ops in the fusion pattern are element-wise.
//   - all the shapes of outputs of fusion pattern are same or have same number
//   of elements, and thus can fit into a same parallel loop.
//
// kInput fusion template satisfies:
//   - any op in the fusion pattern is either element-wise or a reduction.
//   - if a op is a reduction, its output cannot be consumed by other
//     ops in the same fusion pattern.
//   - all the effective shapes of outputs of fusion pattern are same.
//     - For element-wise op, its effective shape is its output shape.
//     - For reduction op, its effective shape is its operand shape.
//   - currently our downstreaming codegen engine only support 2d -> 1d tensor
//   reduction. TODO(disc): lift this limitation.
//     - 2D row reduction: out[i] = sum({in[i][j] for all j})
//     - 2D column reduction: out[j] = sum({in[i][j] for all i}
class FusionPlanner {
 public:
  explicit FusionPlanner(const FusionOptions& options, Block* block)
      : options_(options), block_(block) {
    // Move up metadata-only ops (e.g. dim, shape_of) as far as possible.
    MoveUpMetadataOnlyOpsForFusion();

    for (Operation& op : *block) {
      op_list_.push_back(&op);
    }
    shape_analysis_.reset(new ShapeConstraintAnalysis(op_list_));
    cycle_detector_.reset(new GraphCycles(op_list_.size()));
    BuildNodeMap();
  }

  // Returns a fusion plan if success, otherwise none.
  llvm::Optional<FusionPlan> Run() {
    // Greedily search connected fusible pattern, and ops belonging to
    // a same fusion pattern are grouped into a cluster.
    RunEdgeContractionLoop();

    // After doing edge contraction, each unique cluster having size
    // more than one represents a potential fusion pattern.
    // We collect all these clusters and construct a fusion plan.
    FusionPlan plan;
    DenseSet<Cluster*> seen_clusters;
    for (Operation* op : op_list_) {
      Cluster* cluster = GetClusterForNode(op);
      if (!seen_clusters.insert(cluster).second) continue;
      FusionPattern& fusion_pattern = cluster->fused_pattern();
      // Make sure the ops in a fusion pattern are in topological ordering.
      fusion_pattern.sortFusionOpListBy(op_to_node_id_);
      if (!fusion_pattern.isFusible() || fusion_pattern.effectiveSize() <= 1) {
        continue;
      }
      plan.emplace_back(fusion_pattern);
    }

    // Re-order ops inside the blocks to make sure all producers are placed
    // before its consumers after fusion.
    ReorderOperationsInsideBlock();
    return plan;
  }

  // Returns the op_list this planner operates on.
  const SmallVectorImpl<Operation*>& op_list() const { return op_list_; }

 private:
  // Represent a (partial) fused pattern
  class Cluster {
   public:
    Cluster(int node_id, FusionPlanner* planner)
        : node_id_(node_id), pattern_(planner->op_list()[node_id]) {}

    // Merges `other` into this cluster, and clears `other`.
    void Merge(Cluster* other) {
      pattern_.mergeInplace(other->fused_pattern());
    }

    // The number of nodes in this cluster.
    int cluster_size() { return pattern_.size(); }

    // The ID of the cluster as represented in `cycle_detector_`.
    int cycles_graph_node_id() const { return node_id_; }

    // Sets the ID of the cluster as represented in `cycle_detector_`.
    void set_cycles_graph_node_id(int cycles_graph_node_id) {
      node_id_ = cycles_graph_node_id;
    }

    // Currently the fused pattern this cluster holds.
    FusionPattern& fused_pattern() { return pattern_; }

   private:
    // ID of the representative node of this cluster.
    int node_id_;

    // the fused pattern this cluster holds.
    FusionPattern pattern_;
  };

 private:
  // Returns a new cluster with specified `cycles_graph_node_id`
  Cluster* MakeCluster(int cycles_graph_node_id) {
    cluster_storage_.emplace_back(new Cluster(cycles_graph_node_id, this));
    return cluster_storage_.back().get();
  }

  // Metadata ops (e.g. shapeOf, dimOp) don't change data thus we move forward
  // them as far as possible inside the same block to enable more fusion
  // opportunities.
  void MoveUpMetadataOnlyOpsForFusion() {
    SmallVector<Operation*, 4> ops;
    for (Operation& op : *block_) {
      ops.push_back(&op);
    }

    auto inBlock = [&](Operation* op, Block* block) {
      return op && op->getBlock() == block;
    };

    for (Operation* op : ops) {
      Block* block = op->getBlock();
      if (isa<shape::ShapeOfOp>(op)) {
        Operation* definingOp = op->getOperand(0).getDefiningOp();
        if (!inBlock(definingOp, block)) {
          op->moveBefore(block, block->begin());
        } else {
          op->moveAfter(definingOp);
        }
      } else if (isa<memref::DimOp>(op)) {
        Operation* firstOperandOp = op->getOperand(0).getDefiningOp();
        Operation* secondOperandOp = op->getOperand(1).getDefiningOp();
        if (!inBlock(firstOperandOp, block) &&
            !inBlock(secondOperandOp, block)) {
          op->moveBefore(block, block->begin());
        } else if (!inBlock(firstOperandOp, block)) {
          op->moveAfter(secondOperandOp);
        } else if (!inBlock(secondOperandOp, block)) {
          op->moveAfter(firstOperandOp);
        } else if (firstOperandOp->isBeforeInBlock(secondOperandOp)) {
          op->moveAfter(secondOperandOp);
        } else {
          op->moveAfter(firstOperandOp);
        }
      }
    }
  }

  // Returns all the values touched by this op or its nested ops.
  SmallVector<Value, 4> GetAllPossibleUsedValues(Operation* op) {
    SmallVector<Value, 4> values;
    op->walk([&](Operation* nest_op) {
      for (Value v : nest_op->getOperands()) {
        values.push_back(v);
      }
    });
    return values;
  }

  // Builds the initial dependency graph.
  void BuildNodeMap() {
    int num_nodes = op_list_.size();
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
      Operation* op = op_list_[node_id];
      MakeCluster(node_id);
      op_to_node_id_[op] = node_id;
      leader_for_node_.insert(node_id);
      for (Value operand : GetAllPossibleUsedValues(op)) {
        Operation* operand_op = FindLastWriter(operand);
        // Only consider the operand_op inside the target block.
        auto iter = op_to_node_id_.find(operand_op);
        if (iter == op_to_node_id_.end()) {
          continue;
        }
        // Add an edge to connect the last writer and the current consumer.
        cycle_detector_->InsertEdge(iter->second, node_id);
      }

      // For some ops (e.g. lmhlo ops), some operands are the output memrefs
      // Thus these operands are supposed to be updated.
      // Suppose that a op (or its nested ops) can only write the buffers
      // explicit passed in as operands of this op.
      int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
      for (Value v : op->getOperands().drop_front(num_input_operand)) {
        auto it = last_writer_.try_emplace(v, op);
        (void)it;
        // Currently, a buffer is only supposed to be written once (as the
        // output operand of one lmhlo op).
        assert(it.second);
      }
    }
  }

  // Returns the cluster contains this op.
  Cluster* GetClusterForNode(Operation* n) {
    int id = op_to_node_id_[n];
    id = leader_for_node_.getLeaderValue(id);
    return cluster_storage_[id].get();
  }

  // Returns the cluster contains the op having `node_id`.
  Cluster* GetClusterForCyclesGraphNode(int node_id) {
    return cluster_storage_[leader_for_node_.getLeaderValue(node_id)].get();
  }

  // Merges the clusters `cluster_from` and `cluster_to`.
  bool MergeClusters(Cluster* cluster_from, Cluster* cluster_to) {
    int from = cluster_from->cycles_graph_node_id();
    int to = cluster_to->cycles_graph_node_id();

    auto optional_merged_node = cycle_detector_->ContractEdge(from, to);
    if (!optional_merged_node.hasValue()) {
      llvm::dbgs() << "Could not contract " << from << " -> " << to
                   << " because contracting the edge would create a cycle.";
      return false;
    }

    // Merge the clusters.
    cluster_from->Merge(cluster_to);
    cluster_from->set_cycles_graph_node_id(*optional_merged_node);

    // Merge the UnionFind Set.
    leader_for_node_.unionSets(from, to);
    return true;
  }

  using FnTy = llvm::function_ref<bool(Cluster*, Cluster*)>;
  bool ForEachEdgeInPostOrder(FnTy fn, bool enable_cross_fusion = false) {
    bool changed = false;
    for (int32_t node : cycle_detector_->AllNodesInPostOrder()) {
      Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
      // Make a copy of the set of successors because we may modify the graph in
      // TryToContractEdge.
      std::vector<int32_t> successors_copy =
          cycle_detector_->SuccessorsCopy(cluster_from->cycles_graph_node_id());

      for (int to : successors_copy) {
        Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
        bool contracted_edge = fn(cluster_from, cluster_to);
        changed |= contracted_edge;
      }
    }

    if (!enable_cross_fusion) return changed;

    // To enable even more fusion opportunities (e.g. horizontal fusion)
    for (int32_t lhs : cycle_detector_->AllNodesInPostOrder()) {
      Cluster* cluster_lhs = GetClusterForCyclesGraphNode(lhs);
      if (!cluster_lhs) {
        continue;
      }

      for (int32_t rhs : cycle_detector_->AllNodesInPostOrder()) {
        Cluster* cluster_rhs = GetClusterForCyclesGraphNode(rhs);
        if (!cluster_rhs || cluster_lhs == cluster_rhs) {
          continue;
        }

        bool contracted_edge = fn(cluster_lhs, cluster_rhs);
        changed |= contracted_edge;
      }
    }

    return changed;
  }

  // This function check if fusing `from` with `to` is valid and if so perform
  // the merge. The validity is based on the operations in the clusters and
  // the compatibility of the shapes of the outputs of the would-be fused
  // clusters.
  // Returns true is the merge was performed.
  bool TryToContractEdge(Cluster* from, Cluster* to) {
    // Try merge and check if valid.
    if (!from->fused_pattern().isMergeable(to->fused_pattern())) return false;
    FusionPattern fused_pattern =
        from->fused_pattern().merge(to->fused_pattern());
    auto& op_list = fused_pattern.getOpList();
    auto& operands = fused_pattern.getOperands();
    auto& results = fused_pattern.getResults();

    if (results.size() + operands.size() >
        options_.max_num_arguments_per_kernel) {
      // some backend devices (e.g. GPU) do not support a kernel with
      // too many arguments.
      return false;
    }

    // We currently do not support a constant op as final output of a fusion
    // pattern.
    // TODO(disc): copy small const in case necessary.
    for (Value result : results) {
      Operation* result_op = FindLastWriter(result);
      assert(result_op);
      if (isa<lmhlo::ConstOp>(result_op)) {
        return false;
      }
    }

    // ReduceOp can not have consumer within the fusion pattern.
    for (Operation* op : op_list) {
      if (!isa<lmhlo::ReduceOp>(op)) continue;
      int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
      for (Value v : op->getOperands().drop_front(num_input_operand)) {
        for (Operation* user : getValueUsers(v)) {
          if (user == op) continue;
          if (std::find(op_list.begin(), op_list.end(), user) !=
              op_list.end()) {
            return false;
          }
        }
      }
    }

    // All outputs of a fusion pattern should have compatible shape.
    // Here `compatible` means:
    // - if `to` and `from` are both kInput fusion, all output should have same
    // shape.
    // - otherwise, all output should have same number of elements.

    // No outside users, these ops may be eliminated. We fused it here and let
    // latter pass to do such DCE.
    if (results.empty()) return true;

    bool check_same_shape = (to->fused_pattern().isKInputFusion() &&
                             from->fused_pattern().isKInputFusion());
    auto get_effective_shape = [&](Value v) {
      auto result_op = FindLastWriter(v);
      assert(result_op);
      // effective shape of reduce op is its operand's shape.
      return isa<lmhlo::ReduceOp>(result_op) ? result_op->getOperand(0) : v;
    };

    Value ref_shape = get_effective_shape(results[0]);
    if (!llvm::all_of(results, [&](Value result) {
          Value shape = get_effective_shape(result);
          return check_same_shape
                     ? shape_analysis_->HasSameShape(ref_shape, shape)
                     : shape_analysis_->HasSameNumElements(ref_shape, shape);
        })) {
      return false;
    }

    return MergeClusters(from, to);
  }

  // Greedily fuse connected node.
  bool RunEdgeContractionLoop() {
    using std::placeholders::_1;
    using std::placeholders::_2;
    bool changed = false;

    // Run fusion pass repeatedly until nothing to be fused
    while (ForEachEdgeInPostOrder(
        std::bind(&FusionPlanner::TryToContractEdge, this, _1, _2), false)) {
      // empty statement by design
    }
    return changed;
  }

  // Here `value` is supported to be a pointer to buffer.
  // Returns the defining op of `value `if no known op updates the buffer,
  // otherwise returns the last op that updates the buffer pointed by the
  // `value`.
  Operation* FindLastWriter(Value value) {
    auto it = last_writer_.find(value);
    if (it != last_writer_.end()) {
      return it->second;
    }
    return value.getDefiningOp();
  }

  // Re-order ops inside the block to make sure that producers are before
  // consumers after fusion.
  void ReorderOperationsInsideBlock() {
    auto reorder_func = [&](Cluster* from, Cluster* to) {
      FusionPattern& from_pattern = from->fused_pattern();
      FusionPattern& to_pattern = to->fused_pattern();

      Operation* last_op_in_from = from_pattern.getOpList().back();
      for (Operation* op : llvm::reverse(to_pattern.getOpList())) {
        if (!last_op_in_from->isBeforeInBlock(op))
          op->moveAfter(last_op_in_from);
      }
      return false;
    };

    ForEachEdgeInPostOrder(reorder_func);
  }

  // hyper-parameters that controls the behaviour of the fusion planner.
  FusionOptions options_;

  // The block that fusion planner works on.
  Block* block_;

  // Ops inside the block
  SmallVector<Operation*, 4> op_list_;

  // Shape equality checker
  std::unique_ptr<ShapeConstraintAnalysis> shape_analysis_;

  // op -> node_id
  DenseMap<Operation*, int> op_to_node_id_;

  // make sure not introduce cycle after fusion
  std::unique_ptr<GraphCycles> cycle_detector_;
  std::vector<std::unique_ptr<Cluster>> cluster_storage_;

  // a UnionFind set. Each set represents a (partial) fused pattern
  // and has a leader as representation.
  EquivalenceClasses<int32_t> leader_for_node_;

  // Here `value` is supported to be a pointer to buffer.
  // Returns the defining op of `value `if no known op updates the buffer,
  // otherwise returns the last op that updates the buffer pointed by the
  // `value`.
  DenseMap<Value, Operation*> last_writer_;
};

struct LhloFusionPass : public LhloFusionPassBase<LhloFusionPass> {
  using LhloFusionPassBase<LhloFusionPass>::LhloFusionPassBase;
  explicit LhloFusionPass(int max_num_arguments_per_kernel)
      : LhloFusionPassBase<LhloFusionPass>::LhloFusionPassBase() {
    this->max_num_arguments_per_kernel_ = max_num_arguments_per_kernel;
  }

  void runOnFunction() override {
    FuncOp func = getFunction();

    // collect all blocks inside the function.
    SmallVector<Block*, 4> blocks;
    CollectBlocksInsideFunction(func, blocks);

    // process each block and do fusion within a block.
    FusionOptions options;
    options.max_num_arguments_per_kernel = max_num_arguments_per_kernel_;
    for (Block* block : blocks) {
      FusionPlanner planner(options, block);
      llvm::Optional<FusionPlan> plan = planner.Run();
      if (!plan) {
        emitError(func.getLoc(),
                  "an error occurs while trying to find fusion candidates");
        signalPassFailure();
        return;
      }
      if (!ApplyFusionPlan(*plan)) {
        emitError(func.getLoc(), "apply fusion plan failed");
        signalPassFailure();
        return;
      }
    }
  }

  bool ApplyFusionPlan(FusionPlan& plan) {
    for (FusionPattern& pattern : plan) {
      auto& op_list = pattern.getOpList();
      OpBuilder b(op_list.back());

      // Get the fused locations
      SmallVector<Location, 4> locations;
      locations.reserve(op_list.size());
      for (Operation* op : op_list) {
        locations.push_back(op->getLoc());
      }
      Location fused_loc =
          FusedLoc::get(op_list.back()->getContext(), locations);

      // Move ops inside fusion pattern to the region attached to the fusion op.
      FusionOp fusion = b.create<lmhlo::FusionOp>(fused_loc);
      Region& region = fusion.region();
      Block& block = region.front();
      for (Operation* op : llvm::reverse(op_list)) {
        op->moveBefore(&block, block.begin());
      }
    }
    return true;
  }

  void CollectBlocksInsideFunction(FuncOp op, SmallVectorImpl<Block*>& blocks) {
    op.walk([&](Block* block) {
      // It does not make sense to fuse the region attached to these ops.
      if (!isa<lmhlo::ReduceOp, lmhlo::FusionOp>(block->getParentOp()))
        blocks.push_back(block);
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLhloFusionPass(
    int max_num_arguments_per_kernel) {
  return std::make_unique<LhloFusionPass>(max_num_arguments_per_kernel);
}

}  // namespace lmhlo
}  // namespace mlir

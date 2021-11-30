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

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/utils/cycle_detector.h"
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
// according to their shape constrain traits. Currently, We only consider
// shape equality propagation for elementwise ops (assuming that implicit
// shape broadcast is forbidden). The above process could be built on the
// shape dialect once it is ready.

namespace mlir {
namespace mhlo {
namespace {

using llvm::EquivalenceClasses;
using FusionPattern = std::vector<Operation*>;
using FusionPlan = std::vector<FusionPattern>;

// To support using EquivalenceClasses for Value
class ValueWrapper {
 public:
  explicit ValueWrapper(Value value) : value_(std::move(value)) {}

  Value getValue() const { return value_; }

  bool operator==(const ValueWrapper& rhs) const {
    return getValue() == rhs.getValue();
  }

 private:
  Value value_;
};

bool operator<(const ValueWrapper& lhs, const ValueWrapper& rhs) {
  auto lhs_value = lhs.getValue().getAsOpaquePointer();
  auto rhs_value = rhs.getValue().getAsOpaquePointer();
  return lhs_value < rhs_value;
}

bool IsMhlo(Operation* op) {
  Dialect* dialect = op->getDialect();
  return dialect && isa<MhloDialect>(dialect);
}

bool IsFusibleWithOperand(Operation* op) {
  return IsMhlo(op) &&
         (op->hasTrait<::mlir::OpTrait::Elementwise>() || isa<ReduceOp>(op));
}

bool IsFusibleWithConsumer(Operation* op) {
  return IsMhlo(op) && (op->hasTrait<::mlir::OpTrait::Elementwise>() ||
                        matchPattern(op, m_Constant()));
}

Value InferEffectiveWorkloadShape(Value v) {
  Operation* op = v.getDefiningOp();
  return op && isa<ReduceOp>(op) ? op->getOperand(0) : v;
}

bool IsFusible(Operation* op) {
  return matchPattern(op, m_Constant()) || IsFusibleWithConsumer(op) ||
         IsFusibleWithOperand(op);
}

SmallVector<Value, 4> GetInputsOfFusionPattern(const FusionPattern& pattern) {
  SmallVector<Value, 4> inputs;
  DenseSet<Value> input_set;
  DenseSet<Operation*> op_set;
  for (Operation* op : pattern) {
    bool inserted = op_set.insert(op).second;
    (void)inserted;
    assert(inserted && "FusionPattern contains duplicate operations");
  }

  for (Operation* op : pattern) {
    for (Value operand : op->getOperands()) {
      Operation* operand_op = operand.getDefiningOp();
      if (op_set.find(operand_op) != op_set.end()) {
        // skip if defining op is in the pattern
        continue;
      }
      if (input_set.insert(operand).second) {
        inputs.push_back(operand);
      }
    }
  }
  return inputs;
}

SmallVector<Value, 4> GetOutputsOfFusionPattern(const FusionPattern& pattern) {
  SmallVector<Value, 4> outputs;
  DenseSet<Operation*> op_set;
  for (Operation* op : pattern) {
    bool inserted = op_set.insert(op).second;
    (void)inserted;
    assert(inserted && "FusionPattern contains duplicate operations");
  }

  for (Operation* op : pattern) {
    for (Value result : op->getResults()) {
      bool has_external_user = llvm::any_of(
          result.getUses(),
          [&](OpOperand& use) { return !op_set.count(use.getOwner()); });
      if (has_external_user) {
        outputs.push_back(result);
      }
    }
  }
  return outputs;
}

FusionPattern MergeFusionPattern(const FusionPattern& lhs,
                                 const FusionPattern& rhs) {
  FusionPattern pattern(lhs);
  pattern.insert(pattern.end(), rhs.begin(), rhs.end());
  return pattern;
}

inline int EffectiveSize(const FusionPattern& pattern) {
  return llvm::count_if(
      pattern, [](Operation* op) { return !matchPattern(op, m_Constant()); });
}

// This is an simple shape constraint analysis, which is used to
// guide fusion decision (e.g. we only fuse shape-compatible ops).
//
// Currently, We only consider shape equality propagation based
// on the shape constrain traits of elementwise ops (assuming that
// implicit shape broadcast is forbidden).
class ShapeConstraintAnalysis {
 public:
  explicit ShapeConstraintAnalysis(const SmallVectorImpl<Operation*>& op_list) {
    PropagateEquality(op_list);
  }

  // Returns true is `lhs` and `rhs` are supposed to have same shape.
  bool HasSameShape(Value lhs, Value rhs) {
    return impl_.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs));
  }

 private:
  // shape equality propagation based on the shape constrains of
  // elementwise ops.
  void PropagateEquality(const SmallVectorImpl<Operation*>& op_list) {
    bool converged = true;
    do {
      converged = true;
      auto update = [&](Value lhs, Value rhs) {
        if (!impl_.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs))) {
          converged = false;
          impl_.unionSets(ValueWrapper(lhs), ValueWrapper(rhs));
        }
      };
      for (Operation* op : op_list) {
        auto op_fusibility = dyn_cast<InferShapeEqualityOpInterface>(op);
        if (!op_fusibility) continue;
        int numInput = op->getNumOperands();
        int numOutput = op->getNumResults();
        // shape equality propagation between inputs.
        for (int input1 = 0; input1 < numInput; ++input1)
          for (int input2 = input1 + 1; input2 < numInput; ++input2)
            if (op_fusibility.inferInputsShapeEquality(input1, input2))
              update(op->getOperand(input1), op->getOperand(input2));

        // shape equality propagation between outputs.
        for (int output1 = 0; output1 < numOutput; ++output1)
          for (int output2 = output1 + 1; output2 < numOutput; ++output2)
            if (op_fusibility.inferOutputsShapeEquality(output1, output2))
              update(op->getResult(output1), op->getResult(output2));

        // shape equality propagation between input and output.
        for (int input = 0; input < numInput; ++input)
          for (int output = 0; output < numOutput; ++output)
            if (op_fusibility.inferInputOutputShapeEquality(input, output))
              update(op->getOperand(input), op->getResult(output));
      }
    } while (!converged);
  }

  // a UnionFind set
  EquivalenceClasses<ValueWrapper> impl_;
};

// A fusion planner that can propose a fusion plan for a block of ops.
// The fusion plan is consisted of a group of fusion patterns.
//
// Currently all proposed patterns followed xla kLoop/kInput like fusion
// templates while are adapted to the fully dynamic shape world.
//
// kLoop fusion template satifies:
//   - all ops in the fusion pattern are element-wise.
//   - all the shapes of outputs of fusion pattern are same, and thus can
//     fit into a same parallel loop.
//
// kInput fusion template satifies:
//   - any op in the fusion pattern is either element-wise or a reduction.
//   - if a op is a reduction, its output cannot be consumered by other
//     ops in the same fusion pattern.
//   - all the effective shapes of outputs of fusion pattern are same.
//     - For element-wise op, its effective shape is its output shape.
//     - For reduction op, its effective shape is its operand shape.
class FusionPlanner {
 public:
  explicit FusionPlanner(const SmallVectorImpl<Operation*>& op_list)
      : op_list_(op_list),
        shape_analysis_(op_list),
        cycle_detector_(op_list.size()) {
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
    //
    // Note that the ops in a fusion pattern are in topological ordering.
    FusionPlan plan;
    DenseMap<int, int> pattern_ids;
    for (Operation* op : op_list_) {
      Cluster* cluster = GetClusterForNode(op);
      int node_id = cluster->cycles_graph_node_id();
      if (!IsFusible(op_list_[node_id]) ||
          EffectiveSize(GetClusterForNode(op)->fused_pattern()) <= 1) {
        continue;
      }
      if (!pattern_ids.count(node_id)) {
        int pattern_id = pattern_ids.size();
        pattern_ids[node_id] = pattern_id;
        plan.emplace_back();
      }
      plan[pattern_ids[node_id]].push_back(op);
    }
    return plan;
  }

  // Returns the op_list this planner operates on.
  const SmallVectorImpl<Operation*>& op_list() const { return op_list_; }

 private:
  // Represent a (partial) fused pattern
  class Cluster {
   public:
    Cluster(int node_id, FusionPlanner* planner) : node_id_(node_id) {
      const SmallVectorImpl<Operation*>& op_list = planner->op_list();
      pattern_.push_back(op_list[node_id]);
    }

    // Merges `other` into this cluster, and clears `other`.
    void Merge(Cluster* other) {
      pattern_.insert(pattern_.end(), other->pattern_.begin(),
                      other->pattern_.end());
      other->pattern_.clear();
    }

    // The number of nodes in this cluster.
    int cluster_size() const { return pattern_.size(); }

    // The ID of the cluster as represented in `cycle_detector_`.
    int cycles_graph_node_id() const { return node_id_; }

    // Sets the ID of the cluster as represented in `cycle_detector_`.
    void set_cycles_graph_node_id(int cycles_graph_node_id) {
      node_id_ = cycles_graph_node_id;
    }

    // Currently the fused pattern this cluster holds.
    const FusionPattern& fused_pattern() { return pattern_; }

   private:
    // ID of the representative node of this cluster.
    int node_id_;

    // the fused pattern this cluster holds.
    FusionPattern pattern_;
  };

 private:
  Cluster* MakeCluster(int cycles_graph_node_id) {
    cluster_storage_.emplace_back(new Cluster(cycles_graph_node_id, this));
    return cluster_storage_.back().get();
  }

  void BuildNodeMap() {
    int num_nodes = op_list_.size();
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
      Operation* op = op_list_[node_id];
      MakeCluster(node_id);
      op_to_node_id_[op] = node_id;
      leader_for_node_.insert(node_id);
      for (Value operand : op->getOperands()) {
        Operation* operand_op = operand.getDefiningOp();
        if (operand_op == nullptr) {
          // skip block argument
          continue;
        }
        auto iter = op_to_node_id_.find(operand_op);
        assert(iter != op_to_node_id_.end());
        cycle_detector_.InsertEdge(iter->second, node_id);
      }
    }
  }

  // Returns the cluster contains this op.
  Cluster* GetClusterForNode(Operation* n) {
    int id = op_to_node_id_.at(n);
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

    auto optional_merged_node = cycle_detector_.ContractEdge(from, to);
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

  template <typename FnTy>
  bool ForEachEdgeInPostOrder(FnTy fn) {
    bool changed = false;
    for (int32_t node : cycle_detector_.AllNodesInPostOrder()) {
      Cluster* cluster_from = GetClusterForCyclesGraphNode(node);
      // Make a copy of the set of successors because we may modify the graph in
      // TryToContractEdge.
      std::vector<int32_t> successors_copy =
          cycle_detector_.SuccessorsCopy(cluster_from->cycles_graph_node_id());

      for (int to : successors_copy) {
        Cluster* cluster_to = GetClusterForCyclesGraphNode(to);
        bool contracted_edge = fn(cluster_from, cluster_to);
        changed |= contracted_edge;
      }
    }

    return changed;
  }

  // returns the outputs if two cluster were merged
  SmallVector<Value, 4> GetResultsOfFusedPattern(Cluster* from, Cluster* to) {
    FusionPattern fused_pattern =
        MergeFusionPattern(from->fused_pattern(), to->fused_pattern());
    return GetOutputsOfFusionPattern(fused_pattern);
  }

  // This function check if fusing `from` with `to` is valid and if so perform
  // the merge. The validity is based on the operations in the clusters and
  // the compatibility of the shapes of the outputs of the would-be fused
  // clusters.
  // Returns true is the merge was performed.
  bool TryToContractEdge(Cluster* from, Cluster* to) {
    int node_to = to->cycles_graph_node_id();
    int node_from = from->cycles_graph_node_id();

    // Both node_to and node_from should be fusible
    if (!IsFusible(op_list_[node_to]) || !IsFusible(op_list_[node_from])) {
      return false;
    }

    if (!IsFusibleWithConsumer(op_list_[node_from])) {
      // This op cannot be fused with its consumers.
      return false;
    }

    if (!IsFusibleWithOperand(op_list_[node_to])) {
      // This op cannot be fused with its operands.
      return false;
    }

    // Output shapes of a fusion pattern should be compatible as described in
    // the document of this class.
    SmallVector<Value, 4> results = GetResultsOfFusedPattern(from, to);

    Value ref = InferEffectiveWorkloadShape(results[0]);
    if (!llvm::all_of(results, [&](Value result) {
          Value val = InferEffectiveWorkloadShape(result);
          return shape_analysis_.HasSameShape(ref, val);
        })) {
      return false;
    }

    return MergeClusters(from, to);
  }

  // Greedily fuse connected node.
  bool RunEdgeContractionLoop() {
    using std::placeholders::_1;
    using std::placeholders::_2;
    return ForEachEdgeInPostOrder(
        std::bind(&FusionPlanner::TryToContractEdge, this, _1, _2));
  }

  const SmallVectorImpl<Operation*>& op_list_;

  // Shape equality checker
  ShapeConstraintAnalysis shape_analysis_;

  // op -> node_id
  std::unordered_map<Operation*, int> op_to_node_id_;

  // make sure not introduce cycle after fusion
  GraphCycles cycle_detector_;
  std::vector<std::unique_ptr<Cluster>> cluster_storage_;

  // a UnionFind set. Each set represents a (partial) fused pattern
  // and has a leader as representation.
  EquivalenceClasses<int32_t> leader_for_node_;
};

struct MhloFusionPass : public MhloFusionPassBase<MhloFusionPass> {
  void runOnFunction() override {
    FuncOp func = getFunction();
    if (!IsTargetFunc(func)) {
      return;
    }

    // process each block and do fusion within a block.
    for (Block& block : func) {
      SmallVector<Operation*, 4> op_list;
      for (Operation& op : block) {
        op_list.push_back(&op);
      }

      FusionPlanner planner(op_list);
      llvm::Optional<FusionPlan> plan = planner.Run();
      if (!plan) {
        emitError(func.getLoc(), "can't find a fusion plan");
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

  bool IsTargetFunc(FuncOp func) {
    int num_fusible_ops = 0;
    bool is_target_func = false;
    // We only process the function having enough candidates
    func.walk([&](Operation* op) {
      num_fusible_ops += static_cast<int>(
          dyn_cast<InferShapeEqualityOpInterface>(op) != nullptr);
      is_target_func = (num_fusible_ops > 1);
      // early stop
      if (is_target_func) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return is_target_func;
  }

  bool ApplyFusionPlan(const FusionPlan& plan) {
    for (const FusionPattern& pattern : plan) {
      OpBuilder b(pattern.back());

      SmallVector<Location, 4> locations;
      locations.reserve(pattern.size());
      for (Operation* op : pattern) {
        locations.push_back(op->getLoc());
      }
      Location fused_loc =
          FusedLoc::get(pattern.back()->getContext(), locations);

      SmallVector<Value, 4> inputs = GetInputsOfFusionPattern(pattern);
      SmallVector<Value, 4> outputs = GetOutputsOfFusionPattern(pattern);
      SmallVector<Type, 4> output_types;
      output_types.reserve(outputs.size());
      for (Value v : outputs) {
        output_types.push_back(v.getType());
      }

      //      /-----\
      //     /       V
      // A(fused) -- B -- C(fused) -- D(fused)
      // mlir ops Adjacency List likely above
      // the B is consumer of fused A, so B need move behind D
      // because fusion op create at D's location
      DenseSet<Operation*> fused_set(pattern.begin(), pattern.end());
      DenseSet<Operation*> consumers_set;
      SmallVector<Operation*, 4> consumers_vec;
      auto first_iter = pattern.front()->getIterator();
      auto last_iter = pattern.back()->getIterator();
      for (Operation& cur_op : llvm::make_range(first_iter, last_iter)) {
        // isn't fused op && consumer's op
        // move this after fusion op
        if (!fused_set.contains(&cur_op)) {
          // fused op's consumer or consumer's consumer
          bool is_consumer = llvm::any_of(
              cur_op.getOperands(), [&fused_set, &consumers_set](Value v) {
                auto op = v.getDefiningOp();
                return fused_set.contains(op) || consumers_set.contains(op);
              });
          if (is_consumer) {
            consumers_set.insert(&cur_op);
            consumers_vec.push_back(&cur_op);
          }
        }
      }
      for (auto op : llvm::reverse(consumers_vec)) {
        op->moveAfter(pattern.back());
      }

      FusionOp fusion =
          b.create<mhlo::FusionOp>(fused_loc, output_types, inputs);
      Region& region = fusion.fused_computation();
      region.push_back(new Block);
      Block& block = region.front();
      for (Operation* op : pattern) {
        op->moveBefore(&block, block.end());
      }
      b.setInsertionPoint(&block, block.end());
      b.create<mhlo::ReturnOp>(fused_loc, outputs);

      for (auto output_and_result : llvm::zip(outputs, fusion.getResults())) {
        Value output = std::get<0>(output_and_result);
        Value fusion_result = std::get<1>(output_and_result);
        for (OpOperand& use : llvm::make_early_inc_range(output.getUses())) {
          if (use.getOwner()->getBlock() != &block) use.set(fusion_result);
        }
      }
    }
    return true;
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createMhloFusionPass() {
  return std::make_unique<MhloFusionPass>();
}

}  // namespace mhlo
}  // namespace mlir

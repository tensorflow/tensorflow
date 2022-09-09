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

#include <algorithm>
#include <iterator>
#include <queue>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/types/optional.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/name_utils.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

// This value dictates how many times during layout propagation we allow
// fixing of oscillatory behaviors.
constexpr int kLayoutPropagationMaxStages = 3;

bool AllOpResultsHaveLayouts(
    mlir::ModuleOp* module, mlir::Dialect* tf_dialect,
    const llvm::DenseMap<mlir::Value, Layout>& layouts) {
  const auto& result = module->walk([&](mlir::Operation* op) {
    if (op->getDialect() != tf_dialect ||
        mlir::isa<mlir::TF::DTensorLayout>(op))
      return mlir::WalkResult::advance();
    for (const auto& result : op->getOpResults()) {
      if (layouts.find(result) == layouts.end()) {
        op->emitOpError() << "missing layout for result "
                          << result.getResultNumber();
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return !result.wasInterrupted();
}

void UpdateLayoutForSkippedOps(
    mlir::OpOperand& operand,
    const llvm::DenseMap<llvm::StringRef, mlir::Operation*>& func_to_caller,
    const Layout& layout_to_copy,
    llvm::DenseMap<mlir::Value, Layout>& layouts) {
  llvm::SmallVector<mlir::Value, 4> skipped_values;
  TraceUseToNextTFOp(&operand, func_to_caller, &skipped_values);
  for (const mlir::Value& skipped_value : skipped_values)
    if ((!skipped_value.isa<mlir::OpResult>() ||
         !mlir::isa<mlir::TF::DTensorLayout, mlir::tf_device::ClusterOp>(
             skipped_value.getDefiningOp())) &&
        layouts.find(skipped_value) == layouts.end())
      // TraceUseToNextTFOp's contract is that it only skips over ops that
      // act like the identity (such as function calls, returns, yields,
      // controlflow, DTensorLayouts, etc). This means that operand layout
      // that we came from is the layout we want for this value.
      layouts[skipped_value] = layout_to_copy;
}

// Some ops, which are skipped by TraceUseToNextTFOp, will not have layouts
// for their mlir::OpResults.
// E.g. during the creation of the consumers map, we skip the input and output
// of the WhileRegion op. In particular if we have:
//
// %b = tf.WhileRegion(%a) ({
//     %bb0(%arg0):  # Cond
//       %c = tf.A(%arg0)
//       tf.Yield(%c)
//     }, {
//     %bb0(%arg0):  # Body
//       %d = tf.B(%arg0)
//       tf.Yield(%d)
//     }
//   }
// %e = tf.C(%b)
//
// Then the consumers map would directly connect the mlir::Value %a to input 0
// of tf.A and tf.B, bypassing the WhileRegion and the mlir::Value of %arg1.
// Similarly it would connect the mlir::Value of %d directly to input 0 of
// tf.C bypassing the mlir::Value of %b.
// This means that at the end of layout propagation the skipped values would not
// have an assigned layout. But this layout can be derived by taking the known
// layout of %a and propagating to each mlir::Value that was skipped while
// connecting %a to the input 0 of tf.A and tf.B. Similarly we derive the layout
// for %b from %d.
//
// To get layouts we
// 1) Iterate over all ops that have layouts for their OpResults and call
//    TraceUseToNextTFOp to get the skipped mlir::Values.
// 2) If any skipped mlir::Value doesn't have a layout set, then we set the
//    layout.
mlir::LogicalResult CopyLayoutsForSkippedOps(
    mlir::ModuleOp module, mlir::Dialect* tf_dialect,
    llvm::DenseMap<mlir::Value, Layout>& layouts) {
  llvm::DenseMap<llvm::StringRef, mlir::Operation*> func_to_caller;

  if (mlir::failed(GetFuncToCaller(module, func_to_caller)))
    return mlir::failure();

  // Update layouts derived from ops.
  module->walk([&](mlir::Operation* op) {
    for (mlir::OpOperand& operand : op->getOpOperands()) {
      if (layouts.find(operand.get()) == layouts.end()) continue;
      const Layout layout = layouts[operand.get()];
      UpdateLayoutForSkippedOps(operand, func_to_caller, layout, layouts);
    }
  });

  // Update layouts derived from inputs
  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) return mlir::success();

  for (auto& value : main_func.getArguments()) {
    if (layouts.find(value) == layouts.end()) continue;
    const Layout layout = layouts[value];

    for (mlir::OpOperand& operand : value.getUses())
      UpdateLayoutForSkippedOps(operand, func_to_caller, layout, layouts);
  }

  return mlir::success();
}

namespace {
void FilterkAnySpecs(std::vector<std::string>& proposed_specs) {
  for (auto& spec : proposed_specs) {
    if (spec == Layout::kAny) spec = Layout::kUnshardedDim;
  }
}
}  // namespace

// Merges the producer and consumer layouts into a single layout.
// Assumes that all layouts are of the same rank.
// Consumers are first merged together so that we have the layout which is
// sharded in a tensor dim if and only if all consumers are sharded in the same
// sharding_spec.
// If producer layout is present, we merge the consumer layouts into the layout
// of the producer: if the consumer wants a sharded layout in a tensor dimension
// where the producer is unshared *and* the mesh dimension it wants to be
// sharded over is not already sharded over by the producer, then we add that
// sharding to the producer layout.
StatusOr<Layout> MergeLayouts(
    const absl::optional<Layout>& producer,
    const mlir::DenseMap<mlir::OpOperand*, Layout>& consumers) {
  if (consumers.empty()) return producer.value();

  // Initialize the specs to those of the first consumer layout and merge
  // consumers.
  std::vector<std::string> proposed_specs =
      consumers.begin()->second.sharding_spec_strs();
  int layout_rank = proposed_specs.size();

  // Verify consumer layout ranks match.
  for (const auto& consumer : consumers) {
    const Layout& consumer_layout = consumer.second;
    if (consumer_layout.rank() != layout_rank)
      return errors::InvalidArgument(
          "found two consumer layout of different ranks: ",
          consumer_layout.rank(), " and ", layout_rank);
  }

  // Merge consumer layouts.
  for (const auto& consumer : consumers) {
    const Layout& consumer_layout = consumer.second;

    // Check every tensor dimension.
    for (int j = 0; j < consumer_layout.rank(); ++j) {
      const std::string& consumer_spec_j = consumer_layout.sharding_spec(j);
      if (consumer_spec_j == Layout::kAny) continue;

      // If the proposed spec is set as any, give priority to the consumer spec.
      if (proposed_specs[j] == Layout::kAny) {
        proposed_specs[j] = consumer_spec_j;
        continue;
      }

      // If any consumer layout disagrees with the current merge, set the
      // spec to not sharded.
      if (proposed_specs[j] != consumer_spec_j)
        proposed_specs[j] = Layout::kUnshardedDim;
    }
  }

  // Filter over-sharded specs.
  absl::flat_hash_map<std::string, int> counter;
  for (const std::string& spec : proposed_specs) counter[spec] += 1;
  for (std::string& spec : proposed_specs)
    if (counter[spec] > 1) spec = Layout::kUnshardedDim;

  // Return layout if there is no producer, else move into producer algorithm.
  const Mesh mesh = consumers.begin()->second.mesh();
  if (!producer) {
    FilterkAnySpecs(proposed_specs);
    return Layout::GetLayout(proposed_specs, mesh);
  }

  if (producer->rank() != layout_rank) {
    return errors::InvalidArgument(
        "producer and consumer layout have different ranks: ", producer->rank(),
        " and ", layout_rank);
  }

  // For the producer merge, first we define mesh dims used by the producer to
  // avoid creating a layout that shards twice over the same mesh dim.
  llvm::DenseSet<llvm::StringRef> producer_dims;
  for (int j = 0; j < producer->rank(); ++j) {
    llvm::StringRef spec = producer->sharding_spec(j);
    if (Layout::IsShardedDimension(spec.str())) producer_dims.insert(spec);
  }
  // Merge producer layout with existing layout.
  for (int j = 0; j < producer->rank(); ++j) {
    const std::string& producer_spec = producer->sharding_spec(j);

    if (producer_spec == proposed_specs[j] || producer_spec == Layout::kAny)
      continue;

    if (proposed_specs[j] == Layout::kAny) {
      proposed_specs[j] = producer_spec;
      continue;
    }
    // If producer is unsharded and proposed_spec is sharded. Need to make sure
    // mesh dim is not used elsewhere. If so, set to unsharded.
    if (Layout::IsUnshardedDimension(producer_spec)) {
      bool isMeshDimUsed = producer_dims.contains(proposed_specs[j]);
      if (isMeshDimUsed) {
        proposed_specs[j] = Layout::kUnshardedDim;
      }
    } else {
      // If producer is sharded we can set layout to shard over same
      // mesh dim.
      //
      // If mesh dim is already used in the layout elsewhere it will
      // get unset by the case above.
      proposed_specs[j] = producer_spec;
    }
  }
  FilterkAnySpecs(proposed_specs);
  return Layout::GetLayout(proposed_specs, mesh);
}

mlir::LogicalResult InsertLayoutsForDTensorLayout(
    mlir::ModuleOp& module,
    llvm::DenseMap<mlir::Value, absl::optional<Layout>>& producer_request,
    llvm::DenseSet<mlir::Value>& is_updated,
    llvm::DenseSet<mlir::Value>& is_locked) {
  return mlir::failure(
      module
          .walk([&](mlir::TF::DTensorLayout op) -> mlir::WalkResult {
            // Check there are no "Layout::kAny" or "kMatch" specs in the
            // layouts.
            for (const std::string& spec : op.layout().sharding_spec_strs())
              if (spec == Layout::kAny || spec == Layout::kMatch)
                return op->emitOpError()
                       << "found " << spec
                       << " as a sharding spec which is not allowed";
            // Insert layout.
            producer_request[op.input()].emplace(op.layout());
            is_updated.insert(op.input());
            is_locked.insert(op.input());
            return mlir::WalkResult::advance();
          })
          .wasInterrupted());
}

// Runs ComputeLayout API on all ops inside graph **without** any consumer
// requested layout/ operand layouts populated.
mlir::LogicalResult InsertInitialLayoutsFromComputeLayout(
    mlir::ModuleOp module,
    const llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>>& consumers,
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    llvm::DenseMap<mlir::Value, absl::optional<Layout>>& producer_request,
    llvm::DenseMap<mlir::Value, mlir::DenseMap<mlir::OpOperand*, Layout>>&
        consumer_requests,
    llvm::DenseSet<mlir::Value>& is_updated) {
  auto walk_result = module.walk([&](mlir::Operation* op) {
    // We ignore ops that don't have either an OpResult in consumers or an
    // OpOperand in producers. Note that if one operand is missing from
    // producers then all operands should be missing as well as all op results
    // from consumers and the opposite as well.

    if (op->getNumOperands() > 0) {
      if (producers.find(&op->getOpOperand(0)) == producers.end())
        return mlir::WalkResult::advance();
    } else if (op->getNumResults() > 0) {
      if (consumers.find(op->getOpResult(0)) == consumers.end())
        return mlir::WalkResult::advance();
    } else {
      // Note that this case should never happen (I.e. a TF ops should have
      // either inputs or outputs, but that isn't technically guaranteed).
      return mlir::WalkResult::advance();
    }

    auto* expander = SPMDExpanderRegistry::Global()->GetPropagateFnForOp(op);
    if (expander == nullptr) {
      op->emitOpError() << "does not implement layout propagation";
      return mlir::WalkResult::interrupt();
    }

    // Invoke ComputeLayout on `cluster_op` with empty input/consumer layouts.
    StatusOr<llvm::DenseMap<int, Layout>> forward_result =
        expander->ComputeLayoutForward(
            op, /*input_layouts=*/llvm::DenseMap<int, Layout>(),
            /*output_layouts=*/llvm::DenseMap<int, Layout>());
    if (!forward_result.ok()) {
      op->emitOpError() << "ComputeLayoutForward error: "
                        << forward_result.status().error_message();
      return mlir::WalkResult::interrupt();
    }
    StatusOr<llvm::DenseMap<int, Layout>> backward_result =
        expander->ComputeLayoutBackward(
            op, /*input_layouts=*/llvm::DenseMap<int, Layout>(),
            /*output_layouts=*/llvm::DenseMap<int, Layout>());
    if (!backward_result.ok()) {
      op->emitOpError() << "ComputeLayoutBackward error: "
                        << backward_result.status().error_message();
      return mlir::WalkResult::interrupt();
    }

    // If any operand layouts were returned, add the layout to consumer requests
    // and set the value as updated.
    for (auto const& op_idx_and_op_layout : *backward_result) {
      auto const& op_idx = op_idx_and_op_layout.first;
      auto const& op_layout = op_idx_and_op_layout.second;
      auto& operand = op->getOpOperand(op_idx);
      const auto& producer_values = producers.lookup(&operand);
      for (mlir::Value producer_value : producer_values) {
        if (!consumer_requests[producer_value].count(&operand))
          consumer_requests[producer_value][&operand] = op_layout;

        is_updated.insert(producer_value);
      }
    }

    // If any output layouts were returned, add the layout to producer requests
    // and set the value as updated.
    for (auto const& out_idx_and_out_layout : *forward_result) {
      auto const& out_idx = out_idx_and_out_layout.first;
      auto const& out_layout = out_idx_and_out_layout.second;
      mlir::Value output_value = op->getResult(out_idx);
      producer_request.try_emplace(output_value, out_layout);
      is_updated.insert(output_value);
    }

    return mlir::WalkResult::advance();
  });
  return mlir::failure(walk_result.wasInterrupted());
}

// Propagates mesh and inserts initial layouts for
// * Any DTensorLayout ops (this handles function inputs and other ops with user
//   layouts.
// * CopyToMesh
// * ConstOp
mlir::LogicalResult InsertInitialLayouts(
    mlir::ModuleOp& module, mlir::func::FuncOp& main_func,
    const llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>>& consumers,
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    llvm::DenseMap<mlir::Value, mlir::DenseMap<mlir::OpOperand*, Layout>>&
        consumer_request,
    llvm::DenseMap<mlir::Value, absl::optional<Layout>>& producer_request,
    llvm::DenseSet<mlir::Value>& is_updated,
    llvm::DenseSet<mlir::Value>& is_locked) {
  std::queue<mlir::Operation*> operations;

  if (mlir::failed(InsertLayoutsForDTensorLayout(module, producer_request,
                                                 is_updated, is_locked)))
    return mlir::failure();
  return InsertInitialLayoutsFromComputeLayout(module, consumers, producers,
                                               producer_request,
                                               consumer_request, is_updated);
}

// Given a list of mlir::Values with updated producer or consumer layouts
// update the merged_layouts list and track which layouts actually changed.
mlir::LogicalResult MergeAndGetUpdatedLayouts(
    const llvm::DenseSet<mlir::Value>& is_locked,
    llvm::DenseSet<mlir::Value>& is_updated,
    llvm::DenseMap<mlir::Value, absl::optional<Layout>>& producer_request,
    llvm::DenseMap<mlir::Value, mlir::DenseMap<mlir::OpOperand*, Layout>>&
        consumer_requests,
    llvm::DenseMap<mlir::Value, Layout>& merged_layouts) {
  llvm::DenseSet<mlir::Value> updated_merge;
  for (auto& value : is_updated) {
    auto& producer_layout = producer_request[value];
    if (is_locked.find(value) != is_locked.end()) {
      // Locked values must have a producer request. If the merged_layout is
      // not already set, then this is the first pass, so we set it and mark
      // then entry as updated.
      if (merged_layouts.find(value) == merged_layouts.end()) {
        if (!producer_layout)
          return value.getDefiningOp()->emitError() << "missing locked layout";
        merged_layouts[value] = producer_layout.value();
        updated_merge.insert(value);
      }
      continue;
    }
    auto merged = MergeLayouts(producer_layout, consumer_requests[value]);
    if (!merged.ok())
      return value.getDefiningOp()->emitOpError()
             << merged.status().error_message();

    auto current_layout = merged_layouts.find(value);
    if (current_layout == merged_layouts.end() ||
        current_layout->second != merged.value()) {
      updated_merge.insert(value);
      merged_layouts[value] = merged.value();
    }
  }

  is_updated = updated_merge;
  return mlir::success();
}

// Finds the most sharded merged layout given `layouts`.
mlir::LogicalResult GetMostShardedLayout(llvm::ArrayRef<Layout> layouts,
                                         mlir::Location location,
                                         absl::optional<Layout>* out) {
  // If there are no layouts to merge, leave the output empty.
  if (layouts.empty()) return mlir::success();

  absl::optional<Layout> layout;
  std::map<std::string, std::set<int>> layout_map;
  for (const Layout& layout : layouts) {
    for (int i = 0; i < layout.rank(); ++i) {
      const std::string& mesh_dim = layout.dim(i).sharding_spec();
      if (mesh_dim == Layout::kUnshardedDim) continue;

      layout_map[mesh_dim].insert(i);
    }
  }

  for (auto& it : layout_map)
    if (it.second.size() > 1) it.second.clear();

  std::map<int, std::set<std::string>> dim_to_layout_map;
  for (const auto& it : layout_map) {
    assert(it.second.size() <= 1);
    if (it.second.empty()) continue;

    const int tensor_dim_index = *it.second.begin();
    dim_to_layout_map[tensor_dim_index].insert(it.first);
  }

  for (auto& it : dim_to_layout_map)
    if (it.second.size() > 1) it.second.clear();

  std::vector<std::string> merged_spec;
  assert(!layouts.empty());
  for (int i = 0; i < layouts[0].rank(); ++i) {
    const auto it = dim_to_layout_map.find(i);
    if (it != dim_to_layout_map.end() && !it->second.empty()) {
      assert(it->second.size() == 1);
      merged_spec.emplace_back(*it->second.begin());
    } else {
      merged_spec.emplace_back(Layout::kUnshardedDim);
    }
  }
  const auto new_layout = Layout::GetLayout(merged_spec, layouts[0].mesh());
  if (!new_layout.ok()) {
    return mlir::emitError(
        location, llvm::formatv("error in layout propagation while merging "
                                "producer layouts. {0}",
                                new_layout.status().error_message()));
  }
  out->emplace(*new_layout);
  return mlir::success();
}

// Merge layouts of mlir::Value from multiple producers into a single final
// layout. A mlir::Value can have multiple producers if the value is from a
// tf.If/tf.IfRegion op. Given multiple producer layouts of the same
// mlir::Value, the merging logic is as follows:
//   1) If a dimension can be sharded, shard the dimension as much as possible.
//   2) If mesh dimension is already used or two same mesh dimensions are used
//      in different dimensions, then leave the dimension as replicated.
//
// For example:
//  ("x", replicated) , (replicated, "y") will have ("x", "y") merged layout.
//  ("x", replicated) , (replicated, "x") will have (replicated, replicated)
// merged layout.
mlir::LogicalResult MergeProducerLayouts(
    const llvm::DenseMap<mlir::Value, Layout>& merged_layouts,
    const std::vector<mlir::Value>& producer_values, mlir::Location location,
    absl::optional<Layout>* layout_out) {
  // If there is a single producer for mlir::Value, then return the layout
  // from the producer.
  absl::optional<Layout> layout;
  if (producer_values.size() == 1) {
    const auto it = merged_layouts.find(producer_values[0]);
    if (it != merged_layouts.end()) *layout_out = it->second;
    return mlir::success();
  }

  // For the case with multiple producer, merge the layouts.
  llvm::SmallVector<Layout, 4> candidate_layouts;
  candidate_layouts.reserve(producer_values.size());
  for (mlir::Value value : producer_values) {
    auto it = merged_layouts.find(value);
    if (it == merged_layouts.end()) continue;
    candidate_layouts.emplace_back(it->second);
  }

  if (mlir::failed(GetMostShardedLayout(candidate_layouts, location, &layout)))
    return mlir::failure();

  if (layout) *layout_out = *layout;
  return mlir::success();
}

// For an op, calls the corresponding ComputeLayouts function with the data from
// the merged_layouts map. Records the result in the producer_request and
// consumer_requests maps and notes if any layouts have changed.
mlir::LogicalResult UpdateLayoutsForOp(
    mlir::Operation* op,
    const llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    const llvm::DenseMap<mlir::Value, Layout>& merged_layouts,
    llvm::DenseMap<mlir::Value, absl::optional<Layout>>& producer_request,
    llvm::DenseMap<mlir::Value, mlir::DenseMap<mlir::OpOperand*, Layout>>&
        consumer_requests,
    llvm::DenseSet<mlir::Value>& is_updated) {
  auto* expander = SPMDExpanderRegistry::Global()->GetPropagateFnForOp(op);
  if (expander == nullptr)
    return op->emitOpError() << "does not implement layout propagation";

  // Get input and output layouts for this op from the merged_layouts map.
  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());

  for (int i = 0; i < op->getNumOperands(); ++i) {
    // For inputs, we need to find the producer's mlir::Value that eventually
    // feeds into this op. This is in the producers map.
    // Merge different layouts for multiples producers `values`.
    auto producer_values = producers.find(&(op->getOpOperand(i)));
    if (producer_values == producers.end())
      return op->emitError() << "Unable to find producer for operand " << i;

    absl::optional<Layout> layout;
    if (mlir::failed(MergeProducerLayouts(merged_layouts,
                                          producer_values->getSecond(),
                                          op->getLoc(), &layout)))
      return mlir::failure();

    if (layout) input_layouts[i] = *layout;
  }

  for (int i = 0; i < op->getNumResults(); ++i) {
    auto layout = merged_layouts.find(op->getOpResult(i));
    if (layout != merged_layouts.end()) output_layouts[i] = layout->second;
  }

  auto forward_result =
      expander->ComputeLayoutForward(op, input_layouts, output_layouts);
  if (!forward_result.ok()) {
    return op->emitOpError() << "ComputeLayoutForward error: "
                             << forward_result.status().error_message();
  }
  const auto new_output_layouts = *forward_result;
  auto backward_result =
      expander->ComputeLayoutBackward(op, input_layouts, output_layouts);
  if (!backward_result.ok()) {
    return op->emitOpError() << "ComputeLayoutBackward error: "
                             << backward_result.status().error_message();
  }
  const auto new_input_layouts = *backward_result;

  // Update the consumer layouts for this op.
  for (int i = 0; i < op->getNumOperands(); ++i) {
    mlir::OpOperand* operand = &(op->getOpOperand(i));
    // No need to check that this exists, we already did it above.
    const auto& producer_values = producers.find(operand);
    const auto input_layout = new_input_layouts.find(i);

    for (mlir::Value value : producer_values->getSecond()) {
      auto& consumer_request = consumer_requests[value];
      const auto consumer_request_from_op_operand =
          consumer_request.find(operand);

      // Update the consumer_request for this OpOperand: we respect what compute
      // layout returns and erase the a requested layout if no layout is
      // returned.
      // TODO(hongjunchoi, bfontain): Consider the case when op output type is
      // resource type with subtype information.
      if (input_layout != new_input_layouts.end() &&
          (consumer_request_from_op_operand == consumer_request.end() ||
           input_layout->second != consumer_request_from_op_operand->second)) {
        // RestoreV2 op most likely would have unknown rank upon restoring, and
        // we relax unknown rank check for the inputs that are produced from
        // there.
        const bool exempt_restore_unknown_rank =
            ValueRank(value) == -1 && value.getDefiningOp() &&
            llvm::isa<mlir::TF::RestoreV2Op>(value.getDefiningOp());
        if (!exempt_restore_unknown_rank &&
            input_layout->second.rank() != ValueRank(value))
          return op->emitOpError()
                 << "Rank for input " << i << " layout is "
                 << input_layout->second.rank() << " but actual rank is "
                 << ValueRank(value);

        // If there was a layout returned and either no previous request or the
        // request changed, insert and mark as updated.
        consumer_request[operand] = input_layout->second;
        is_updated.insert(value);
      } else if (input_layout == new_input_layouts.end() &&
                 consumer_request_from_op_operand != consumer_request.end()) {
        // If no layout was returned and there is previous request, erase the
        // old consumer request.
        consumer_request.erase(operand);
        is_updated.insert(value);
      }
    }
  }

  // Update the producer request for this op.
  // If the output is different from what is in the request list, update the
  // the request and mark the mlir::Value as having an updated Layout request.
  for (int i = 0; i < op->getNumResults(); ++i) {
    const auto output_layout = new_output_layouts.find(i);
    if (output_layout == new_output_layouts.end()) continue;
    const auto& result = op->getOpResult(i);
    if (producer_request[result] != output_layout->second) {
      if (output_layout->second.rank() != ValueRank(result))
        return op->emitOpError() << "Rank for output " << i << " layout is "
                                 << output_layout->second.rank()
                                 << " but actual rank is " << ValueRank(result);
      producer_request[result] = output_layout->second;
      is_updated.insert(result);
    }
  }
  return mlir::success();
}

mlir::LogicalResult InsertDTensorLayoutOps(
    mlir::OpBuilder& builder,
    const llvm::DenseMap<mlir::Value, Layout>& merged_layouts) {
  for (const auto& merged_layout : merged_layouts) {
    // merged_layout is a pair of mlir::Value and Layout.
    // If there is only one user of the Value and that user is a DTensorLayout
    // op, then we can skip creating the op as the layout is already there. Note
    // that we specifically do not allow updating a layout in an already present
    // DTensorLayout op as we have considered them to be 'locked' throughout
    // the algorithm.
    const auto& users = merged_layout.first.getUsers();
    int num_users = std::distance(users.begin(), users.end());
    if (num_users == 1 && mlir::isa<mlir::TF::DTensorLayout>(*users.begin()))
      continue;
    builder.setInsertionPointAfterValue(merged_layout.first);
    // Handles resource and variant as the real shape is embedded in the
    // resource type elements.
    mlir::Type value_type = GetSubtypeOrSelf(merged_layout.first);

    if (auto type = value_type.dyn_cast<mlir::TensorType>()) {
      auto layout_op = builder.create<mlir::TF::DTensorLayout>(
          merged_layout.first.getLoc(), merged_layout.first,
          mlir::dtensor::LayoutAttr::get(builder.getContext(),
                                         merged_layout.second),
          mlir::TF::ShapeAttr::get(builder.getContext(), type));
      llvm::SmallPtrSet<mlir::Operation*, 4> exception{layout_op};
      merged_layout.first.replaceAllUsesExcept(layout_op.output(), exception);
    } else {
      mlir::emitError(merged_layout.first.getLoc())
          << "value type is not TensorType as expected.";
    }
  }

  return mlir::success();
}

void GetOperationsNeedingUpdate(
    const llvm::DenseSet<mlir::Value>& is_updated,
    const llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>>& consumers,
    llvm::DenseSet<mlir::Operation*>& operations) {
  for (auto& value : is_updated) {
    auto uses = consumers.find(value);
    // Some values have no consumers (e.g. outputs of the main function).
    if (uses != consumers.end())
      for (auto* use : uses->second)
        if (!mlir::isa<mlir::TF::CopyToMeshOp>(use->getOwner()))
          operations.insert(use->getOwner());
    // If this is an OpResult, also add the op that produces it.
    if (value.isa<mlir::OpResult>() &&
        !mlir::isa<mlir::TF::CopyToMeshOp>(value.getDefiningOp()))
      operations.insert(value.getDefiningOp());
  }
}

namespace {

// Custom printing class which prints out layouts and ignores DTensorLayout
// ops and also non registered attributes.
class LayoutPrinter : public mlir::OpAsmPrinter {
 public:
  explicit LayoutPrinter(
      llvm::raw_ostream& os,
      const llvm::DenseMap<mlir::Value, Layout>& merged_layouts)
      : indent_level_(0),
        os_(os),
        current_location_(0),
        next_argument_(0),
        merged_layouts_(merged_layouts) {}

  llvm::raw_ostream& getStream() const override { return os_; }

  void printRegionArgument(mlir::BlockArgument arg,
                           llvm::ArrayRef<mlir::NamedAttribute> argAttrs,
                           bool omitType) override {
    printOperand(arg);
    if (!omitType) {
      os_ << ": ";
      printType(arg.getType());
    }
    printOptionalAttrDict(argAttrs, llvm::None);
  }

  void printOperand(mlir::Value value) override { printOperand(value, os_); }

  /// Print a newline and indent the printer to the start of the current
  /// operation.
  void printNewline() override {
    os_ << "\n";
    os_.indent(indent_level_);
  }

  // Note that we ignore the parameters printEntryBlockArgs and
  // printBlockTerminators for simplicity.
  void printRegion(mlir::Region& blocks, bool printEntryBlockArgs,
                   bool printBlockTerminators,
                   bool printEmptyBlock = false) override {
    os_ << " {\n";
    for (auto& b : blocks.getBlocks()) print(b);
    os_.indent(indent_level_) << "}";
  }

  void print(mlir::Block& block) {
    // Each nested block level increases the indent.
    os_.indent(indent_level_) << "%bb(";
    for (int i = 0; i < block.getNumArguments(); ++i) {
      if (arguments_.find(block.getArgument(i)) == arguments_.end())
        arguments_[block.getArgument(i)] = next_argument_++;
      if (i > 0) os_ << ", ";
      os_ << "%arg" << arguments_[block.getArgument(i)];
    }
    os_ << "):\n";
    indent_level_ += 2;
    for (auto& op : block.getOperations()) print(op);
    indent_level_ -= 2;
  }

  // Prints the TF node name from `loc`.
  void printLoc(mlir::Location loc) {
    os_ << " [" << mlir::GetNameFromLoc(loc) << "]";
  }

  void print(mlir::Operation& op) {
    // Don't print tf.DTensorLayout ops.
    if (mlir::isa<mlir::TF::DTensorLayout>(op)) return;

    // Don't print functions with empty bodies.
    if (auto func_op = mlir::dyn_cast<mlir::func::FuncOp>(op))
      if (func_op.empty()) return;

    // Each operation is on its own line, so we start by indenting the
    // the line.
    os_.indent(indent_level_);

    // Record a unique identifier for the op (this will be used for printing
    // op results and operands).
    location_[&op] = current_location_++;

    // Print the outputs.
    for (int i = 0; i < op.getNumResults(); ++i) {
      if (i > 0) os_ << ", ";
      printOperand(op.getOpResult(i), os_);
    }

    if (op.getNumResults() > 0) os_ << " = ";

    // Some ops have a special printing method, call this if it exists.
    if (auto opInfo = op.getRegisteredInfo()) {
      opInfo->printAssembly(&op, *this, /*defaultDialect=*/"");
      printLoc(op.getLoc());
      os_ << "\n";
      return;
    }

    // Otherwise we do a generic printing.
    printGenericOp(&op, true);
    printLoc(op.getLoc());

    os_ << "\n";
  }

  // Print an operand, this could be both the OpResult or a BlockArgument.
  // We also print the layout if it exists and the type.
  void printOperand(mlir::Value value, llvm::raw_ostream& os) override {
    if (auto result = value.dyn_cast<mlir::OpResult>()) {
      // If DTensorLayout ops are already in the module, we need to skip them
      // since we aren't printing them out.
      if (mlir::isa<mlir::TF::DTensorLayout>(result.getDefiningOp())) {
        printOperand(result.getDefiningOp()->getOperand(0));
        return;
      }

      // OpResult are of the format %op_number:%result_number. We elide the
      // result number if there is only one result (the case for most ops).
      os << "%" << location_[result.getDefiningOp()];
      if (result.getDefiningOp()->getNumResults() > 1)
        os << ":" << result.getResultNumber();
    } else if (auto argument = value.dyn_cast<mlir::BlockArgument>()) {
      if (arguments_.find(argument) == arguments_.end())
        arguments_[argument] = next_argument_++;
      os << "%arg" << arguments_[argument];
    }
    auto layout = merged_layouts_.find(value);
    if (layout != merged_layouts_.end()) {
      os << " \"";
      printLayout(layout->second, os);
      os << "\"";
    }
    os << " ";
    printType(value.getType());
  }

  void printLayout(const Layout& layout, llvm::raw_ostream& os) {
    // Layouts are printed with * for an unsharded dim and the mesh dim for a
    // sharded dim. This keeps the layout compact.
    for (int i = 0; i < layout.rank(); ++i) {
      if (i > 0) os << ",";
      if (Layout::IsUnshardedDimension(layout.sharding_spec(i)))
        os << "*";
      else
        os << layout.sharding_spec(i);
    }
  }

  // A generic op consists of a name, and any of the following:
  // * arguments,
  // * attributes
  // * regions
  // These are printed out in that order.
  void printGenericOp(mlir::Operation* op, bool printOpName) override {
    if (printOpName) os_ << "\"" << op->getName().getStringRef() << "\"";
    os_ << "(";
    for (int i = 0; i < op->getNumOperands(); ++i) {
      if (i > 0) os_ << ", ";
      printOperand(op->getOperand(i), os_);
    }
    os_ << ")";

    if (!op->getAttrs().empty()) {
      std::vector<mlir::NamedAttribute> filtered;
      for (auto attr : op->getAttrs())
        if (*attr.getName().str().begin() != '_' &&
            attr.getName().str() != "device")
          filtered.emplace_back(attr);
      if (!filtered.empty()) {
        os_ << " {";
        for (int i = 0; i < filtered.size(); ++i) {
          if (i > 0) os_ << ", ";
          printNamedAttribute(filtered[i]);
        }
        os_ << "}";
      }
    }

    if (op->getNumRegions() > 0) {
      os_ << " (";
      for (auto& region : op->getRegions()) printRegion(region, false, false);
      os_ << ")";
    }
  };

  void printSymbolName(llvm::StringRef symbolRef) override {
    os_ << symbolRef;
  };

  void printNamedAttribute(mlir::NamedAttribute attr) {
    os_ << attr.getName().strref() << " = ";
    printAttribute(attr.getValue());
  }

  void printAttribute(mlir::Attribute attr) override { attr.print(os_); }

  void printType(mlir::Type type) override { type.print(os_); }

  // The following functions are part of the printing interface but aren't
  // needed for the compact printing form for Layout printing.
  void printAttributeWithoutType(mlir::Attribute attr) override{};
  void printSuccessor(mlir::Block* successor) override{};
  void printSuccessorAndUseList(mlir::Block* successor,
                                mlir::ValueRange succOperands) override{};
  void printOptionalAttrDict(
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<llvm::StringRef> elidedAttrs) override{};
  void printOptionalAttrDictWithKeyword(
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<llvm::StringRef> elidedAttrs) override{};

  void shadowRegionArgs(mlir::Region& region,
                        mlir::ValueRange namesToUse) override{};
  void printAffineMapOfSSAIds(mlir::AffineMapAttr mapAttr,
                              mlir::ValueRange operands) override{};

  void printAffineExprOfSSAIds(mlir::AffineExpr expr,
                               mlir::ValueRange dimOperands,
                               mlir::ValueRange symOperands) override{};

 private:
  int indent_level_;
  llvm::raw_ostream& os_;
  llvm::DenseMap<mlir::Operation*, int> location_;
  int current_location_;
  llvm::DenseMap<mlir::BlockArgument, int> arguments_;
  int next_argument_;
  const llvm::DenseMap<mlir::Value, Layout>& merged_layouts_;
};

// Log the current set of layouts to a file marked by the hash of the input
// module and the stage.
void LogLayoutsAndOps(const int stage, const uint64_t module_hash,
                      const llvm::DenseMap<mlir::Value, Layout>& merged_layouts,
                      mlir::ModuleOp& module) {
  if (module->hasAttr(kDoNotLog) || ((ClientId() != 0) && !LogOnAllTasks()))
    return;

  std::string prefix = tensorflow::GetDumpDirFromEnvVar();
  if (prefix.empty()) return;

  auto* env = tensorflow::Env::Default();
  auto status = env->RecursivelyCreateDir(prefix);
  if (!status.ok()) {
    LOG(WARNING) << "cannot create directory '" + prefix +
                        "': " + status.error_message();
    return;
  }

  absl::StrAppend(&prefix, "/layout_propagation_v2_module_", module_hash,
                  "_stage_", stage, "_");
  if (!tensorflow::Env::Default()->CreateUniqueFileName(&prefix, ".mlir")) {
    LOG(WARNING) << "cannot create unique filename, won't dump MLIR module.";
    return;
  }

  std::unique_ptr<WritableFile> file_writer;
  status = env->NewWritableFile(prefix, &file_writer);
  if (!status.ok()) {
    LOG(WARNING) << "cannot open file '" + prefix +
                        "': " + status.error_message();
    return;
  }

  // Print the module to a string before writing to the file.
  std::string txt_module;
  {
    llvm::raw_string_ostream os(txt_module);
    LayoutPrinter printer(os, merged_layouts);
    module.print(printer);
  }

  status = file_writer->Append(txt_module);
  if (!status.ok()) {
    LOG(WARNING) << "error writing to file '" + prefix +
                        "': " + status.error_message();
    return;
  }
  (void)file_writer->Close();
  LOG(INFO) << "Dumped MLIR module to " << prefix;
}

// Canonicalizer and DCE transformation passes may removed ops in the graph and
// result in multiple consecutive DTensorLayout ops. Detect all such cases and
// replace unnecessary DTensorLayout ops with Identity ops.
mlir::LogicalResult ReplaceAuxiliaryDTensorLayoutOpsWithIdentity(
    mlir::ModuleOp module) {
  llvm::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops;
  module.walk([&](mlir::TF::DTensorLayout op) { layout_ops.emplace_back(op); });

  for (auto layout_op : llvm::reverse(layout_ops)) {
    auto input_op = layout_op.input().getDefiningOp();
    if (auto input_layout_op =
            llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(input_op)) {
      // Check that layout of input DTensorLayout op is equivalent to
      // the layout of its connected DTensorLayout op.
      if (layout_op.layout() != input_layout_op.layout())
        return layout_op.emitOpError(
            "Found inconsistent layout. This should never happen.");

      // Replace DTensorLayout op with identity op.
      mlir::OpBuilder builder(layout_op);
      auto identity = builder.create<mlir::TF::IdentityOp>(
          layout_op->getLoc(), layout_op.getType(), layout_op.input());
      layout_op.output().replaceAllUsesWith(identity.output());
      layout_op.erase();
    }
  }

  return mlir::success();
}

// Inserts/changes DTensorLayout op after IfRegion op and results of then/else
// branches to ensure that the return values of IfRegion ops are consistent.
// After layout propagation, layouts of return value of tf.IfRegion op, and
// layouts of terminators of then/else branches of IfRegion op may be different.
// In that case, the layouts of returns values must be merged to a same layout
// as return values of IfRegion op and results of then/else branches are
// semantically equivalent.
mlir::LogicalResult InsertDTensorLayoutForIfRegionOp(
    const llvm::SmallVectorImpl<mlir::TF::IfRegionOp>& if_ops,
    mlir::MLIRContext* context) {
  for (mlir::TF::IfRegionOp if_op : if_ops) {
    for (mlir::OpResult if_result : if_op.getResults()) {
      const int result_index = if_result.getResultNumber();
      mlir::Value then_branch_result = if_op.then_branch()
                                           .front()
                                           .getTerminator()
                                           ->getOpOperand(result_index)
                                           .get();
      mlir::Value else_branch_result = if_op.else_branch()
                                           .front()
                                           .getTerminator()
                                           ->getOpOperand(result_index)
                                           .get();

      auto if_result_layout =
          llvm::dyn_cast<mlir::TF::DTensorLayout>(*if_result.user_begin());
      auto then_result_layout = llvm::dyn_cast<mlir::TF::DTensorLayout>(
          *then_branch_result.getDefiningOp());
      auto else_result_layout = llvm::dyn_cast<mlir::TF::DTensorLayout>(
          *else_branch_result.getDefiningOp());
      llvm::SmallVector<Layout, 4> layouts{if_result_layout.layout(),
                                           then_result_layout.layout(),
                                           else_result_layout.layout()};
      std::set<Layout> layouts_set{layouts.begin(), layouts.end()};
      if (layouts_set.size() == 1) continue;

      absl::optional<Layout> merged_layout;
      if (mlir::failed(
              GetMostShardedLayout(layouts, if_op.getLoc(), &merged_layout)))
        return mlir::failure();
      assert(merged_layout);

      if_result_layout->setAttr(
          kQualifiedLayoutAttr,
          mlir::dtensor::LayoutAttr::get(context, *merged_layout));
      then_result_layout->setAttr(
          kQualifiedLayoutAttr,
          mlir::dtensor::LayoutAttr::get(context, *merged_layout));
      else_result_layout->setAttr(
          kQualifiedLayoutAttr,
          mlir::dtensor::LayoutAttr::get(context, *merged_layout));
    }
  }
  return mlir::success();
}

// Inserts necessary DTensorRelayout ops so that the layouts for while loops
// are correct.
//
// Due to how while loop layout propagation is done, we may need to fix the
// layouts so that the second and beyond step of the loop receive a tensor with
// the correct layout.
// E.g.
// %b = tf.WhileRegion(%a) ({
//     %bb0(%arg0):  # Cond
//       %c = tf.A(%arg0)
//       tf.Yield(%c)
//     }, {
//     %bb0(%arg0):  # Body
//       %d = tf.B(%arg0)
//       tf.Yield(%d)
//     }
//   }
// %e = tf.C(%b)
//
// Layout propagation treats the loop body as if it were an inlined function and
// does not have a condition which fixes the layout of %d, as return value, to
// match the layout of %arg0 (or %a).
//
// Towards this, we:
// 1) Check the layout of %arg0 and see if matches the layout of the input 0
//    (%d) of tf.Yield.
// 2) If it doesn't match we update the we insert a DTensorRelayout op between
//    %d and tf.Yield with the correct layout and insert a second
//    DTensorRelayout op after the loop body.
//
// NOTE: that it is necessary in general to insert both DTensorRelayout ops,
// as opposed to just updating the layout of %d (which would in general be more
// efficient) since %d may still be used by other ops in the loop body.
//
// NOTE: this is not needed for the condition as the output of the condition is
// a scalar and therefore always replicated.
mlir::LogicalResult InsertRelayoutForWhileLoops(
    const llvm::SmallVectorImpl<mlir::TF::WhileRegionOp>& while_ops,
    mlir::OpBuilder& builder) {
  for (mlir::TF::WhileRegionOp op : while_ops) {
    // Get the terminator so we can check the output layouts of the loop body.
    mlir::Operation* yield_op = op.body().front().getTerminator();
    if (!mlir::isa<mlir::TF::YieldOp>(yield_op))
      return op->emitOpError() << "body terminator is not a Yield op.";

    for (int i = 0; i < op.body().getNumArguments(); ++i) {
      // Inputs should only have one, a DTensorLayout op.
      mlir::Value argument = op.body().getArgument(i);
      if (!argument.hasOneUse())
        return op.emitOpError()
               << "body argument " << i << " doesn't have a single use.";
      mlir::Operation* input_layout_op = argument.getUses().begin().getUser();
      if (!mlir::isa<mlir::TF::DTensorLayout>(input_layout_op))
        return op.emitOpError() << "body argument " << i
                                << " is not consumed by a DTensorLayout op.";
      const Layout input_layout =
          mlir::cast<mlir::TF::DTensorLayout>(input_layout_op).layout();

      // Inputs to Yield should also be a DTensorLayout op.
      if (!yield_op->getOperand(i).isa<mlir::OpResult>() ||
          !mlir::isa<mlir::TF::DTensorLayout>(
              yield_op->getOperand(i).getDefiningOp()))
        return yield_op->emitOpError()
               << "argument " << i << " to is not a DTensorLayout op.";
      mlir::Operation* output_layout_op =
          yield_op->getOperand(i).getDefiningOp();
      const Layout output_layout =
          mlir::cast<mlir::TF::DTensorLayout>(output_layout_op).layout();

      // If the layouts are equal we have nothing to do. Note that this caches
      // the case that that input and output are a resource, since the layout
      // of a resource is fixed.
      if (input_layout == output_layout) continue;

      // Insert the first Relayout op (in the loop body).
      builder.setInsertionPointAfter(output_layout_op);
      if (!yield_op->getOperand(i).getType().isa<mlir::TensorType>())
        return yield_op->emitOpError()
               << "operand " << i << " does not have TensorType";
      mlir::TF::ShapeAttr global_shape = mlir::TF::ShapeAttr::get(
          builder.getContext(),
          yield_op->getOperand(i).getType().cast<mlir::TensorType>());
      mlir::TF::RelayoutOp first_relayout =
          builder.create<mlir::TF::RelayoutOp>(
              op.getLoc(), yield_op->getOperand(i).getType(),
              yield_op->getOperand(i), input_layout.ToString());
      mlir::TF::DTensorLayout first_layout_op =
          builder.create<mlir::TF::DTensorLayout>(
              op.getLoc(), first_relayout.output(),
              mlir::dtensor::LayoutAttr::get(builder.getContext(),
                                             input_layout),
              global_shape);
      yield_op->setOperand(i, first_layout_op.output());

      // Insert the second relayout op after the loop itself.
      builder.setInsertionPointAfter(op);
      mlir::TF::DTensorLayout second_layout_op =
          builder.create<mlir::TF::DTensorLayout>(
              op.getLoc(), op->getResult(i),
              mlir::dtensor::LayoutAttr::get(builder.getContext(),
                                             input_layout),
              global_shape);
      mlir::TF::RelayoutOp second_relayout =
          builder.create<mlir::TF::RelayoutOp>(
              op.getLoc(), second_layout_op.output().getType(),
              second_layout_op.output(), output_layout.ToString());
      op->getResult(i).replaceAllUsesExcept(
          second_relayout.output(), llvm::SmallPtrSet<mlir::Operation*, 1>{
                                        second_layout_op.getOperation()});
    }
  }
  return mlir::success();
}

// For all constants with multiple usages, clone the constants so that each
// constant operation has at most 1 usage.
void DuplicateConstants(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::TF::ConstOp, 4> const_ops;
  module.walk(
      [&](mlir::TF::ConstOp const_op) { const_ops.emplace_back(const_op); });

  for (mlir::TF::ConstOp const_op : const_ops) {
    mlir::OpBuilder builder(const_op);
    auto uses = const_op->getUses();
    if (uses.empty()) return;

    llvm::SmallDenseMap<mlir::Operation*, mlir::OpOperand*> const_use_map;
    mlir::OpOperand& first_use = *uses.begin();
    for (mlir::OpOperand& use : uses) {
      if (&use == &first_use) continue;

      mlir::Operation* new_const = builder.clone(*const_op);
      const_use_map.try_emplace(new_const, &use);
    }

    for (const auto& it : const_use_map) it.second->set(it.first->getResult(0));
  }
}

// Find the root(s) values of "current_value" within the cycle, and put it
// into "roots".
void FindRoot(
    const llvm::DenseSet<mlir::Value>& is_updated,
    const mlir::Value& current_value,
    llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    llvm::DenseSet<mlir::Value>* roots) {
  // Standard BFS to find root values of current_value.
  std::deque<mlir::Value> to_process;
  to_process.push_back(current_value);

  llvm::DenseSet<mlir::Value> visited;
  visited.insert(current_value);

  while (!to_process.empty()) {
    int level_size = to_process.size();
    for (int UNUSED = 0; UNUSED < level_size; ++UNUSED) {
      mlir::Value cur_val = to_process.front();
      to_process.pop_front();

      // Terminating condition, if there is no defining op then this is a root.
      mlir::Operation* defining_op = cur_val.getDefiningOp();
      if (defining_op == nullptr) {
        roots->insert(current_value);
        continue;
      }

      // Expand out from 'cur_val' one step closer to roots. If there was
      // no-one one step closer to root, then this is a root.
      bool is_root = true;
      for (int i = 0; i < defining_op->getNumOperands(); ++i) {
        mlir::Value operand = defining_op->getOperand(i);
        if (operand != cur_val && is_updated.contains(operand)) {
          is_root = false;

          if (!visited.contains(operand)) {
            visited.insert(operand);
            to_process.push_back(operand);
          }
        }
      }

      if (is_root) roots->insert(cur_val);
    }
  }
}

// Finds the root value(s) of the values that have layouts cycling back and
// forth in an infinite loop during layout propagation and prints the closest TF
// op that consumes those root value(s). This allows users and developers to
// debug the root cause of layouts that should be changed to prevent infinite
// layout propagation.
void FindRootsAndEmitError(
    mlir::ModuleOp& module,
    llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>> producers,
    const llvm::DenseSet<mlir::Value>& is_updated) {
  llvm::DenseSet<mlir::Value> roots;
  for (auto& value : is_updated) {
    FindRoot(is_updated, value, producers, &roots);
  }
  module.emitOpError()
      << "Maximum number of layout propagation steps reached. Unable to "
         "converge to a fixed layout. Please rerun with a higher limit in the "
         "DTENSOR_LAYOUT_PROPAGATION_MAX_STEPS environment variable.";
  for (auto& root : roots) {
    for (mlir::OpOperand& operand : root.getUses()) {
      llvm::DenseMap<llvm::StringRef, mlir::Operation*> func_to_caller;
      llvm::SmallVector<mlir::Value, 4> skipped_values;

      // For each root value that may need a different layout, find the
      // closest TF op that consumes it and print it.
      llvm::SmallVector<mlir::OpOperand*, 4> consuming_operands =
          TraceUseToNextTFOp(&operand, func_to_caller, &skipped_values);

      for (mlir::OpOperand* new_operand : consuming_operands) {
        mlir::Operation* operation = new_operand->getOwner();
        mlir::Location loc = operation->getLoc();
        operation->emitOpError() << '\n'
                                 << "The following op consumes tensors that "
                                    "may need a different layout. "
                                    "["
                                 << mlir::GetNameFromLoc(loc) << "]" << '\n';
      }
    }
  }
}
}  // namespace

// Runs an iteration of layout propagation, where we merge producer and consumer
// requests and then recompute recommended layouts on all operations that
// are connected to an updated layout.
Status RunOneIteration(
    llvm::DenseSet<mlir::Value>& is_locked,
    llvm::DenseSet<mlir::Value>& is_updated,
    llvm::DenseMap<mlir::Value, absl::optional<Layout>>& producer_request,
    llvm::DenseMap<mlir::Value, mlir::DenseMap<mlir::OpOperand*, Layout>>&
        consumer_requests,
    llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>>& producers,
    llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>>& consumers,
    llvm::DenseMap<mlir::Value, Layout>& merged_layouts, mlir::ModuleOp& module,
    const uint64_t module_hash, int* stage) {
  if (is_updated.empty()) return Status::OK();
  // Merge any possibly updated layouts.
  if (mlir::failed(
          MergeAndGetUpdatedLayouts(is_locked, is_updated, producer_request,
                                    consumer_requests, merged_layouts)))
    return errors::Internal(
        "MergeAndGetUpdatedLayouts failed to merge layouts.");

  // Compile a list of operations with updated inputs or outputs.
  llvm::DenseSet<mlir::Operation*> operations_needing_update;
  GetOperationsNeedingUpdate(is_updated, consumers, operations_needing_update);
  is_updated.clear();

  if (VLOG_IS_ON(2)) {
    LogLayoutsAndOps(*stage, module_hash, merged_layouts, module);
  }

  for (auto* op : operations_needing_update) {
    if (mlir::failed(UpdateLayoutsForOp(op, producers, merged_layouts,
                                        producer_request, consumer_requests,
                                        is_updated)))
      return errors::Internal("UpdateLayoutsForOp failed to update layouts.");
  }
  ++(*stage);
  return Status::OK();
}

// Compares every value's layouts in `merged_a` with the ones in `merged_b`,
// and store the values that differ in `changed`.
Status CompareMergedLayouts(const llvm::DenseMap<mlir::Value, Layout>& merged_a,
                            const llvm::DenseMap<mlir::Value, Layout>& merged_b,
                            llvm::DenseSet<mlir::Value>& changed) {
  if (merged_a.size() != merged_b.size())
    return errors::Internal(
        "Both merged_layouts did not have the same number of set layouts.");
  for (const auto& value_and_layout : merged_a) {
    const mlir::Value value = value_and_layout.getFirst();
    const Layout& layout = value_and_layout.getSecond();
    auto value_and_layout_in_b = merged_b.find(value);
    if (value_and_layout_in_b == merged_b.end())
      return errors::Internal(
          "Comparing merged_layouts that contain different mlir::Value's.");
    if (value_and_layout_in_b->second != layout) {
      changed.insert(value);
    }
  }
  return Status::OK();
}

// MLIR pass that propagates layout for all ops the module.
struct DLayoutPropagationPassV2
    : public DTensorLayoutPropagationV2Base<DLayoutPropagationPassV2> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto module = getOperation();

    if (mlir::failed(ReplaceAuxiliaryDTensorLayoutOpsWithIdentity(module)))
      return signalPassFailure();

    // In order to ensure that constant operations with multiple usages with
    // different consumer layout requests does not lead to replicated constant
    // tensors, we duplicate all constants to have at most 1 usages.
    // After SPMD Expansion, these duplicated constants will be merged back
    // during SCCP pass.
    DuplicateConstants(module);

    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) return;

    mlir::Dialect* tf_dialect =
        context.getLoadedDialect<mlir::TF::TensorFlowDialect>();

    // This maps from OpResults to a list of OpOperands that consume this.
    // Note that this will pass over/through
    // (Stateful)PartitionedCall and other control flow, directly connecting
    // producing ops to their consumers in the function. I.e. it presents
    // flattened/inlined view of the flow of data.
    llvm::DenseMap<mlir::Value, std::vector<mlir::OpOperand*>> consumers;
    // Maintain a reverse mapping.
    llvm::DenseMap<mlir::OpOperand*, std::vector<mlir::Value>> producers;
    // For each mlir::Value this is what the producer would like to have the
    // layout be.
    llvm::DenseMap<mlir::Value, absl::optional<Layout>> producer_request;
    // For each mlir::Value this is what the consumers would like to have the
    // layout be. Note the map is in 'parallel' to the consumers map above.
    llvm::DenseMap<mlir::Value, mlir::DenseMap<mlir::OpOperand*, Layout>>
        consumer_requests;
    // Tracks if the layout was updated since last cycle.
    llvm::DenseSet<mlir::Value> is_updated;
    // Tracks if the layout is locked. In this case we don't pass consumer
    // layouts to MergeLayouts. Used for input layouts and user defined layouts.
    llvm::DenseSet<mlir::Value> is_locked;

    // Create consumers and producers maps.
    if (mlir::failed(
            PopulateConsumersFromModule(&module, tf_dialect, consumers)))
      return signalPassFailure();

    for (auto& consumer : consumers) {
      for (auto* operand : consumer.second) {
        if (producers.find(operand) == producers.end()) {
          producers[operand] = std::vector<mlir::Value>{consumer.first};
        } else {
          producers[operand].emplace_back(consumer.first);
        }
      }
    }

    // Setup the initial starting conditions for the layout algorithm
    if (mlir::failed(InsertInitialLayouts(
            module, main_func, consumers, producers, consumer_requests,
            producer_request, is_updated, is_locked)))
      return signalPassFailure();

    const auto module_hash = OpHash(module);
    int stage = 0;

    llvm::DenseMap<mlir::Value, Layout> merged_layouts;
    Status status;

    while (!is_updated.empty() && stage < kLayoutPropagationMaxStages) {
      ++stage;
      int steps = 0;
      // Step 1. Run the layout propagation v2 until convergence or max steps.
      while (!is_updated.empty() && steps < LayoutPropagationMaxSteps()) {
        Status status = RunOneIteration(
            is_locked, is_updated, producer_request, consumer_requests,
            producers, consumers, merged_layouts, module, module_hash, &steps);
        if (!status.ok()) {
          module.emitOpError() << "Failure running iteration.";
          return signalPassFailure();
        }
      }
      if (VLOG_IS_ON(2)) {
        LOG(INFO) << "Failed to converge in stage " << stage;
      }
      // Step 2. If we are here, then we failed to converge, and likely
      // there is an oscillation of layouts. Detect all the edges that are
      // changing layouts.
      llvm::DenseMap<mlir::Value, Layout> merged_layouts_at_max_steps =
          merged_layouts;
      llvm::DenseSet<mlir::Value> changed;
      int previous_change_size = -1;

      while (changed.size() > previous_change_size) {
        if (!RunOneIteration(is_locked, is_updated, producer_request,
                             consumer_requests, producers, consumers,
                             merged_layouts, module, module_hash, &steps)
                 .ok()) {
          module.emitOpError() << "Failure running iteration.";
          return signalPassFailure();
        }
        if (!CompareMergedLayouts(merged_layouts_at_max_steps, merged_layouts,
                                  changed)
                 .ok()) {
          module.emitOpError() << "Failure comparing merged layouts.";
          return signalPassFailure();
        }
        previous_change_size = changed.size();
      }

      // Step 3. Layouts that haven't changed means they're not part of the
      // cyclic problem, so freeze them.
      for (const auto& value_and_layout : merged_layouts) {
        const mlir::Value value = value_and_layout.getFirst();
        if (changed.find(value) == changed.end()) {
          is_locked.insert(value);
        }
      }
      // Step 4. Any information corresponding to the changed layouts
      // should be disinfected, we do this by clearing all information
      // regarding them.
      for (const mlir::Value changed_value : changed) {
        producer_request.erase(changed_value);
        consumer_requests.erase(changed_value);
        merged_layouts.erase(changed_value);
      }

      // Step 5. ComputeLayout again on all the ops linked to the changed
      // layouts. The next iteration will take this information and merge again.
      llvm::DenseSet<mlir::Operation*> operations_needing_update;
      is_updated = changed;
      GetOperationsNeedingUpdate(is_updated, consumers,
                                 operations_needing_update);
      is_updated.clear();

      for (auto* op : operations_needing_update) {
        if (mlir::failed(UpdateLayoutsForOp(op, producers, merged_layouts,
                                            producer_request, consumer_requests,
                                            is_updated))) {
          module.emitOpError() << "Failure in UpdateLayoutsForOp.";
          return signalPassFailure();
        }
      }
    }

    if (stage >= kLayoutPropagationMaxStages) {
      FindRootsAndEmitError(module, producers, is_updated);
      return signalPassFailure();
    }

    if (mlir::failed(
            CopyLayoutsForSkippedOps(module, tf_dialect, merged_layouts)))
      return signalPassFailure();

    if (VLOG_IS_ON(2)) {
      LogLayoutsAndOps(stage, module_hash, merged_layouts, module);
    }

    if (!AllOpResultsHaveLayouts(&module, tf_dialect, merged_layouts))
      return signalPassFailure();

    if (mlir::failed(InsertDTensorLayoutOps(builder, merged_layouts)))
      return signalPassFailure();

    // Handle layout of control flow operations.
    llvm::SmallVector<mlir::TF::IfRegionOp, 4> if_ops;
    llvm::SmallVector<mlir::TF::WhileRegionOp, 4> while_ops;
    module.walk([&](mlir::Operation* op) {
      if (auto if_op = llvm::dyn_cast<mlir::TF::IfRegionOp>(op))
        if_ops.emplace_back(if_op);
      else if (auto while_op = llvm::dyn_cast<mlir::TF::WhileRegionOp>(op))
        while_ops.emplace_back(while_op);
    });

    if (mlir::failed(InsertRelayoutForWhileLoops(while_ops, builder)))
      return signalPassFailure();

    if (mlir::failed(
            InsertDTensorLayoutForIfRegionOp(if_ops, builder.getContext())))
      return signalPassFailure();
  };
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorLayoutPropagationPassV2() {
  return std::make_unique<DLayoutPropagationPassV2>();
}

}  // namespace dtensor
}  // namespace tensorflow

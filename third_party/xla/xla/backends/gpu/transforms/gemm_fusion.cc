/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/gemm_fusion.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/status/status_macros.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/codegen/triton/support_legacy.h"
#include "xla/backends/gpu/transforms/bitcast_utils.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

using triton_fusion::CombineDotRequirements;
using triton_fusion::DimensionOrder;
using triton_fusion::DimOrderMap;
using triton_fusion::DimOrdersAndReqs;
using triton_fusion::DimOrdersAndReqsOrError;
using triton_fusion::DotProperties;
using triton_fusion::DotRequirements;
using triton_fusion::DotRequirementsOrError;
using triton_fusion::FusionContext;
using triton_fusion::GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible;
using triton_fusion::TransformDirection;

// This represents a directed graph.
class AdjacencyList {
 public:
  using NodeId = int64_t;

  NodeId AddNode() {
    adj_.emplace_back();
    return adj_.size() - 1;
  }

  const std::vector<NodeId>& GetOutNeighbors(NodeId node_id) const {
    return adj_.at(node_id);
  }

  void ReserveSpaceForOutNeighbors(NodeId node_id, size_t count) {
    adj_.at(node_id).reserve(count);
  }

  void AddArc(NodeId from, NodeId to) { adj_.at(from).push_back(to); }

  // Currently the Root node is the node which was added first.
  NodeId GetRoot() const {
    CHECK(!adj_.empty());
    return 0;
  }

 private:
  // Adjacency list: A vector of out-neighbors for each node.
  std::vector<std::vector<NodeId>> adj_;
};

struct HloAndIterSpec {
  HloInstruction* original_hlo;
  TensorIterationSpec iter_spec;

  auto ToTuple() const { return std::make_tuple(original_hlo, iter_spec); }
  bool operator==(const HloAndIterSpec& other) const {
    return ToTuple() == other.ToTuple();
  }
  template <typename H>
  friend H AbslHashValue(H h, const HloAndIterSpec& key) {
    return H::combine(std::move(h), key.ToTuple());
  }
};

struct NodeFusionPlan {
  HloInstruction* original_hlo = nullptr;
  bool should_fuse = false;
};

struct FusionPlan {
  // The graph describing the structure of the fusion that we build - nodes
  // corresponding to the instructions and arcs pointing from users to operands.
  AdjacencyList graph;
  // The fusion plan for each node.
  absl::flat_hash_map<AdjacencyList::NodeId, NodeFusionPlan> map;
};

struct FusionPlanAndRequirements {
  FusionPlan fusion_plan;
  DotRequirements requirements;
};

struct HlosAndRequirements {
  // The original HLO (which is outside the fusion computation).
  HloInstruction* original_hlo = nullptr;
  // The fused HLO inside the new fusion computation, built by the builder.
  //
  // This can have the same opcode as `original_hlo` or it can be a parameter if
  // the original HLO can't be fused.
  HloInstruction* fused_hlo = nullptr;
  // The requirements imposed by the fused operations.
  //
  // If we fuse further operations they may have to conform to these
  // requirements.
  DotRequirements requirements;
};

// Clones the hero kDot operation into the fusion.
HloInstruction& FuseDot(HloInstruction& dot,
                        const std::vector<HlosAndRequirements>& hlos_and_reqs,
                        HloComputation::Builder& builder  // append
) {
  VLOG(3) << "Fusing " << dot.ToString();

  std::vector<HloInstruction*> hlo_new_operands;
  hlo_new_operands.reserve(dot.operand_count());
  for (int i = 0; i < hlos_and_reqs.size(); ++i) {
    hlo_new_operands.push_back(hlos_and_reqs[i].fused_hlo);
  }
  return *builder.AddInstruction(
      dot.CloneWithNewOperands(dot.shape(), hlo_new_operands));
}

// Tells how many new parameters does a fusion gain by fusing the operation as
// an input.
int64_t NumAddedParameters(const HloInstruction& hlo) {
  // Non-scalar constant is equivalent to a parameter: one input, one output.
  if (hlo.opcode() == HloOpcode::kParameter ||
      (hlo.opcode() == HloOpcode::kConstant &&
       !ShapeUtil::IsScalar(hlo.shape()))) {
    return 0;
  }
  // All other instructions add all own inputs and remove own single output.
  return hlo.operand_count() - 1;
}

// Just a helper to reduce "unwrapping" code where we use this.
std::optional<DimOrdersAndReqs> GetOperandDimOrdersAndCombinedReqs(
    const HloInstruction& hlo, const DimensionOrder& dim_order,
    const DotProperties& properties,
    const se::GpuComputeCapability& gpu_version,
    const DotRequirements& requirements) {
  DimOrdersAndReqsOrError dim_orders_and_new_reqs =
      GetPropagatedDimOrdersAndRequirements(
          hlo, dim_order, TransformDirection::kOutputToInput, properties);
  if (std::holds_alternative<FusionDecision>(dim_orders_and_new_reqs)) {
    VLOG(5) << "Not fusing " << hlo.ToString()
            << " to the output due to the decision: "
            << std::get<FusionDecision>(dim_orders_and_new_reqs).Explain();
    return std::nullopt;
  }
  DotRequirementsOrError combined_reqs = CombineDotRequirements(
      requirements,
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).requirements);
  if (std::holds_alternative<FusionDecision>(combined_reqs)) {
    VLOG(5) << "Not fusing " << hlo.ToString()
            << " to the output due to the decision: "
            << std::get<FusionDecision>(combined_reqs).Explain();
    return std::nullopt;
  }
  return DimOrdersAndReqs{
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).dim_orders,
      std::get<DotRequirements>(combined_reqs)};
}

// Just a helper to reduce "unwrapping" code where we use this.
std::optional<DimOrdersAndReqs> GetOperandDimOrdersAndCombinedReqsIfProfitable(
    const HloInstruction& hlo, const DimensionOrder& dim_order,
    const DotProperties& properties,
    const se::GpuComputeCapability& gpu_version,
    const DotRequirements& requirements) {
  DimOrdersAndReqsOrError dim_orders_and_new_reqs =
      GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
          hlo, TransformDirection::kOutputToInput,
          /*src_operand_index=*/std::nullopt, dim_order, gpu_version,
          properties);
  if (std::holds_alternative<FusionDecision>(dim_orders_and_new_reqs)) {
    VLOG(5) << "Not fusing " << hlo.ToString()
            << " to the output due to the decision: "
            << std::get<FusionDecision>(dim_orders_and_new_reqs).Explain();
    return std::nullopt;
  }
  DotRequirementsOrError combined_reqs = CombineDotRequirements(
      requirements,
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).requirements);
  if (std::holds_alternative<FusionDecision>(combined_reqs)) {
    VLOG(5) << "Not fusing " << hlo.ToString()
            << " to the output due to the decision: "
            << std::get<FusionDecision>(combined_reqs).Explain();
    return std::nullopt;
  }
  return DimOrdersAndReqs{
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).dim_orders,
      std::get<DotRequirements>(combined_reqs)};
}

// Just a helper to reduce "unwrapping" code where we use this.
std::optional<DimOrdersAndReqs> GetUserDimOrdersAndCombinedReqsIfProfitable(
    const HloInstruction& hlo, const DimensionOrder& hlo_dim_order,
    const HloInstruction& user, const DotProperties& properties,
    const se::GpuComputeCapability& gpu_version,
    const DotRequirements& requirements) {
  DimOrdersAndReqsOrError dim_orders_and_new_reqs =
      GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
          user, TransformDirection::kInputToOutput, user.operand_index(&hlo),
          hlo_dim_order, gpu_version, properties);
  if (std::holds_alternative<FusionDecision>(dim_orders_and_new_reqs)) {
    VLOG(5) << "Not fusing " << user.ToString()
            << " to the input due to the decision: "
            << std::get<FusionDecision>(dim_orders_and_new_reqs).Explain();
    return std::nullopt;
  }
  DotRequirementsOrError combined_reqs = CombineDotRequirements(
      requirements,
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).requirements);
  if (std::holds_alternative<FusionDecision>(combined_reqs)) {
    VLOG(5) << "Not fusing " << user.ToString()
            << " to the input due to the decision: "
            << std::get<FusionDecision>(combined_reqs).Explain();
    return std::nullopt;
  }
  return DimOrdersAndReqs{
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).dim_orders,
      std::get<DotRequirements>(combined_reqs)};
}

class FusionPlanBuilder {
 public:
  // Builds and returns the FusionPlan. Clears internal state.
  FusionPlan BuildPlan() {
    FusionPlan fusion_plan;
    for (auto& [node_id, node] : node_map_) {
      CHECK(node.should_fuse.has_value());
      fusion_plan.map[node_id] =
          NodeFusionPlan{node.original_hlo, *node.should_fuse};
    }

    node_map_.clear();
    node_reuse_map_.clear();
    fusion_plan.graph = std::move(graph_);
    return fusion_plan;
  }

  void ReserveSpaceForOutNeighbors(AdjacencyList::NodeId node_id,
                                   size_t count) {
    graph_.ReserveSpaceForOutNeighbors(node_id, count);
  }

  void AddArc(AdjacencyList::NodeId from, AdjacencyList::NodeId to) {
    graph_.AddArc(from, to);
  }

  HloInstruction* GetOriginalHlo(AdjacencyList::NodeId node_id) const {
    return node_map_.at(node_id).original_hlo;
  }

  const DimensionOrder& GetDimOrder(AdjacencyList::NodeId node_id) const {
    return node_map_.at(node_id).dim_order;
  }

  // Inserts a node for the given HLO and `dim_order` unless already exists.
  // Returns the node id and a bool indicating if a new node was inserted.
  std::pair<AdjacencyList::NodeId, bool> InsertNode(
      HloInstruction& hlo, const DimensionOrder& dim_order) {
    HloAndIterSpec reuse_key{&hlo, dim_order.ToTensorIterationSpec()};

    // Attempt to insert a placeholder. If the key already exists, inserted is
    // false.
    auto [it, inserted] = node_reuse_map_.insert({reuse_key, -1});
    if (!inserted) {
      return {it->second, false};
    }

    // Key was not present. Create the node and update the map.
    AdjacencyList::NodeId node_id = graph_.AddNode();
    it->second = node_id;
    CHECK(node_map_
              .insert({node_id,
                       Node{&hlo, dim_order, /*should_fuse=*/std::nullopt}})
              .second);
    return {node_id, true};
  }

  // Assigns fusion decision for the specified node.
  // The node must not have an already assigned decision.
  void SetShouldFuseNode(AdjacencyList::NodeId node_id, bool should_fuse) {
    Node& node = node_map_.at(node_id);
    CHECK(!node.should_fuse.has_value());
    node.should_fuse = should_fuse;
  }

 private:
  AdjacencyList graph_;

  struct Node {
    HloInstruction* original_hlo;
    DimensionOrder dim_order;
    std::optional<bool> should_fuse;
  };
  absl::flat_hash_map<AdjacencyList::NodeId, Node> node_map_;

  // Allows reusing nodes when multiple instructions iterate over the same HLO
  // using the same iteration spec. In that case we don't duplicate the
  // instruction in the fusion.
  absl::flat_hash_map<HloAndIterSpec, AdjacencyList::NodeId> node_reuse_map_;
};

// Builds the fusion map and the requirements which can later be used to
// actually fuse that subgraph.
FusionPlanAndRequirements BuildFusionPlanTowardOperands(
    HloInstruction& root_hlo, const DimensionOrder& root_dim_order,
    const std::optional<int>& max_params,
    const se::GpuComputeCapability& gpu_version,
    const DotProperties& properties,
    const DotRequirements& requirements_so_far) {
  CHECK(!max_params.has_value() || max_params.value() >= 1);

  FusionPlanBuilder fusion_builder;

  // The requirements imposed by the fusion choices made in this function,
  // combined with the existing requirements. This is one of the outputs of
  // this function.
  DotRequirements combined_reqs = requirements_so_far;

  AdjacencyList::NodeId root =
      fusion_builder.InsertNode(root_hlo, root_dim_order).first;

  // Nodes at the fusion edge that can either get fused too or become parameters
  // of the fusion. Used to track the number of parameters.
  absl::flat_hash_set<AdjacencyList::NodeId> inputs({root});

  std::queue<AdjacencyList::NodeId> queue({root});
  int64_t num_requeued = 0;

  // BFS
  // If all queued instructions are re-queued, they all exceed the parameter
  // limit, so stop fusing.
  while (queue.size() > num_requeued) {
    AdjacencyList::NodeId node_id = queue.front();
    queue.pop();
    HloInstruction& original_hlo = *fusion_builder.GetOriginalHlo(node_id);
    const DimensionOrder& dim_order = fusion_builder.GetDimOrder(node_id);

    // Watch the total number of fusion parameters.
    if (max_params.has_value() &&
        inputs.size() + NumAddedParameters(original_hlo) > max_params.value()) {
      // Re-queue: the number of parameters may go down when other instructions
      // are processed.
      queue.push(node_id);
      // Prevent infinite loops.
      ++num_requeued;
      continue;
    }
    num_requeued = 0;

    if (original_hlo.opcode() == HloOpcode::kParameter) {
      fusion_builder.SetShouldFuseNode(node_id, false);
      continue;
    }

    // TODO(b/393299275): this check cannot be replaced by a
    // `IsTritonSupportedComputation` because we will do some rewrites
    // later that might change the decision. For example 'scaled-dot-rewriter'
    // replaces unsupported F8E8M0FNU with u8. We should have a more principled
    // way check if we will be able to emit the triton code for the fusion.
    if (original_hlo.opcode() == HloOpcode::kDynamicSlice) {
      // TODO(b/417172838): support dynamic slice op.
      fusion_builder.SetShouldFuseNode(node_id, false);
      VLOG(5) << "Not fusing dynamic slice: " << original_hlo.ToString();
      continue;
    }

    auto opt_result = GetOperandDimOrdersAndCombinedReqsIfProfitable(
        original_hlo, dim_order, properties, gpu_version, combined_reqs);
    if (!opt_result.has_value()) {
      fusion_builder.SetShouldFuseNode(node_id, false);
      continue;
    }

    const DimOrderMap operand_dim_orders = std::move(opt_result->dim_orders);
    combined_reqs = std::move(opt_result->requirements);

    inputs.erase(node_id);
    fusion_builder.ReserveSpaceForOutNeighbors(node_id,
                                               original_hlo.operand_count());
    for (HloInstruction* operand : original_hlo.operands()) {
      const DimensionOrder& operand_dim_order = operand_dim_orders.at(operand);
      auto [operand_node_id, is_new_node] =
          fusion_builder.InsertNode(*operand, operand_dim_order);
      fusion_builder.AddArc(node_id, operand_node_id);
      if (is_new_node) {
        VLOG(6) << "Enqueueing " << operand->ToString() << ":"
                << operand_dim_order.ToString();
        inputs.insert(operand_node_id);
        queue.push(operand_node_id);
      }
    }
    fusion_builder.SetShouldFuseNode(node_id, true);
  }
  // Handle the remaining requeued items.
  for (; !queue.empty(); queue.pop()) {
    AdjacencyList::NodeId node_id = queue.front();
    fusion_builder.SetShouldFuseNode(node_id, false);
  }
  return {fusion_builder.BuildPlan(), std::move(combined_reqs)};
}

// Builds the HLO instructions for the fusion represented by `fusion_plan`,
// starting from `node_id`.
HloInstruction& BuildFusionTowardOperandsImpl(
    AdjacencyList::NodeId node_id, const FusionPlan& fusion_plan,
    absl::flat_hash_map<AdjacencyList::NodeId, HloInstruction*>&
        fused_hlo_map,                           // read/append
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  if (auto it = fused_hlo_map.find(node_id); it != fused_hlo_map.end()) {
    return *it->second;
  }

  const NodeFusionPlan& node_fusion_plan = fusion_plan.map.at(node_id);
  const bool should_fuse = node_fusion_plan.should_fuse;
  HloInstruction& original_hlo = *node_fusion_plan.original_hlo;

  HloInstruction* fused_hlo = nullptr;
  if (should_fuse) {
    HloInstruction::InstructionVector new_operands;
    for (AdjacencyList::NodeId operand_id :
         fusion_plan.graph.GetOutNeighbors(node_id)) {
      new_operands.push_back(&BuildFusionTowardOperandsImpl(
          operand_id, fusion_plan, fused_hlo_map, builder, fusion_params));
    }
    fused_hlo = builder.AddInstruction(
        original_hlo.CloneWithNewOperands(original_hlo.shape(), new_operands));
  } else {
    fusion_params.push_back(&original_hlo);
    fused_hlo = builder.AddInstruction(HloInstruction::CreateParameter(
        fusion_params.size() - 1, original_hlo.shape(),
        absl::StrCat("parameter_", fusion_params.size() - 1)));
  }

  CHECK(fused_hlo_map.insert({node_id, fused_hlo}).second);
  return *fused_hlo;
}

// Builds the HLO instructions for the fusion represented by `fusion_plan`.
HloInstruction& BuildFusionTowardOperands(
    const FusionPlan& fusion_plan,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  absl::flat_hash_map<AdjacencyList::NodeId, HloInstruction*> fused_hlo_map;
  return BuildFusionTowardOperandsImpl(fusion_plan.graph.GetRoot(), fusion_plan,
                                       fused_hlo_map, builder, fusion_params);
}

// Grows the fusion toward the operands.
//
// This always succeeds.
//
// If it's not possible to fuse something, it fuses a parameter instead.
//
// The fusion can grow until it has `max_params` params and it can only grow
// with operations for which the DimOrder propagation works and they don't
// impose requirements contradicting the existing requirements.
//
// The return value contains the HLOs corresponding to `root_hlo` and the
// requirements corresponding to the whole fusion so far.
HlosAndRequirements FuseTowardOperands(
    HloInstruction& root_hlo, const DimensionOrder& root_dim_order,
    const std::optional<int>& max_params,
    const se::GpuComputeCapability& gpu_version,
    const DotProperties& properties, const DotRequirements& requirements_so_far,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  FusionPlanAndRequirements fusion_plan_and_reqs =
      BuildFusionPlanTowardOperands(root_hlo, root_dim_order, max_params,
                                    gpu_version, properties,
                                    requirements_so_far);
  HloInstruction& fused_hlo_or_param = BuildFusionTowardOperands(
      fusion_plan_and_reqs.fusion_plan, builder, fusion_params);
  return HlosAndRequirements{&root_hlo, &fused_hlo_or_param,
                             fusion_plan_and_reqs.requirements};
}

// Grows the fusion toward the given dot operand.
//
// This always succeeds.
//
// If it's not possible to fuse something, it fuses a parameter instead.
//
// The fusion can grow until it has `max_params` params and it can only grow
// with operations for which the DimOrder propagation works and they don't
// impose requirements contradicting the existing requirements.
//
// The return value contains the HLOs corresponding to the given dot operand and
// the requirements corresponding to the whole fusion so far.
absl::StatusOr<HlosAndRequirements> FuseDotOperand(
    HloInstruction& dot, int operand_index,
    const se::GpuComputeCapability& gpu_version,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  // Direct dot inputs have well defined dimension orders.
  ASSIGN_OR_RETURN(const FusionContext context,
                   FusionContext::FromDotOperand(dot, operand_index));
  HloInstruction& operand = *dot.mutable_operand(operand_index);
  return FuseTowardOperands(operand, context.dim_orders().at(&operand),
                            TritonFusionAnalysis::kMaxParameterPerDotOperand,
                            gpu_version, context.dot_properties(),
                            context.requirements(), builder, fusion_params);
}

// Grows the fusion toward the users.
//
// This always succeeds.
//
// The fusion can grow as long as the DimOrder propagation works and the users
// don't impose requirements contradicting the existing requirements.
//
// The return value contains the HLOs corresponding to the "lowest" fused user
// or `hlo` if no users can be fused.
//
// It also grows the fusion upward, toward the "other" operands of the users,
// but currently only in special cases, such as binary elementwise operation
// with broadcast of scalar constant.
HlosAndRequirements FuseTowardUsers(
    HloInstruction& hlo, HloInstruction& fused_hlo,
    const DimensionOrder& hlo_dim_order,
    const se::GpuComputeCapability& gpu_version,
    const DotProperties& properties, const DotRequirements& requirements,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  HlosAndRequirements existing_hlos_and_requirements = {&hlo, &fused_hlo,
                                                        requirements};
  if (hlo.user_count() != 1) {
    return existing_hlos_and_requirements;
  }
  HloInstruction& user = *hlo.users()[0];

  // Get the dim orders for the user.
  auto opt_user_result = GetUserDimOrdersAndCombinedReqsIfProfitable(
      hlo, hlo_dim_order, user, properties, gpu_version, requirements);
  if (!opt_user_result.has_value()) {
    return existing_hlos_and_requirements;
  }
  DimensionOrder user_dim_order = opt_user_result->dim_orders.at(&user);
  DotRequirements combined_requirements = opt_user_result->requirements;

  HloInstruction::InstructionVector new_operands;
  if (user.operand_count() == 1) {
    new_operands.push_back(&fused_hlo);
  } else {
    // Get the dim orders for the operands of the user.
    // We shouldn't do a profitability check here, we made that decision in
    // GetUserDimOrdersAndCombinedReqsIfProfitable.
    auto opt_operand_result = GetOperandDimOrdersAndCombinedReqs(
        user, user_dim_order, properties, gpu_version, combined_requirements);
    // This shouldn't fail, because currently we only encounter this when we
    // have just propagated down the DimOrders on a binary elementwise
    // operation (user). In that case propagating up the DimOrders should always
    // work.
    if (!opt_operand_result.has_value()) {
      return existing_hlos_and_requirements;
    }
    DimOrderMap operand_dim_orders = opt_operand_result->dim_orders;
    combined_requirements = opt_operand_result->requirements;

    // Fuse the other operands of the user.
    for (int i = 0; i < user.operand_count(); ++i) {
      HloInstruction& operand = *user.mutable_operand(i);
      if (&operand == &hlo) {
        new_operands.push_back(&fused_hlo);
      } else {
        HlosAndRequirements hlos_and_requirements = FuseTowardOperands(
            operand, operand_dim_orders.at(&operand),
            /*max_params=*/std::nullopt, gpu_version, properties,
            combined_requirements, builder, fusion_params);
        new_operands.push_back(hlos_and_requirements.fused_hlo);
        combined_requirements = hlos_and_requirements.requirements;
      }
    }
  }

  HloInstruction& fused_user = *builder.AddInstruction(
      user.CloneWithNewOperands(user.shape(), new_operands));
  return FuseTowardUsers(user, fused_user, user_dim_order, gpu_version,
                         properties, combined_requirements, builder,
                         fusion_params);
}

// Grows the fusion toward the users of the dot.
//
// This always succeeds.
//
// The fusion can grow as long as the DimOrder propagation works and the users
// don't impose requirements contradicting the existing requirements.
//
// The return value contains the HLOs corresponding to the "lowest" fused user
// or `dot` if no users can be fused.
//
// It also grows the fusion towards the "other" operands of the users, but
// currently only in special cases, such as binary elementwise operation with
// broadcast of scalar constant.
HlosAndRequirements FuseDotOutput(
    HloInstruction& dot, HloInstruction& fused_dot,
    const se::GpuComputeCapability& gpu_version,
    const DotRequirements& requirements,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  const auto context = FusionContext::FromDotOutput(dot, requirements);
  return FuseTowardUsers(dot, fused_dot, context.dim_orders().at(&dot),
                         gpu_version, context.dot_properties(),
                         context.requirements(), builder, fusion_params);
}

namespace {

// Holds all information necessary to insert a fusion computation into the
// original HLO module.
struct Fusion {
  // Ordered pointers to HLO instructions in the original module that are used
  // as parameters in the fusion computation.
  std::vector<HloInstruction*> inputs;
  // The fusion computation.
  std::unique_ptr<HloComputation> computation;
  // Pointer to the output in the original HLO module that the fusion
  // computation replaces.
  HloInstruction* output = nullptr;
};

// Some instructions can be codegened by Triton, but we don't allow them in
// GEMM fusions specifically or it doesn't make sense to fuse.
bool AllowedInGemmFusion(const HloInstruction& instr) {
  if (!instr.IsFusible()) {
    return false;
  }
  return HloPredicateIsNotOp<HloOpcode::kFusion, HloOpcode::kDot,
                             HloOpcode::kParameter, HloOpcode::kReduce>(&instr);
}

// Returns true if we should consider fusing the instruction into the GEMM
// fusion.
bool IncludeInSearchSpace(const HloInstruction& instr,
                          const se::GpuComputeCapability& gpu_version) {
  return AllowedInGemmFusion(instr) &&
         IsTritonSupportedInstruction(instr, gpu_version);
}

// Returns a set of descendants from `ancestor`, not including ancestor.
absl::flat_hash_set<const HloInstruction*> GetDescendants(
    const HloInstruction* ancestor) {
  absl::flat_hash_set<const HloInstruction*> descendants;
  std::queue<const HloInstruction*> worklist;
  worklist.push(ancestor);
  while (!worklist.empty()) {
    const HloInstruction* current = worklist.front();
    worklist.pop();
    for (const HloInstruction* user : current->users()) {
      if (descendants.insert(user).second) {
        worklist.push(user);
      }
    }
  }
  return descendants;
}

HloInstruction* CreateBitcastWithShape(Shape shape,
                                       HloInstruction& new_operand) {
  CopyElementType(new_operand.shape(), &shape);
  return new_operand.parent()->AddInstruction(
      HloInstruction::CreateBitcast(shape, &new_operand));
}

// Holds a module of the search space surrounding a dot instruction in order to
// create an optimal fusion.
class FusionSearchSpace {
 public:
  FusionSearchSpace(HloInstruction* dot,
                    const se::GpuComputeCapability& gpu_version)
      : original_dot_(dot) {
    module_ = std::make_unique<HloModule>(
        absl::StrCat(dot->name(), "_fusion_search_space"),
        dot->GetModule()->config());
    HloComputation::Builder builder(absl::StrCat(dot->name(), "_computation"));
    // Find the highest suitable user of the dot to be the root of the
    // fusion.
    HloInstruction* fusion_output = dot;
    while (fusion_output->user_count() == 1 &&
           IncludeInSearchSpace(*fusion_output->users()[0], gpu_version)) {
      fusion_output = fusion_output->users()[0];
    }
    // Starting from the root of the fusion, fuse upwards.
    HloInstruction* cloned_output =
        FuseOperandsRecursively(fusion_output, builder, gpu_version);
    entry_ = module_->AddEntryComputation(builder.Build(cloned_output));
  }

  HloInstruction* original_dot() const { return original_dot_; }
  HloComputation* entry() const { return entry_; }

  const absl::flat_hash_map<HloInstruction*, HloInstruction*>&
  original_to_fused() const {
    return original_to_fused_;
  }

  const absl::flat_hash_map<HloInstruction*, HloInstruction*>&
  fused_to_original() const {
    return fused_to_original_;
  }

  // Moves bitcasts in the search space computation away from the dot
  // instruction to make fusions more likely to tile.
  absl::Status ShepherdBitcastsAwayFromDot(HloComputation* computation);

  // Given a fusion operand + fusion parameter pair in the fusion search space,
  // returns the corresponding instruction in the original HLO module. If there
  // is a shape mismatch (due to bitcast shepherding) or it's a newly added
  // bitcast without a corresponding original instruction, it will create a new
  // instruction in the original module and return it.
  absl::StatusOr<HloInstruction*> GetOrCreateOriginalInstruction(
      HloInstruction* fusion_operand, HloInstruction* fusion_param);

 private:
  // Swaps a bitcast with its operand, if possible. Returns true if mutated.
  // If its operand is a parameter, it updates the parameter shape,
  // removes the bitcast, and returns true.
  absl::StatusOr<bool> HoistBitcast(HloInstruction* instr);

  // Swaps a bitcast with its user, if possible. Returns true if mutated.
  // If the bitcast is the root instruction, it removes the bitcast,
  // updates the root, and returns true.
  absl::StatusOr<bool> SinkBitcast(HloInstruction* instr);

  // Recursive DFS to create maximum possible fusion.
  HloInstruction* FuseOperandsRecursively(
      HloInstruction* instr, HloComputation::Builder& builder,
      const se::GpuComputeCapability& gpu_version);
  // A module containing the search space. Each op is cloned and can be mapped
  // to the original HLO with `fused_to_original`.
  std::unique_ptr<HloModule> module_;
  // Ordered pointers to HLO instructions in the original module that are used
  // as parameters in the fusion computation.
  std::vector<HloInstruction*> inputs_;
  // The entry computation of the module.
  HloComputation* entry_ = nullptr;
  // Maps between original and search space instructions.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> original_to_fused_;
  // Maps between search space and original instructions.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> fused_to_original_;
  // Pointer to the dot instruction in the original module.
  HloInstruction* original_dot_ = nullptr;
};

HloInstruction* FusionSearchSpace::FuseOperandsRecursively(
    HloInstruction* instr, HloComputation::Builder& builder,
    const se::GpuComputeCapability& gpu_version) {
  if (auto it = original_to_fused_.find(instr);
      it != original_to_fused_.end()) {
    // We have already processed this instruction. Return the corresponding
    // instruction within the fusion space.
    return it->second;
  }

  if (instr == original_dot_ || IncludeInSearchSpace(*instr, gpu_version)) {
    VLOG(10) << "Including in search space: " << instr->ToString();
    // Found a candidate to fuse - recurse on its operands.
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(instr->operand_count());
    for (HloInstruction* operand : instr->operands()) {
      new_operands.push_back(
          FuseOperandsRecursively(operand, builder, gpu_version));
    }
    HloInstruction* cloned = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
    original_to_fused_[instr] = cloned;
    fused_to_original_[cloned] = instr;
    return cloned;
  }

  // Boundary reached: create a parameter.
  VLOG(10) << "Not including in search space: " << instr->ToString();
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          inputs_.size(), instr->shape(),
          absl::StrCat("parameter_", inputs_.size())));
  inputs_.push_back(instr);
  original_to_fused_[instr] = param;
  fused_to_original_[param] = instr;
  return param;
}

// Shepherd bitcasts in the search space computation away from the dot
// instruction to make it more likely to tile.
absl::Status FusionSearchSpace::ShepherdBitcastsAwayFromDot(
    HloComputation* computation) {
  VLOG(5) << "Shepherding bitcasts away from dot in computation: "
          << computation->ToString();
  HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  if (dot == nullptr) {
    return absl::OkStatus();
  }

  bool changed = true;
  while (changed) {
    changed = false;
    absl::flat_hash_set<const HloInstruction*> descendants =
        GetDescendants(dot);
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (HloPredicateIsNotOp<HloOpcode::kBitcast, HloOpcode::kReshape>(
              instr)) {
        continue;
      }
      if (descendants.find(instr) == descendants.end()) {
        ASSIGN_OR_RETURN(changed, HoistBitcast(instr));
      } else {
        ASSIGN_OR_RETURN(changed, SinkBitcast(instr));
      }
      if (changed) {
        break;
      }
    }
  }
  VLOG(5) << "Computation after shepherding bitcasts: "
          << computation->ToString();
  return absl::OkStatus();
}

// Returns true if all users of the parameter are bitcast or reshape
// instructions with the given shape. This means we can safely swap out the
// parameter shape for the new shape.
bool CanReplaceParameterShape(HloInstruction* parameter,
                              const Shape& new_shape) {
  for (HloInstruction* user : parameter->users()) {
    if (HloPredicateIsNotOp<HloOpcode::kBitcast, HloOpcode::kReshape>(user) ||
        user->shape() != new_shape) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<bool> FusionSearchSpace::HoistBitcast(HloInstruction* instr) {
  HloInstruction* operand = instr->mutable_operand(0);
  HloInstruction* new_instr = nullptr;

  switch (operand->opcode()) {
    case HloOpcode::kParameter: {
      if (!CanReplaceParameterShape(operand, instr->shape())) {
        return false;
      }
      // Just update parameter shape in the submodule.
      // We will realize the bitcast in the original module later.
      Shape new_shape = instr->shape();
      CopyElementType(operand->shape(), &new_shape);

      *operand->mutable_shape() = new_shape;
      // Copy users to a new vector to avoid invalidating the iterator.
      std::vector<HloInstruction*> users(operand->users().begin(),
                                         operand->users().end());
      for (HloInstruction* user : users) {
        RETURN_IF_ERROR(user->ReplaceAllUsesWith(operand));
        RETURN_IF_ERROR(
            user->parent()->RemoveInstructionAndUnusedOperands(user));
      }
      return true;
    }
    case HloOpcode::kConstant: {
      auto* constant = Cast<HloConstantInstruction>(operand);
      if (!ShapeUtil::IsEffectiveScalar(constant->shape())) {
        return false;
      }
      ASSIGN_OR_RETURN(Literal new_literal, constant->literal().Reshape(
                                                instr->shape().dimensions()));
      new_instr = instr->parent()->AddInstruction(
          HloInstruction::CreateConstant(std::move(new_literal)));
      break;
    }
    case HloOpcode::kBroadcast: {
      absl::StatusOr<BitcastParams> params = CalculateBitcastOfBroadcast(
          Cast<HloBroadcastInstruction>(operand), instr->shape());
      if (!params.ok()) {
        return false;
      }
      HloInstruction* arg = operand->mutable_operand(0);
      HloInstruction* new_bitcast =
          CreateBitcastWithShape(params->new_shape, *arg);
      new_instr =
          instr->parent()->AddInstruction(HloInstruction::CreateBroadcast(
              instr->shape(), new_bitcast, params->new_dims));
      break;
    }
    case HloOpcode::kTranspose: {
      absl::StatusOr<BitcastParams> params = CalculateBitcastOfTranspose(
          Cast<HloTransposeInstruction>(operand), instr->shape());
      if (!params.ok()) {
        return false;
      }
      HloInstruction* arg = operand->mutable_operand(0);
      HloInstruction* new_bitcast =
          CreateBitcastWithShape(params->new_shape, *arg);
      new_instr =
          instr->parent()->AddInstruction(HloInstruction::CreateTranspose(
              instr->shape(), new_bitcast, params->new_dims));
      break;
    }
    default: {
      if (!operand->IsElementwise()) {
        return false;
      }
      // bitcast(op(a, b)) -> op(bitcast(a), bitcast(b))
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* arg : operand->operands()) {
        new_operands.push_back(CreateBitcastWithShape(instr->shape(), *arg));
      }
      new_instr = instr->parent()->AddInstruction(
          operand->CloneWithNewOperands(instr->shape(), new_operands));
      break;
    }
  }

  if (new_instr == nullptr) {
    return false;
  }

  // Preserve mapping to original HLO.
  if (auto it = fused_to_original_.find(operand);
      it != fused_to_original_.end()) {
    HloInstruction* original_op = it->second;
    fused_to_original_[new_instr] = original_op;
    original_to_fused_[original_op] = new_instr;
  }

  RETURN_IF_ERROR(instr->ReplaceAllUsesWith(new_instr));
  RETURN_IF_ERROR(instr->parent()->RemoveInstructionAndUnusedOperands(instr));
  return true;
}

absl::StatusOr<bool> FusionSearchSpace::SinkBitcast(HloInstruction* instr) {
  HloInstruction* operand = instr->mutable_operand(0);

  // If bitcast is root, strip it to push it out of the fusion.
  if (instr->IsRoot()) {
    instr->parent()->set_root_instruction(operand,
                                          /*accept_different_shape=*/true);
    RETURN_IF_ERROR(instr->parent()->RemoveInstructionAndUnusedOperands(instr));
    return true;
  }
  // We only build fusions where epilogues have single users. This could be
  // extended to handle the general case, but isn't needed for now.
  if (instr->user_count() != 1) {
    return false;
  }

  HloInstruction* user = instr->users()[0];
  HloInstruction* new_instr = nullptr;
  switch (user->opcode()) {
    case HloOpcode::kTranspose: {
      absl::StatusOr<BitcastParams> params = CalculateTransposeOfBitcast(
          Cast<HloTransposeInstruction>(user), operand->shape());
      if (!params.ok()) {
        return false;
      }
      Shape new_transpose_shape = params->new_shape;
      CopyElementType(operand->shape(), &new_transpose_shape);
      new_instr =
          instr->parent()->AddInstruction(HloInstruction::CreateTranspose(
              new_transpose_shape, operand, params->new_dims));
      break;
    }
    case HloOpcode::kBroadcast: {
      absl::StatusOr<BitcastParams> params = CalculateBroadcastOfBitcast(
          Cast<HloBroadcastInstruction>(user), operand->shape());
      if (!params.ok()) {
        return false;
      }
      Shape new_broadcast_shape = params->new_shape;
      CopyElementType(operand->shape(), &new_broadcast_shape);
      new_instr =
          instr->parent()->AddInstruction(HloInstruction::CreateBroadcast(
              new_broadcast_shape, operand, params->new_dims));
      break;
    }
    default: {
      if (!user->IsElementwise()) {
        return false;
      }
      // Sink past elementwise users.
      // op(bitcast(a), b) -> bitcast(op(a, bitcast(b)))
      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* arg : user->operands()) {
        if (arg == instr) {
          new_operands.push_back(operand);
        } else {
          new_operands.push_back(
              CreateBitcastWithShape(operand->shape(), *arg));
        }
      }
      Shape new_user_shape = operand->shape();
      CopyElementType(user->shape(), &new_user_shape);
      new_instr = instr->parent()->AddInstruction(
          user->CloneWithNewOperands(new_user_shape, new_operands));
      break;
    }
  }

  if (new_instr == nullptr) {
    return false;
  }
  HloInstruction* new_bitcast = instr->parent()->AddInstruction(
      HloInstruction::CreateBitcast(user->shape(), new_instr));
  // Preserve mapping to original HLO.
  if (auto it = fused_to_original_.find(user); it != fused_to_original_.end()) {
    HloInstruction* original_op = it->second;
    fused_to_original_[new_instr] = original_op;
    fused_to_original_[new_bitcast] = original_op;
    original_to_fused_[original_op] = new_instr;
  }
  RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_bitcast));
  RETURN_IF_ERROR(instr->parent()->RemoveInstructionAndUnusedOperands(user));
  return true;
}

// Checks if the fusion can be tiled by SymbolicTileAnalysis.
FusionDecision CanTile(mlir::MLIRContext& mlir_context,
                       const HloFusionAdaptor& fusion) {
  if (fusion.GetRoots()[0]
          .instruction()
          .GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_enable_tiling_propagation()) {
    namespace ge = ::xla::gpu::experimental;
    auto ts = ge::TilingSpace::Create(fusion, &mlir_context);
    if (!ts.ok()) {
      return FusionDecision::Forbid(absl::StrCat(
          "Failed to create tiling space: ", ts.status().message()));
    }
    auto tiled_computation =
        ge::TiledHloComputation::Tile(fusion, std::move(ts.value()));
    if (!tiled_computation.ok()) {
      return FusionDecision::Forbid(
          absl::StrCat("Fusion is not tileable with experimental tiling: ",
                       tiled_computation.status().message()));
    }
    return FusionDecision::Allow();
  }
  auto fusion_analysis =
      SymbolicTileAnalysis::AnalyzeFusion(fusion, &mlir_context);
  if (std::holds_alternative<FusionDecision>(fusion_analysis)) {
    return std::get<FusionDecision>(fusion_analysis);
  }
  return FusionDecision::Allow();
}

// Returns true if fusing `producer` into `consumer` is possible and supported.
FusionDecision CanFuse(mlir::MLIRContext& mlir_context,
                       HloInstruction* producer, HloInstruction* consumer) {
  // If the candidate is not a user of the fusion, we have already fused the
  // instruction.
  if (!consumer->IsUserOf(producer)) {
    return FusionDecision::Forbid("Consumer is not a user of producer.");
  }
  // Parameter means we have reached the end of the search space.
  if (HloPredicateIsOp<HloOpcode::kParameter>(producer)) {
    return FusionDecision::Forbid("Cannot fuse parameter.");
  }
  return CanTile(mlir_context,
                 *HloFusionAdaptor::ForProducerConsumer(producer, consumer));
}

// Attempts to fuse all candidates and their operands into the fusion.
void FuseOperandsBFS(mlir::MLIRContext& mlir_context,
                     const HloInstruction::InstructionVector& candidates,
                     HloInstruction* fusion) {
  std::queue<HloInstruction*> queue;
  for (HloInstruction* operand : candidates) {
    queue.push(operand);
  }
  while (!queue.empty() &&
         fusion->operand_count() <
             TritonFusionAnalysis::kMaxParameterPerDotOperand * 2) {
    HloInstruction* candidate = queue.front();
    queue.pop();

    if (FusionDecision decision = CanFuse(mlir_context, candidate, fusion);
        !decision.IsAllowed()) {
      VLOG(5) << "Cannot fuse operand: " << decision.Explain();
      continue;
    }
    VLOG(5) << "Fusing operand: " << candidate->ToString();
    fusion->FuseInstruction(candidate);
    for (HloInstruction* operand : candidate->operands()) {
      queue.push(operand);
    }
  }
}

// Fuses user into the fusion. If the user has more operands than the fusion,
// then it will also attempt to fuse its operands. Returns the new fusion as the
// only way to fuse a user is to create a new fusion.
absl::StatusOr<HloInstruction*> FuseUserAndOperands(
    mlir::MLIRContext& mlir_context, HloInstruction* fusion,
    HloInstruction* user) {
  int64_t operand_count = fusion->operand_count();
  HloInstruction* new_fusion =
      fusion->parent()->AddInstruction(HloInstruction::CreateFusion(
          user->shape(), HloInstruction::FusionKind::kCustom, user));
  RETURN_IF_ERROR(fusion->parent()->ReplaceInstruction(user, new_fusion));
  new_fusion->MergeFusionInstruction(fusion);
  CHECK_EQ(0, fusion->users().size());
  RETURN_IF_ERROR(fusion->parent()->RemoveInstruction(fusion));
  if (new_fusion->operand_count() > operand_count) {
    FuseOperandsBFS(mlir_context, new_fusion->operands(), new_fusion);
  }
  return new_fusion;
}

absl::StatusOr<HloInstruction*>
FusionSearchSpace::GetOrCreateOriginalInstruction(
    HloInstruction* fusion_operand, HloInstruction* fusion_param) {
  if (auto it = fused_to_original_.find(fusion_operand);
      it != fused_to_original_.end()) {
    HloInstruction* original_instr = it->second;
    if (original_instr->shape() == fusion_param->shape()) {
      return original_instr;
    }
    // Shape mismatch due to bitcast hoisting past boundaries.
    HloInstruction* instr = original_instr->parent()->AddInstruction(
        HloInstruction::CreateBitcast(fusion_param->shape(), original_instr));
    return instr;
  }
  // Some bitcasts were created in the search space and do not have a
  // corresponding original instruction. Find their parent and add the bitcast
  // to the original module.
  if (fusion_operand->opcode() == HloOpcode::kBitcast) {
    ASSIGN_OR_RETURN(HloInstruction * original_operand,
                     GetOrCreateOriginalInstruction(
                         fusion_operand->mutable_operand(0), fusion_param));
    HloInstruction* bitcast = original_operand->parent()->AddInstruction(
        HloInstruction::CreateBitcast(fusion_param->shape(), original_operand));
    fused_to_original_[fusion_operand] = bitcast;
    original_to_fused_[bitcast] = fusion_operand;
    return bitcast;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Non-bitcast instruction not mapped to original module: ",
                   fusion_operand->ToString()));
}

// Creates a fusion from the search space. Starting from the dot instruction,
// it fuses tileable operands using BFS. Then it fuses tileable users and their
// operands until it reaches the root of the search space.
absl::StatusOr<std::variant<Fusion, FusionDecision>> CreateTileableFusion(
    FusionSearchSpace& fusion_search_space,
    const se::GpuComputeCapability gpu_version, absl::string_view name) {
  HloInstruction* original_dot = fusion_search_space.original_dot();
  HloInstruction* dot =
      fusion_search_space.original_to_fused().at(original_dot);
  mlir::MLIRContext mlir_context;
  if (!CanTile(mlir_context, *HloFusionAdaptor::ForInstruction(dot))) {
    return FusionDecision::Forbid("Cannot tile the dot instruction.");
  }

  // Start with a fusion containing only the dot instruction.
  auto entry = fusion_search_space.entry();
  auto fusion = entry->AddInstruction(HloInstruction::CreateFusion(
      dot->shape(), HloInstruction::FusionKind::kCustom, dot));
  RETURN_IF_ERROR(entry->ReplaceInstruction(dot, fusion));

  // Keep track of the original HLO the fusion is replacing.
  HloInstruction* original_output = original_dot;

  // BFS of operands until we cannot tile.
  FuseOperandsBFS(mlir_context, fusion->operands(), fusion);

  // Fuse in users until we cannot tile or reach the root.
  while (!fusion->IsRoot()) {
    // Search space was created so that the result only ever has a single user.
    CHECK_EQ(fusion->users().size(), 1);
    auto user = fusion->users()[0];
    if (FusionDecision decision = CanFuse(mlir_context, fusion, user);
        !decision.IsAllowed()) {
      VLOG(5) << "Cannot fuse user: " << decision.Explain();
      break;
    }
    VLOG(5) << "Fusing user into epilogue: " << user->ToString();
    ASSIGN_OR_RETURN(fusion, FuseUserAndOperands(mlir_context, fusion, user));
    original_output = fusion_search_space.fused_to_original().at(user);
  }

  // Move bitcasts out of the fusion computation in case we have some on the
  // boundary.
  auto fusion_computation =
      Cast<HloFusionInstruction>(fusion)->fused_instructions_computation();
  RETURN_IF_ERROR(
      fusion_search_space.ShepherdBitcastsAwayFromDot(fusion_computation));

  // Find inputs to the fusion from the original module.
  std::vector<HloInstruction*> fusion_inputs;
  for (auto [index, operand] : llvm::enumerate(fusion->operands())) {
    HloInstruction* fusion_param =
        fusion_computation->parameter_instruction(index);
    ASSIGN_OR_RETURN(HloInstruction * original_instruction,
                     fusion_search_space.GetOrCreateOriginalInstruction(
                         operand, fusion_param));
    fusion_inputs.push_back(original_instruction);
  }

  return Fusion{std::move(fusion_inputs),
                fusion_computation->Clone(std::string(name)), original_output};
}

absl::StatusOr<std::variant<Fusion, FusionDecision>> CreateDotFusionV2(
    HloDotInstruction& dot, const se::GpuComputeCapability gpu_version,
    absl::string_view name) {
  VLOG(3) << "Creating dot fusion v2 around dot: " << dot.ToString();
  FusionSearchSpace fusion_search_space(&dot, gpu_version);
  VLOG(3) << "Found fusion search space: \n"
          << fusion_search_space.entry()->ToString();

  RETURN_IF_ERROR(fusion_search_space.ShepherdBitcastsAwayFromDot(
      fusion_search_space.entry()));

  return CreateTileableFusion(fusion_search_space, gpu_version, name);
}

}  // namespace

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation.
absl::StatusOr<std::variant<Fusion, FusionDecision>> CreateDotFusion(
    HloDotInstruction& dot, const se::GpuComputeCapability gpu_version,
    absl::string_view name) {
  VLOG(5) << dot.ToString();
  if (CodegenDecision is_supported =
          IsTritonSupportedInstruction(dot, gpu_version);
      !is_supported) {
    return is_supported;
  }

  if (dot.GetModule()
          ->config()
          .debug_options()
          .xla_gpu_experimental_gemm_fusion_v2()) {
    return CreateDotFusionV2(dot, gpu_version, name);
  }

  HloComputation::Builder builder(name);
  std::vector<HloInstruction*> fusion_inputs;

  std::vector<HlosAndRequirements> hlos_and_reqs;
  hlos_and_reqs.reserve(dot.operand_count());
  ASSIGN_OR_RETURN(HlosAndRequirements lhs_hlos_and_reqs,
                   FuseDotOperand(dot, /*operand_index=*/0, gpu_version,
                                  builder, fusion_inputs));
  hlos_and_reqs.push_back(lhs_hlos_and_reqs);
  ASSIGN_OR_RETURN(HlosAndRequirements rhs_hlos_and_reqs,
                   FuseDotOperand(dot, /*operand_index=*/1, gpu_version,
                                  builder, fusion_inputs));
  hlos_and_reqs.push_back(rhs_hlos_and_reqs);
  HloInstruction& fused_dot = FuseDot(dot, hlos_and_reqs, builder);
  // For now the RHS doesn't support splits, so it also doesn't impose any
  // requirements.
  HlosAndRequirements fused_output_and_reqs =
      FuseDotOutput(dot, fused_dot, gpu_version, lhs_hlos_and_reqs.requirements,
                    builder, fusion_inputs);

  HloInstruction* fusion_output = fused_output_and_reqs.original_hlo;

  // We cannot handle int4 parameters if the batch dimension is the minor one.
  // The cost of analysis could be expensive, so we only do it if we have to.
  bool has_int4_param =
      absl::c_any_of(fusion_inputs, [](const HloInstruction* hlo) {
        return hlo->shape().element_type() == PrimitiveType::S4;
      });
  if (has_int4_param) {
    // Trace the position of the batch dimension of the dot to the parameters.
    auto analysis_or = TritonFusionAnalysis::Execute(dot);
    if (analysis_or.ok()) {
      const auto& analysis = analysis_or.value();
      if (!analysis.IsBatchDimMinorForInt4Parameter(
              dot, TritonFusionAnalysis::Scope::LHS) ||
          !analysis.IsBatchDimMinorForInt4Parameter(
              dot, TritonFusionAnalysis::Scope::RHS)) {
        return FusionDecision::Forbid(
            "Fusion is not possible because the parameter with the type S4 has "
            "minor batch dimension.");
      }
    }
  }

  const PrecisionConfig::Algorithm algorithm =
      dot.precision_config().algorithm();
  if (algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X9 ||
      algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6 ||
      algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
      algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32 ||
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32 ||
      algorithm == PrecisionConfig::ALG_DOT_TF32_TF32_F32_X3 ||
      algorithm == PrecisionConfig::ALG_DOT_F32_F32_F32 ||
      dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
    return Fusion{std::move(fusion_inputs), builder.Build(), fusion_output};
  }

  bool is_pure_matmul = true;
  (void)builder.ForEachInstruction([&](const HloInstruction* fused_hlo) {
    static constexpr std::array<HloOpcode, 4> kPureOpcodes = {
        HloOpcode::kBitcast, HloOpcode::kDot, HloOpcode::kParameter,
        HloOpcode::kReshape};
    if (absl::c_find(kPureOpcodes, fused_hlo->opcode()) == kPureOpcodes.end()) {
      is_pure_matmul = false;
      // Stop iterating.
      return absl::CancelledError();
    }
    return absl::OkStatus();
  });

  if (is_pure_matmul) {
    return FusionDecision::Forbid("Pure Matmul");
  }

  return Fusion{std::move(fusion_inputs), builder.Build(), fusion_output};
}

// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmFusionVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmFusionVisitor(const se::GpuComputeCapability& gpu_version)
      : gpu_version_(gpu_version) {}
  // Checks that a dot() should be targeting the triton GEMM emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  absl::Status HandleDot(HloInstruction* dot) override {
    CHECK_EQ(dot->opcode(), HloOpcode::kDot);

    int64_t gemm_rewrite_size_threshold =
        dot->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_gemm_rewrite_size_threshold();
    ASSIGN_OR_RETURN(bool is_matmul_tiny,
                     IsMatrixMultiplicationTooSmallForRewriting(
                         *dot, gemm_rewrite_size_threshold));
    if (is_matmul_tiny && IsDotSupportedByClassicalEmitters(*dot)) {
      return absl::OkStatus();
    }

    std::string fusion_name = absl::StrCat("gemm_fusion_", dot->name());
    ASSIGN_OR_RETURN(
        auto fusion_or_decision,
        CreateDotFusion(*Cast<HloDotInstruction>(dot), gpu_version_,
                        absl::StrCat(fusion_name, "_computation")));

    if (std::holds_alternative<FusionDecision>(fusion_or_decision)) {
      const FusionDecision& decision =
          std::get<FusionDecision>(fusion_or_decision);
      VLOG(3) << "Not fusing: " << decision.Explain();
      return absl::OkStatus();
    }

    Fusion fusion = std::get<Fusion>(std::move(fusion_or_decision));

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(
            std::move(fusion.computation), /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion.inputs, computation));
    // Copy the metadata of the `dot` to the newly created `fusion` op. This
    // is convenient for handling metadata in split-k rewriting subsequently.
    dot_fusion->set_metadata(dot->metadata());
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion, fusion_name);

    ASSIGN_OR_RETURN(auto gpu_config,
                     dot_fusion->backend_config<GpuBackendConfig>());
    FusionBackendConfig& backend_config =
        *gpu_config.mutable_fusion_backend_config();
    backend_config.set_kind(kTritonGemmFusionKind);
    RETURN_IF_ERROR(dot_fusion->set_backend_config(gpu_config));

    HloInstruction* replacement = dot_fusion;
    if (fusion.output->shape() != dot_fusion->shape()) {
      replacement = dot->parent()->AddInstruction(
          HloInstruction::CreateBitcast(fusion.output->shape(), dot_fusion));
    }

    if (fusion.output->IsRoot()) {
      fusion.output->parent()->set_root_instruction(replacement);
      RETURN_IF_ERROR(
          fusion.output->parent()->RemoveInstructionAndUnusedOperands(
              fusion.output));
      MarkAsChanged();
    } else {
      RETURN_IF_ERROR(ReplaceInstruction(fusion.output, replacement));
    }
    XLA_VLOG_LINES(5, computation->ToString(HloPrintOptions::ShortParsable()));
    return absl::OkStatus();
  }

  absl::Status HandleScaledDot(HloInstruction* scaled_dot) override {
    CHECK_EQ(scaled_dot->opcode(), HloOpcode::kScaledDot);
    HloComputation::Builder builder(
        absl::StrCat("fusion_", scaled_dot->name()));

    std::vector<HloInstruction*> fusion_inputs;

    std::vector<HlosAndRequirements> hlos_and_reqs;
    hlos_and_reqs.reserve(scaled_dot->operand_count());
    ASSIGN_OR_RETURN(HlosAndRequirements lhs_hlos_and_reqs,
                     FuseDotOperand(*scaled_dot, /*operand_index=*/0,
                                    gpu_version_, builder, fusion_inputs));
    hlos_and_reqs.push_back(lhs_hlos_and_reqs);
    ASSIGN_OR_RETURN(HlosAndRequirements rhs_hlos_and_reqs,
                     FuseDotOperand(*scaled_dot, /*operand_index=*/1,
                                    gpu_version_, builder, fusion_inputs));
    hlos_and_reqs.push_back(rhs_hlos_and_reqs);
    ASSIGN_OR_RETURN(HlosAndRequirements lhs_scale_hlos_and_reqs,
                     FuseDotOperand(*scaled_dot, /*operand_index=*/2,
                                    gpu_version_, builder, fusion_inputs));
    hlos_and_reqs.push_back(lhs_scale_hlos_and_reqs);
    ASSIGN_OR_RETURN(HlosAndRequirements rhs_scale_hlos_and_reqs,
                     FuseDotOperand(*scaled_dot, /*operand_index=*/3,
                                    gpu_version_, builder, fusion_inputs));
    hlos_and_reqs.push_back(rhs_scale_hlos_and_reqs);

    HloInstruction& fused_dot = FuseDot(*scaled_dot, hlos_and_reqs, builder);

    HlosAndRequirements fused_output_and_reqs =
        FuseDotOutput(*scaled_dot, fused_dot, gpu_version_,
                      lhs_hlos_and_reqs.requirements, builder, fusion_inputs);
    HloComputation* computation =
        scaled_dot->GetModule()->AddComputationAndUnifyNamesAndIds(
            builder.Build(),
            /*is_entry=*/false);

    HloInstruction* fusion =
        scaled_dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion_inputs, computation));

    ASSIGN_OR_RETURN(auto gpu_config,
                     fusion->backend_config<GpuBackendConfig>());
    FusionBackendConfig& backend_config =
        *gpu_config.mutable_fusion_backend_config();
    backend_config.set_kind(kTritonGemmFusionKind);
    RETURN_IF_ERROR(fusion->set_backend_config(gpu_config));
    HloInstruction* fusion_output = fused_output_and_reqs.original_hlo;
    RETURN_IF_ERROR(ReplaceInstruction(fusion_output, fusion));
    MarkAsChanged();
    return absl::OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_version_;
};

absl::StatusOr<bool> RunOnComputation(
    HloComputation* computation, const se::GpuComputeCapability& gpu_version) {
  GemmFusionVisitor visitor(gpu_version);
  RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // namespace

absl::StatusOr<bool> GemmFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  RETURN_IF_ERROR(EnsureTritonSupportsComputeCapability(compute_capability_));

  bool changed = false;
  for (HloComputation* computation :
       GetFusibleComputations(*module, execution_threads)) {
    ASSIGN_OR_RETURN(bool result,
                     RunOnComputation(computation, compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

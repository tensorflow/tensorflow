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

#include "xla/service/gpu/gemm_fusion.h"

#include <array>
#include <cstddef>
#include <cstdint>
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
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_support.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {

namespace {

using triton_fusion::CombineRequirements;
using triton_fusion::DimensionOrder;
using triton_fusion::DimOrderMap;
using triton_fusion::DimOrdersAndReqs;
using triton_fusion::DimOrdersAndReqsOrError;
using triton_fusion::DotRequirements;
using triton_fusion::FusionContext;
using triton_fusion::GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible;
using triton_fusion::HeroProperties;
using triton_fusion::Requirements;
using triton_fusion::RequirementsOrError;
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

struct HloAndDimOrder {
  const HloInstruction* original_hlo = nullptr;
  DimensionOrder dim_order;
};

struct HloAndIterSpec {
  const HloInstruction* original_hlo;
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
  const HloInstruction* original_hlo = nullptr;
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
  Requirements requirements;
};

struct HlosAndRequirements {
  // The original HLO (which is outside the fusion computation).
  const HloInstruction* original_hlo = nullptr;
  // The fused HLO inside the new fusion computation, built by the builder.
  //
  // This can have the same opcode as `original_hlo` or it can be a parameter if
  // the original HLO can't be fused.
  const HloInstruction* fused_hlo = nullptr;
  // The requirements imposed by the fused operations.
  //
  // If we fuse further operations they may have to conform to these
  // requirements.
  Requirements requirements;
};

// Clones the hero kDot operation into the fusion.
HloInstruction& FuseDot(const HloDotInstruction& dot,
                        const HloInstruction& fused_lhs,
                        const HloInstruction& fused_rhs,
                        std::optional<const HloInstruction*> fused_meta,
                        HloComputation::Builder& builder  // append
) {
  VLOG(3) << "Fusing " << dot.ToString();

  std::vector<HloInstruction*> hlo_new_operands = {
      const_cast<HloInstruction*>(&fused_lhs),
      const_cast<HloInstruction*>(&fused_rhs)};
  if (fused_meta.has_value()) {
    hlo_new_operands.push_back(const_cast<HloInstruction*>(fused_meta.value()));
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
    const HeroProperties& properties,
    const se::GpuComputeCapability& gpu_version,
    const Requirements& requirements) {
  DimOrdersAndReqsOrError dim_orders_and_new_reqs =
      GetPropagatedDimOrdersAndRequirements(
          hlo, dim_order, TransformDirection::kOutputToInput, properties);
  if (!std::holds_alternative<DimOrdersAndReqs>(dim_orders_and_new_reqs)) {
    return std::nullopt;
  }
  RequirementsOrError combined_reqs = CombineRequirements(
      requirements,
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).requirements);
  if (!std::holds_alternative<Requirements>(combined_reqs)) {
    return std::nullopt;
  }
  return DimOrdersAndReqs{
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).dim_orders,
      std::get<Requirements>(combined_reqs)};
}

// Just a helper to reduce "unwrapping" code where we use this.
std::optional<DimOrdersAndReqs> GetOperandDimOrdersAndCombinedReqsIfProfitable(
    const HloInstruction& hlo, const DimensionOrder& dim_order,
    const HeroProperties& properties,
    const se::GpuComputeCapability& gpu_version,
    const Requirements& requirements) {
  DimOrdersAndReqsOrError dim_orders_and_new_reqs =
      GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
          hlo, TransformDirection::kOutputToInput,
          /*src_operand_index=*/std::nullopt, dim_order, gpu_version,
          properties);
  if (!std::holds_alternative<DimOrdersAndReqs>(dim_orders_and_new_reqs)) {
    return std::nullopt;
  }
  RequirementsOrError combined_reqs = CombineRequirements(
      requirements,
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).requirements);
  if (!std::holds_alternative<Requirements>(combined_reqs)) {
    return std::nullopt;
  }
  return DimOrdersAndReqs{
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).dim_orders,
      std::get<Requirements>(combined_reqs)};
}

// Just a helper to reduce "unwrapping" code where we use this.
std::optional<DimOrdersAndReqs> GetUserDimOrdersAndCombinedReqsIfProfitable(
    const HloInstruction& hlo, const DimensionOrder& hlo_dim_order,
    const HloInstruction& user, const HeroProperties& properties,
    const se::GpuComputeCapability& gpu_version,
    const Requirements& requirements) {
  DimOrdersAndReqsOrError dim_orders_and_new_reqs =
      GetPropagatedDimOrdersAndRequirementsIfProfitablyFusible(
          user, TransformDirection::kInputToOutput, user.operand_index(&hlo),
          hlo_dim_order, gpu_version, properties);
  if (!std::holds_alternative<DimOrdersAndReqs>(dim_orders_and_new_reqs)) {
    return std::nullopt;
  }
  RequirementsOrError combined_reqs = CombineRequirements(
      requirements,
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).requirements);
  if (!std::holds_alternative<Requirements>(combined_reqs)) {
    return std::nullopt;
  }
  return DimOrdersAndReqs{
      std::get<DimOrdersAndReqs>(dim_orders_and_new_reqs).dim_orders,
      std::get<Requirements>(combined_reqs)};
}

// Builds the fusion map and the requirements which can later be used to
// actually fuse that subgraph.
FusionPlanAndRequirements BuildFusionPlanTowardOperands(
    const HloInstruction& root_hlo, const DimensionOrder& root_dim_order,
    const std::optional<int>& max_params,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties, const Requirements& requirements_so_far) {
  CHECK(!max_params.has_value() || max_params.value() >= 1);

  // The graph describing the structure of the fusion that we build - nodes
  // corresponding to the instructions and arcs pointing from users to operands.
  // We can build and modify this graph easily without the need to create
  // HloInstructions at this point.
  AdjacencyList graph;
  // Stores the original HLO and the dimension order for each node. This is a
  // temporary map which is used when processing the nodes in this function.
  absl::flat_hash_map<AdjacencyList::NodeId, HloAndDimOrder>
      hlo_and_dim_order_map;
  // Stores the information needed to build the fused HLO for each node (what
  // was the original HLO and whether we should fuse it or create a parameter).
  // This is one of the outputs of this function.
  absl::flat_hash_map<AdjacencyList::NodeId, NodeFusionPlan> fusion_plan_map;
  // Allows reusing nodes when multiple instructions iterate over the same HLO
  // using the same iteration spec. In that case we don't duplicate the
  // instruction in the fusion.
  absl::flat_hash_map<HloAndIterSpec, AdjacencyList::NodeId> node_reuse_map;
  // The requirements imposed by the fusion choices made in this function,
  // combined with the existing requirements. This is one of the outputs of this
  // function.
  Requirements combined_reqs = requirements_so_far;

  auto get_or_create_fusion_node =
      [&](const HloInstruction& hlo, const DimensionOrder& dim_order,
          bool* is_new_node = nullptr) -> AdjacencyList::NodeId {
    HloAndIterSpec reuse_key = {&hlo, dim_order.ToTensorIterationSpec()};
    if (auto it = node_reuse_map.find(reuse_key); it != node_reuse_map.end()) {
      if (is_new_node != nullptr) {
        *is_new_node = false;
      }
      return it->second;
    }
    AdjacencyList::NodeId node_id = graph.AddNode();
    CHECK(hlo_and_dim_order_map.insert({node_id, {&hlo, dim_order}}).second);
    CHECK(node_reuse_map.insert({reuse_key, node_id}).second);
    if (is_new_node != nullptr) {
      *is_new_node = true;
    }
    return node_id;
  };
  AdjacencyList::NodeId root =
      get_or_create_fusion_node(root_hlo, root_dim_order);

  // Nodes at the fusion edge that can either get fused too or become parameters
  // of the fusion. Used to track the number of parameters.
  absl::flat_hash_set<AdjacencyList::NodeId> inputs({root});
  std::queue<AdjacencyList::NodeId> queue({root});
  int64_t num_requeued = 0;
  // BFS
  while (queue.size() > num_requeued) {
    AdjacencyList::NodeId node_id = queue.front();
    queue.pop();
    const HloAndDimOrder& hlo_and_dim_order = hlo_and_dim_order_map.at(node_id);
    const HloInstruction& original_hlo = *hlo_and_dim_order.original_hlo;
    const DimensionOrder& dim_order = hlo_and_dim_order.dim_order;

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
      CHECK(fusion_plan_map
                .insert({node_id, {&original_hlo, /*should_fuse=*/false}})
                .second);
      continue;
    }
    auto opt_result = GetOperandDimOrdersAndCombinedReqsIfProfitable(
        original_hlo, dim_order, properties, gpu_version, combined_reqs);
    if (!opt_result.has_value()) {
      CHECK(fusion_plan_map
                .insert({node_id, {&original_hlo, /*should_fuse=*/false}})
                .second);
      continue;
    }
    const DimOrderMap operand_dim_orders = std::move(opt_result->dim_orders);
    combined_reqs = std::move(opt_result->requirements);
    inputs.erase(node_id);
    graph.ReserveSpaceForOutNeighbors(node_id, original_hlo.operand_count());
    for (int64_t i = 0; i < original_hlo.operand_count(); ++i) {
      const HloInstruction& operand = *original_hlo.operand(i);
      const DimensionOrder& operand_dim_order = operand_dim_orders.at(&operand);
      bool is_new_node = false;
      AdjacencyList::NodeId operand_node_id =
          get_or_create_fusion_node(operand, operand_dim_order, &is_new_node);
      graph.AddArc(node_id, operand_node_id);
      if (is_new_node) {
        VLOG(6) << "Enqueueing " << operand.ToString() << ":"
                << operand_dim_order.ToString();
        inputs.insert(operand_node_id);
        queue.push(operand_node_id);
      }
    }
    CHECK(
        fusion_plan_map.insert({node_id, {&original_hlo, /*should_fuse=*/true}})
            .second);
  }
  // Handle the remaining requeued items.
  while (!queue.empty()) {
    AdjacencyList::NodeId node_id = queue.front();
    queue.pop();

    const HloAndDimOrder& hlo_and_dim_order = hlo_and_dim_order_map.at(node_id);
    CHECK(fusion_plan_map
              .insert({node_id,
                       {hlo_and_dim_order.original_hlo, /*should_fuse=*/false}})
              .second);
  }
  return {{std::move(graph), std::move(fusion_plan_map)},
          std::move(combined_reqs)};
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
  const HloInstruction& original_hlo = *node_fusion_plan.original_hlo;

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
    fusion_params.push_back(const_cast<HloInstruction*>(&original_hlo));
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
    const HloInstruction& root_hlo, const DimensionOrder& root_dim_order,
    const std::optional<int>& max_params,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties, const Requirements& requirements_so_far,
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
    const HloInstruction& dot, int operand_index,
    const se::GpuComputeCapability& gpu_version,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  // Direct dot inputs have well defined dimension orders.
  TF_ASSIGN_OR_RETURN(const FusionContext context,
                      FusionContext::FromDotOperand(dot, operand_index));
  const HloInstruction& operand = *dot.operand(operand_index);
  return FuseTowardOperands(operand, context.dim_orders().at(&operand),
                            TritonFusionAnalysis::kMaxParameterPerDotOperand,
                            gpu_version, context.hero_properties(),
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
    const HloInstruction& hlo, const HloInstruction& fused_hlo,
    const DimensionOrder& hlo_dim_order,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties, const Requirements& requirements,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  const HlosAndRequirements existing_hlos_and_requirements = {&hlo, &fused_hlo,
                                                              requirements};
  if (hlo.user_count() != 1) {
    return existing_hlos_and_requirements;
  }
  const HloInstruction& user = *hlo.users()[0];
  if (!IsDistributiveOverAddition(user)) {
    return existing_hlos_and_requirements;
  }

  // Get the dim orders for the user.
  auto opt_user_result = GetUserDimOrdersAndCombinedReqsIfProfitable(
      hlo, hlo_dim_order, user, properties, gpu_version, requirements);
  if (!opt_user_result.has_value()) {
    return existing_hlos_and_requirements;
  }
  DimensionOrder user_dim_order = opt_user_result->dim_orders.at(&user);
  Requirements combined_requirements = opt_user_result->requirements;

  HloInstruction::InstructionVector new_operands;
  if (user.operand_count() == 1) {
    new_operands.push_back(const_cast<HloInstruction*>(&fused_hlo));
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
      const HloInstruction& operand = *user.operand(i);
      if (&operand == &hlo) {
        new_operands.push_back(const_cast<HloInstruction*>(&fused_hlo));
      } else {
        HlosAndRequirements hlos_and_requirements = FuseTowardOperands(
            operand, operand_dim_orders.at(&operand),
            /*max_params=*/std::nullopt, gpu_version, properties,
            combined_requirements, builder, fusion_params);
        new_operands.push_back(
            const_cast<HloInstruction*>(hlos_and_requirements.fused_hlo));
        combined_requirements = hlos_and_requirements.requirements;
      }
    }
  }

  const HloInstruction& fused_user = *builder.AddInstruction(
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
    const HloInstruction& dot, const HloInstruction& fused_dot,
    const se::GpuComputeCapability& gpu_version,
    const DotRequirements& requirements,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  const auto context =
      FusionContext::FromDotOutput(dot, /*split_k=*/1, requirements);
  return FuseTowardUsers(dot, fused_dot, context.dim_orders().at(&dot),
                         gpu_version, context.hero_properties(),
                         context.requirements(), builder, fusion_params);
}

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation constructed using the builder. fusion_inputs
// get populated with the non-fused instructions that become operands of the
// call to this fusion. fusion_output_ptr (if not nullptr) gets assigned the
// original instruction that has to be replaced by the call to the fusion.
absl::StatusOr<FusionDecision> CreateDotFusion(
    const HloDotInstruction& dot, const se::GpuComputeCapability gpu_version,
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_inputs,
    HloInstruction** fusion_output_ptr) {
  VLOG(5) << dot.ToString();
  if (CodegenDecision is_supported =
          IsTritonSupportedInstruction(dot, gpu_version);
      !is_supported) {
    VLOG(3) << is_supported.Explain();
    return is_supported;
  }

  // Verify sparse dot constraints.
  if (dot.sparse_operands()) {
    const SparsityDescriptor& descriptor = dot.sparsity().front();
    if (dot.sparse_operands() != 1 || descriptor.index() != 0) {
      return InvalidArgument("Sparsity is only supported on left operand");
    }
    if (descriptor.type() != SparsityType::SPARSITY_STRUCTURED_N_M ||
        descriptor.n() != 2 || descriptor.m() != 4) {
      return InvalidArgument("Only 2:4 structured sparsity is supported");
    }
    // DotDimensionSorter pass makes sure the sparse dimension is minor.
    CHECK_EQ(descriptor.dimension(), dot.operand(0)->shape().rank() - 1);
  }

  TF_ASSIGN_OR_RETURN(HlosAndRequirements lhs_hlos_and_reqs,
                      FuseDotOperand(dot, /*operand_index=*/0, gpu_version,
                                     builder, fusion_inputs));
  TF_ASSIGN_OR_RETURN(HlosAndRequirements rhs_hlos_and_reqs,
                      FuseDotOperand(dot, /*operand_index=*/1, gpu_version,
                                     builder, fusion_inputs));
  std::optional<const HloInstruction*> meta_hlo;
  if (dot.sparse_operands()) {
    TF_ASSIGN_OR_RETURN(HlosAndRequirements meta_hlos_and_reqs,
                        FuseDotOperand(dot, /*operand_index=*/2, gpu_version,
                                       builder, fusion_inputs));
    meta_hlo.emplace(meta_hlos_and_reqs.fused_hlo);
  }
  HloInstruction& fused_dot =
      FuseDot(dot, *lhs_hlos_and_reqs.fused_hlo, *rhs_hlos_and_reqs.fused_hlo,
              meta_hlo, builder);
  // For now the RHS doesn't support splits, so it also doesn't impose any
  // requirements.
  HlosAndRequirements fused_output_and_reqs =
      FuseDotOutput(dot, fused_dot, gpu_version,
                    std::get<DotRequirements>(lhs_hlos_and_reqs.requirements),
                    builder, fusion_inputs);

  if (fusion_output_ptr != nullptr) {
    *fusion_output_ptr =
        const_cast<HloInstruction*>(fused_output_and_reqs.original_hlo);
  }

  const PrecisionConfig::Algorithm algorithm =
      dot.precision_config().algorithm();
  if (algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6 ||
      algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3 ||
      dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any() ||
      dot.sparse_operands()) {
    return FusionDecision{};
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
  if (!is_pure_matmul) {
    return FusionDecision{};
  }

  return "No profitable operations to fuse.";
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
    TF_ASSIGN_OR_RETURN(bool is_matmul_tiny,
                        IsMatrixMultiplicationTooSmallForRewriting(
                            *dot, gemm_rewrite_size_threshold));
    if (is_matmul_tiny && IsDotSupportedByClassicalEmitters(*dot)) {
      return absl::OkStatus();
    }

    std::string fusion_name = absl::StrCat("gemm_fusion_", dot->name());
    HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));
    std::vector<HloInstruction*> fusion_inputs;
    HloInstruction* fusion_output = nullptr;
    TF_ASSIGN_OR_RETURN(
        const FusionDecision should_fuse,
        CreateDotFusion(*Cast<HloDotInstruction>(dot), gpu_version_, builder,
                        fusion_inputs, &fusion_output));
    if (builder.last_added_instruction() == nullptr) {
      return absl::OkStatus();
    }
    // If a GEMM requiring padding for cuBLAS is encountered here this
    // happened because earlier ShouldTritonHandleGEMM() accepted it and padding
    // was skipped. Accept it ignoring profitability checks.
    // TODO(rocm): check ROCM padding requirements.
    if (std::holds_alternative<se::CudaComputeCapability>(gpu_version_)) {
      if (!CublasRequiresPadding(
              *Cast<HloDotInstruction>(dot),
              std::get<se::CudaComputeCapability>(gpu_version_)) &&
          !should_fuse) {
        return OkStatus();
      }
    }

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion_inputs, computation));
    // Copy the metadata of the `dot` to the newly created `fusion` op. This
    // is convenient for handling metadata in split-k rewriting subsequently.
    dot_fusion->set_metadata(dot->metadata());
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion, fusion_name);

    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        dot_fusion->backend_config<GpuBackendConfig>());
    FusionBackendConfig& backend_config =
        *gpu_config.mutable_fusion_backend_config();
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(gpu_config));

    if (fusion_output->IsRoot()) {
      fusion_output->parent()->set_root_instruction(dot_fusion);
      TF_RETURN_IF_ERROR(
          fusion_output->parent()->RemoveInstructionAndUnusedOperands(
              fusion_output));
      MarkAsChanged();
    } else {
      TF_RETURN_IF_ERROR(ReplaceInstruction(fusion_output, dot_fusion));
    }
    XLA_VLOG_LINES(5, computation->ToString(HloPrintOptions::ShortParsable()));
    return absl::OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_version_;
};

absl::StatusOr<bool> RunOnComputation(
    HloComputation* computation, const se::GpuComputeCapability& gpu_version) {
  GemmFusionVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}


}  // namespace

bool ShouldTritonHandleGEMM(HloDotInstruction& dot,
                            const se::GpuComputeCapability& gpu_version) {
  std::vector<HloInstruction*> fusion_inputs;
  HloComputation::Builder builder("disposable");
  return CreateDotFusion(dot, gpu_version, builder, fusion_inputs,
                         /*fusion_output_ptr=*/nullptr)
      ->CanFuse();
}

absl::StatusOr<bool> GemmFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto cuda_compute_capability =
      std::get_if<se::CudaComputeCapability>(&gpu_version_);
  if (!cuda_compute_capability || !cuda_compute_capability->IsAtLeastAmpere()) {
    return absl::FailedPreconditionError(
        "Triton support is only enabled for Ampere GPUs and up.");
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result,
                        RunOnComputation(computation, gpu_version_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla

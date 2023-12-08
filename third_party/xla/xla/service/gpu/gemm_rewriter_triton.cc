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

#include "xla/service/gpu/gemm_rewriter_triton.h"

#include <array>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
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
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace xla {
namespace gpu {

namespace {

template <class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

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

// This represents a path in a graph which helps in separating different uses of
// HLOs in fusions.
//
// For example let's say that we can reach an HLO in 2 ways:
// 1. dot->operand(0)->operand(1)->operand(2);
// 2. dot->operand(0)->operand(2)->operand(0);
// Then the corresponding paths will be:
// 1. "0,1,2"
// 2. "0,2,0"
// dot->users()[0]->users()[0]->operand(1) would be represented like this:
// "-1,-1,1"
class GraphPath final {
 public:
  static inline constexpr int64_t kUserIndex = -1;
  GraphPath() = default;
  GraphPath GetPathOfOperand(int64_t operand_index) const {
    CHECK_GE(operand_index, 0);
    GraphPath path = *this;
    path.path_.push_back(operand_index);
    return path;
  }

  GraphPath GetPathOfUser() const {
    GraphPath path = *this;
    path.path_.push_back(kUserIndex);
    return path;
  }
  std::string ToString() const { return absl::StrJoin(path_, ","); }

  bool operator==(const GraphPath& other) const { return path_ == other.path_; }

  template <typename H>
  friend H AbslHashValue(H h, const GraphPath& path) {
    return H::combine(std::move(h), path.path_);
  }

 private:
  std::vector<int64_t> path_;
};

struct FusionQueueItem {
  FusionQueueItem(GraphPath path, const HloInstruction& hlo,
                  DimensionOrder dim_order)
      : path(path), hlo(hlo), dim_order(dim_order) {}

  std::string ToString() const {
    return absl::StrCat(path.ToString(), ":", hlo.ToString(), ":",
                        dim_order.ToString());
  }

  const GraphPath path;
  const HloInstruction& hlo;
  const DimensionOrder dim_order;
};

struct FusionDecisionAndIterspec {
  bool fuse = false;
  TensorIterationSpec iterspec;
};
using FusionMap = absl::flat_hash_map<GraphPath, FusionDecisionAndIterspec>;
struct FusionMapAndRequirements {
  FusionMap fusion_map;
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
                        HloComputation::Builder& builder  // append
) {
  CHECK_EQ(dot.operand_count(), 2);
  VLOG(3) << "Fusing " << dot.ToString();

  std::array<HloInstruction*, 2> hlo_new_operands = {
      const_cast<HloInstruction*>(&fused_lhs),
      const_cast<HloInstruction*>(&fused_rhs)};
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
FusionMapAndRequirements BuildFusionMapAndRequirementsTowardOperands(
    const GraphPath& root_path, const HloInstruction& root_hlo,
    const DimensionOrder& root_dim_order, const std::optional<int>& max_params,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties, const Requirements& requirements_so_far) {
  CHECK(!max_params.has_value() || max_params.value() >= 1);
  FusionMap fusion_map;
  Requirements combined_requirements = requirements_so_far;
  auto add_to_fusion_map = [&](const FusionQueueItem& item, bool fuse) {
    CHECK(
        fusion_map
            .insert({item.path, {fuse, item.dim_order.ToTensorIterationSpec()}})
            .second);
  };

  // GraphPaths at the fusion edge that can either get fused too or
  // become parameters of the fusion. Used to track the number of parameters.
  absl::flat_hash_set<GraphPath> inputs({root_path});
  std::queue<FusionQueueItem> fusion_queue(
      {FusionQueueItem(root_path, root_hlo, root_dim_order)});
  int64_t num_requeued = 0;
  // BFS
  while (fusion_queue.size() > num_requeued) {
    FusionQueueItem item = fusion_queue.front();
    fusion_queue.pop();

    // Watch the total number of fusion parameters.
    if (max_params.has_value() &&
        inputs.size() + NumAddedParameters(item.hlo) > max_params.value()) {
      // Re-queue: the number of parameters may go down when other instructions
      // are processed.
      fusion_queue.push(item);
      // Prevent infinite loops.
      ++num_requeued;
      continue;
    }
    num_requeued = 0;
    if (item.hlo.opcode() == HloOpcode::kParameter) {
      add_to_fusion_map(item, /*fuse=*/false);
      continue;
    }
    auto opt_result = GetOperandDimOrdersAndCombinedReqsIfProfitable(
        item.hlo, item.dim_order, properties, gpu_version,
        combined_requirements);
    if (!opt_result.has_value()) {
      add_to_fusion_map(item, /*fuse=*/false);
      continue;
    }
    const DimOrderMap operand_dim_orders = std::move(opt_result->dim_orders);
    combined_requirements = std::move(opt_result->requirements);
    inputs.erase(item.path);
    for (int64_t i = 0; i < item.hlo.operand_count(); ++i) {
      GraphPath operand_path = item.path.GetPathOfOperand(i);
      const HloInstruction& operand = *item.hlo.operand(i);
      const DimensionOrder& operand_dim_order = operand_dim_orders.at(&operand);

      FusionQueueItem new_item(operand_path, operand, operand_dim_order);
      VLOG(6) << "Enqueueing " << new_item.ToString();
      inputs.insert(new_item.path);
      fusion_queue.push(new_item);
    }
    add_to_fusion_map(item, /*fuse=*/true);
  }
  // Handle the remaining requeued items.
  while (!fusion_queue.empty()) {
    add_to_fusion_map(fusion_queue.front(), /*fuse=*/false);
    fusion_queue.pop();
  }
  return {fusion_map, combined_requirements};
}

// Builds the nodes for the fusion represented by the fusion map.
HloInstruction& BuildFusionTowardOperands(
    const FusionMap& fusion_map, const GraphPath& path,
    const HloInstruction& hlo,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  auto fusion_map_it = fusion_map.find(path);
  CHECK(fusion_map_it != fusion_map.end());
  FusionDecisionAndIterspec decision_and_iterspec = fusion_map_it->second;

  HloInstruction* new_hlo = nullptr;
  if (decision_and_iterspec.fuse) {
    HloInstruction::InstructionVector new_operands;
    for (int i = 0; i < hlo.operand_count(); ++i) {
      const HloInstruction* operand = hlo.operand(i);
      new_operands.push_back(
          &BuildFusionTowardOperands(fusion_map, path.GetPathOfOperand(i),
                                     *operand, builder, fusion_params));
    }
    new_hlo = builder.AddInstruction(
        hlo.CloneWithNewOperands(hlo.shape(), new_operands));
  } else {
    fusion_params.push_back(const_cast<HloInstruction*>(&hlo));
    new_hlo = builder.AddInstruction(HloInstruction::CreateParameter(
        fusion_params.size() - 1, hlo.shape(),
        absl::StrCat("parameter_", fusion_params.size() - 1)));
  }
  return *new_hlo;
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
    const GraphPath& root_path, const HloInstruction& root_hlo,
    const DimensionOrder& root_dim_order, const std::optional<int>& max_params,
    const se::GpuComputeCapability& gpu_version,
    const HeroProperties& properties, const Requirements& requirements_so_far,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  FusionMapAndRequirements fusion_map_and_reqs =
      BuildFusionMapAndRequirementsTowardOperands(
          root_path, root_hlo, root_dim_order, max_params, gpu_version,
          properties, requirements_so_far);
  HloInstruction& fused_hlo_or_param =
      BuildFusionTowardOperands(fusion_map_and_reqs.fusion_map, root_path,
                                root_hlo, builder, fusion_params);
  return HlosAndRequirements{&root_hlo, &fused_hlo_or_param,
                             fusion_map_and_reqs.requirements};
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
HlosAndRequirements FuseDotOperand(
    const HloInstruction& dot, int operand_index,
    const se::GpuComputeCapability& gpu_version,
    HloComputation::Builder& builder,            // append
    std::vector<HloInstruction*>& fusion_params  // append
) {
  // Direct dot inputs have well defined dimension orders.
  const FusionContext context =
      FusionContext::FromDotOperand(dot, operand_index);
  const HloInstruction& operand = *dot.operand(operand_index);
  return FuseTowardOperands(GraphPath().GetPathOfOperand(operand_index),
                            operand, context.dim_orders().at(&operand),
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
    const GraphPath& hlo_path, const HloInstruction& hlo,
    const HloInstruction& fused_hlo, const DimensionOrder& hlo_dim_order,
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
  GraphPath user_path = hlo_path.GetPathOfUser();
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
            user_path.GetPathOfOperand(i), operand,
            operand_dim_orders.at(&operand),
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
  return FuseTowardUsers(user_path, user, fused_user, user_dim_order,
                         gpu_version, properties, combined_requirements,
                         builder, fusion_params);
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
  return FuseTowardUsers(GraphPath(), dot, fused_dot,
                         context.dim_orders().at(&dot), gpu_version,
                         context.hero_properties(), context.requirements(),
                         builder, fusion_params);
}

// Fuses dot and the compatible and profitable to fuse operations around it
// into a new fusion computation constructed using the builder. fusion_inputs
// get populated with the non-fused instructions that become operands of the
// call to this fusion. fusion_output_ptr (if not nullptr) gets assigned the
// original instruction that has to be replaced by the call to the fusion.
StatusOr<FusionDecision> CreateDotFusion(
    const HloDotInstruction& dot, const se::GpuComputeCapability gpu_version,
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_inputs,
    HloInstruction** fusion_output_ptr) {
  VLOG(5) << dot.ToString();
  if (FusionDecision can_handle = CanTritonHandleGEMM(dot, gpu_version);
      !can_handle) {
    VLOG(3) << can_handle.Explain();
    return can_handle;
  }

  HlosAndRequirements lhs_hlos_and_reqs = FuseDotOperand(
      dot, /*operand_index=*/0, gpu_version, builder, fusion_inputs);
  HlosAndRequirements rhs_hlos_and_reqs = FuseDotOperand(
      dot, /*operand_index=*/1, gpu_version, builder, fusion_inputs);
  HloInstruction& fused_dot = FuseDot(dot, *lhs_hlos_and_reqs.fused_hlo,
                                      *rhs_hlos_and_reqs.fused_hlo, builder);
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

  if (dot.GetModule()->config().debug_options().xla_gpu_triton_gemm_any()) {
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
    return OkStatus();
  });
  if (!is_pure_matmul) {
    return FusionDecision{};
  }

  return "No profitable operations to fuse.";
}

// Extracts into fused computations parts of HLO graph including dot()
// operations that can target the triton GEMM emitter.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(
      const se::GpuComputeCapability& gpu_version)
      : gpu_version_(gpu_version) {}
  // Checks that a dot() should be targeting the triton GEMM emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  Status HandleDot(HloInstruction* dot) override {
    CHECK_EQ(dot->opcode(), HloOpcode::kDot);

    std::string fusion_name = absl::StrCat("triton_gemm_", dot->name());
    HloComputation::Builder builder(absl::StrCat(fusion_name, "_computation"));
    std::vector<HloInstruction*> fusion_inputs;
    HloInstruction* fusion_output = nullptr;
    TF_ASSIGN_OR_RETURN(
        const FusionDecision should_fuse,
        CreateDotFusion(*Cast<HloDotInstruction>(dot), gpu_version_, builder,
                        fusion_inputs, &fusion_output));
    if (builder.last_added_instruction() == nullptr) {
      return OkStatus();
    }
    // If a GEMM requiring padding for cuBLAS is encountered here this
    // happened because earlier ShouldTritonHandleGEMM() accepted it and padding
    // was skipped. Accept it ignoring profitability checks.
    if (!CublasRequiresPadding(*Cast<HloDotInstruction>(dot), gpu_version_) &&
        !should_fuse) {
      return OkStatus();
    }

    HloComputation* computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction* dot_fusion =
        dot->parent()->AddInstruction(HloInstruction::CreateFusion(
            computation->root_instruction()->shape(),
            HloInstruction::FusionKind::kCustom, fusion_inputs, computation));
    dot_fusion->GetModule()->SetAndUniquifyInstrName(dot_fusion, fusion_name);

    TF_ASSIGN_OR_RETURN(auto backend_config,
                        dot_fusion->backend_config<FusionBackendConfig>());
    backend_config.set_kind(std::string(kTritonGemmFusionKind));
    TF_RETURN_IF_ERROR(dot_fusion->set_backend_config(backend_config));

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
    return OkStatus();
  }

 private:
  se::GpuComputeCapability gpu_version_;
};

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                const se::GpuComputeCapability& gpu_version) {
  GemmRewriterTritonVisitor visitor(gpu_version);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // namespace

FusionDecision CanTritonHandleGEMM(
    const HloDotInstruction& dot, const se::GpuComputeCapability& gpu_version) {
  if (!tsl::tensor_float_32_execution_enabled() ||
      absl::c_any_of(dot.precision_config().operand_precision(),
                     [](int x) { return x != PrecisionConfig::DEFAULT; })) {
    return "Non-default precision.";
  }

  auto supported_output_type = [&](const PrimitiveType t) {
    switch (t) {
      case F16:
      case F32:
        return true;
      case BF16:
        return std::visit(
            Overload{[](const se::CudaComputeCapability& cc) {
                       return cc.IsAtLeast(
                           stream_executor::CudaComputeCapability::AMPERE);
                     },
                     [](const se::RocmComputeCapability&) {
                       return true;  // TODO check rocm support!
                     }},
            gpu_version);
      default:
        return false;
    }
  };

  // TODO(b/266862493): Support more output types.
  if (!supported_output_type(dot.shape().element_type())) {
    return "Unsupported output data type.";
  }

  if (!IsTritonSupportedDataType(dot.operand(0)->shape().element_type(),
                                 gpu_version) ||
      !IsTritonSupportedDataType(dot.operand(1)->shape().element_type(),
                                 gpu_version)) {
    return "Unsupported input data type.";
  }

  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  // TODO(b/269580541): support multiple batch dimensions.
  if (dim_numbers.lhs_batch_dimensions().size() > 1) {
    return "Multiple batch dimensions.";
  }

  // Cases where lhs or rhs have no non-contracting dims are not handled.
  if (dim_numbers.lhs_batch_dimensions().size() +
              dim_numbers.lhs_contracting_dimensions().size() ==
          dot.operand(0)->shape().rank() ||
      dim_numbers.rhs_batch_dimensions().size() +
              dim_numbers.rhs_contracting_dimensions().size() ==
          dot.operand(1)->shape().rank()) {
    return "No non-contracting dimensions.";
  }

  for (int operand_number = 0; operand_number <= 1; ++operand_number) {
    // This pass relies on dot decomposer which ensures that all non-contracting
    // dimensions are merged into one. Using NonContractingDimensionIndex is
    // sufficient.
    const int64_t nc_size =
        dot.operand(operand_number)
            ->shape()
            .dimensions(NonContractingDimensionIndex(dot, operand_number));
    if (nc_size <= 1) {
      return "Trivial non-contracting dimensions.";
    }
  }

  return FusionDecision{};
}

bool ShouldTritonHandleGEMM(HloDotInstruction& dot,
                            const se::GpuComputeCapability& gpu_version) {
  std::vector<HloInstruction*> fusion_inputs;
  HloComputation::Builder builder("disposable");
  return CreateDotFusion(dot, gpu_version, builder, fusion_inputs,
                         /*fusion_output_ptr=*/nullptr)
      ->CanFuse();
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
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

/*Copyright 2022 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0(the "License");
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
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_wrapper.h"
#include "xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/sharding_propagation.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace spmd {
namespace {

using DimMap = StableMap</*tensor dim*/ int, /* mesh dim*/ int>;
using MeshDims = absl::Span<const int64_t>;

struct Enumeration {
  MeshDims mesh_dims;
  int64_t i;
  int64_t j;
};

// Contains base functionality common to both DotHandler and ConvHandler.
class HandlerBase {
 protected:
  HandlerBase(std::unique_ptr<StrategyGroup>& strategy_group,
              StrategyMap& strategy_map, const HloInstruction* ins,
              const int64_t instruction_id,
              const HloInstructionSequence& instruction_sequence,
              const HloCostAnalysis& hlo_cost_analysis,
              const ClusterEnvironment& cluster_env,
              const InstructionBatchDimMap& batch_map,
              const AutoShardingOption& option, const CallGraph& call_graph)
      : strategy_group_(strategy_group),
        strategy_map_(strategy_map),
        ins_(ins),
        instruction_id_(instruction_id),
        instruction_sequence_(instruction_sequence),
        hlo_cost_analysis_(hlo_cost_analysis),
        cluster_env_(cluster_env),
        batch_map_(batch_map),
        option_(option),
        call_graph_(call_graph),
        device_mesh_(cluster_env.device_mesh_),
        device_mesh_1d_(cluster_env.device_mesh_1d_),
        lhs_(ins->operand(0)),
        rhs_(ins->operand(1)) {}

  virtual ~HandlerBase() = default;

  inline Array<int64_t> GetDeviceMesh() {
    return is_generating_strategies_1d_ ? device_mesh_1d_ : device_mesh_;
  }

  void AppendNewStrategy(const std::string& name,
                         const HloSharding& output_spec,
                         absl::Span<const HloSharding> input_specs,
                         double compute_cost, double communication_cost);

  HloSharding CreateInputSpec(const HloInstruction* ins, const DimMap& dim_map,
                              const Array<int64_t>& device_mesh) const {
    if (dim_map.empty()) return HloSharding::Replicate();
    std::vector<int64_t> tensor_dims, mesh_dims;
    for (const auto& [tensor_dim, mesh_dim] : dim_map) {
      tensor_dims.push_back(tensor_dim);
      mesh_dims.push_back(mesh_dim);
    }
    return Tile(ins->shape(), tensor_dims, mesh_dims, device_mesh);
  }

  // Given lhs and rhs dim maps, infers a sharding for the output by relying on
  // the sharding_propagation pass.
  void MaybeAppend(
      const std::string& name, const DimMap& lhs_dim_map,
      const DimMap& rhs_dim_map,
      const std::optional<DimMap>& expected_output_dim_map,
      const Array<int64_t>& device_mesh, double compute_cost = 0,
      const std::optional<std::function<double(const HloSharding&)>>&
          communication_cost_fn = std::nullopt);

  // Given lhs and rhs dim maps, infers a sharding for the output by relying on
  // the sharding_propagation pass.
  void MaybeAppendInternal(
      const std::string& name, const DimMap& lhs_dim_map,
      const DimMap& rhs_dim_map,
      const std::optional<DimMap>& expected_output_dim_map,
      const Array<int64_t>& device_mesh, double compute_cost = 0,
      const std::optional<std::function<double(const HloSharding&)>>&
          communication_cost_fn = std::nullopt);

  // Given an existing (non-allreduce) sharding candidate, generate a
  // corresponding candidate by additionally sharding (if possible) the passed
  // in operand, such that, the generated candidate can trigger all-gather
  // windowed einsum during partitioning.
  virtual void AppendAllGatherWindowedEinsumStrategyForOperand(
      int operand_num, const std::string& name, const DimMap& lhs_dim_map,
      const DimMap& rhs_dim_map, const DimMap& output_dim_map,
      const Array<int64_t>& device_mesh, double compute_cost) {}

  // Given an existing (allreduce) sharding candidate, generate a corresponding
  // candidate by additionally sharding (if possible) the dot/conv output, such
  // that, the generated candidate can trigger reduce-scatter windowed einsum
  // during partitioning.
  virtual void AppendReduceScatterWindowedEinsumStrategy(
      const std::string& name, const DimMap& lhs_dim_map,
      const DimMap& rhs_dim_map, const DimMap& output_dim_map,
      const Array<int64_t>& device_mesh, double compute_cost) {}

  std::optional<HloSharding> GetShardingFromUser(const HloSharding& lhs_spec,
                                                 const HloSharding& rhs_spec);

  // Enumerates combinations of the given mesh + tensor dimensions.
  void Enumerate(std::function<void(const Enumeration&)> split_func,
                 size_t num_outer_dims = 2, size_t num_inner_dims = 2,
                 bool half = false) {
    absl::Span<const int64_t> mesh_shape = GetDeviceMesh().dimensions();
    for (int64_t dim0 = 0; dim0 < mesh_shape.size(); ++dim0) {
      for (int64_t dim1 = 0; dim1 < mesh_shape.size(); ++dim1) {
        if (dim0 == dim1) continue;
        for (int64_t i = 0; i < num_outer_dims; ++i) {
          for (int64_t j = half ? i + 1 : 0; j < num_inner_dims; ++j) {
            split_func({{dim0, dim1}, i, j});
          }
        }
      }
    }
  }

  // Given a set of tensor dims, and a set of mesh dims, enumerates all mappings
  // where a subset of all tensor dims is mapped to a subset of mesh dims, such
  // that each tensor dim is mapped to at most mesh dim, and no two tensor dims
  // are mapped to the same mesh dim.
  // TODO(b/356201483): We might need to generalize this to also allow cases
  // where a tensor dim can be mapped to multiple mesh dims.
  void EnumerateGeneral(std::function<void(const DimMap&)> split_func,
                        int tensor_rank, int current_tensor_dim,
                        const absl::flat_hash_set<int>& unassigned_mesh_dims,
                        const DimMap& current_dim_map) {
    if (current_tensor_dim == tensor_rank) {
      split_func(current_dim_map);
      return;
    }
    // current_tensor_dim is unsharded
    EnumerateGeneral(split_func, tensor_rank, current_tensor_dim + 1,
                     unassigned_mesh_dims, current_dim_map);
    // current_tensor_dim is sharded across one of the remaining mesh dims
    for (int mesh_dim : unassigned_mesh_dims) {
      DimMap updated_dim_map = current_dim_map;
      updated_dim_map[current_tensor_dim] = mesh_dim;
      absl::flat_hash_set<int> updated_unassigned_mesh_dims =
          unassigned_mesh_dims;
      updated_unassigned_mesh_dims.erase(
          updated_unassigned_mesh_dims.find(mesh_dim));
      EnumerateGeneral(split_func, tensor_rank, current_tensor_dim + 1,
                       updated_unassigned_mesh_dims, updated_dim_map);
    }
  }

  // Enumerates *half* of the combinations (if inner & outer dims are the same).
  void EnumerateHalf(std::function<void(const Enumeration&)> split_func,
                     size_t num_outer_dims = 2, size_t num_inner_dims = 2) {
    Enumerate(split_func, num_outer_dims, num_inner_dims, true);
  }

  // Sorts strategies in the increasing order of their memory costs. Anecdotal
  // experience suggests that such a sorted list of strategies works better
  void SortStrategies();

  std::unique_ptr<StrategyGroup>& strategy_group_;
  StrategyMap& strategy_map_;
  const HloInstruction* ins_;
  const int64_t instruction_id_;
  const HloInstructionSequence& instruction_sequence_;
  const HloCostAnalysis& hlo_cost_analysis_;
  const ClusterEnvironment& cluster_env_;
  const InstructionBatchDimMap& batch_map_;
  const AutoShardingOption& option_;
  const CallGraph& call_graph_;

  const Array<int64_t>& device_mesh_;
  const Array<int64_t>& device_mesh_1d_;
  const HloInstruction* lhs_;
  const HloInstruction* rhs_;

  bool is_generating_strategies_1d_;
};

class DotHandler : public HandlerBase {
 public:
  DotHandler(std::unique_ptr<StrategyGroup>& strategy_group,
             StrategyMap& strategy_map, const HloDotInstruction* ins,
             int64_t instruction_id,
             const HloInstructionSequence& instruction_sequence,
             const HloCostAnalysis& hlo_cost_analysis,
             const ClusterEnvironment& cluster_env,
             const InstructionBatchDimMap& batch_map,
             const AutoShardingOption& option, const CallGraph& call_graph);

  DotHandler(
      std::unique_ptr<StrategyGroup>& strategy_group, StrategyMap& strategy_map,
      const HloConvolutionInstruction* ins, int64_t instruction_id,
      const HloInstructionSequence& instruction_sequence,
      const HloCostAnalysis& hlo_cost_analysis,
      const dot_as_convolution_util::DotConvolutionDimsInfo& conv_as_dot_dims,
      const ClusterEnvironment& cluster_env,
      const InstructionBatchDimMap& batch_map, const AutoShardingOption& option,
      const CallGraph& call_graph);

  ~DotHandler() override = default;

  std::string GenerateNameForDotSharding(const DimMap& output_dim_map,
                                         const DimMap& lhs_dim_map);

  void GenerateDotShardingStrategiesFromOutputSharding(
      const DimMap& output_dim_map);

  void AppendAllGatherWindowedEinsumStrategyForOperand(
      int operand_num, const std::string& name, const DimMap& lhs_dim_map,
      const DimMap& rhs_dim_map, const DimMap& output_dim_map,
      const Array<int64_t>& device_mesh, double compute_cost) override;

  void AppendReduceScatterWindowedEinsumStrategy(
      const std::string& name, const DimMap& lhs_dim_map,
      const DimMap& rhs_dim_map, const DimMap& output_dim_map,
      const Array<int64_t>& device_mesh, double compute_cost) override;

  absl::Status RegisterStrategies();

  // Dimension information
  bool is_dot_;
  int64_t space_base_dim_;
  tsl::protobuf::RepeatedField<int64_t> lhs_space_dims_, rhs_space_dims_;
  tsl::protobuf::RepeatedField<int64_t> out_lhs_space_dims_,
      out_rhs_space_dims_;
  tsl::protobuf::RepeatedField<int64_t> lhs_con_dims_;
  tsl::protobuf::RepeatedField<int64_t> rhs_con_dims_;
  tsl::protobuf::RepeatedField<int64_t> lhs_batch_dims_;
  tsl::protobuf::RepeatedField<int64_t> rhs_batch_dims_;
  std::vector<int64_t> out_batch_dims_;
};

class ConvHandler : public HandlerBase {
 public:
  ConvHandler(std::unique_ptr<StrategyGroup>& strategy_group,
              StrategyMap& strategy_map, const HloInstruction* ins,
              int64_t instruction_id,
              const HloInstructionSequence& instruction_sequence,
              const HloCostAnalysis& hlo_cost_analysis,
              const ClusterEnvironment& cluster_env,
              const InstructionBatchDimMap& batch_map,
              const AutoShardingOption& option, const CallGraph& call_graph);

  ~ConvHandler() override = default;

  void SplitLhsBatchRhsOutchannel();

  void SplitLhsBatchBothInchannel();

  void SplitRhsOutchannelBothInchannel();

  void Add1DDataParallel();

  void SplitDepthwise(bool forward);

  void GenerateConvolutionShardingStrategiesFromOutputSharding(
      const DimMap& output_dim_map);

  absl::Status RegisterStrategies();

  // Dimension information
  const ConvolutionDimensionNumbers& conv_dnums_;
  int64_t lhs_batch_dim_, lhs_in_channel_dim_;
  int64_t rhs_in_channel_dim_, rhs_out_channel_dim_;
  int64_t out_batch_dim_, out_out_channel_dim_;
};

/************** HandlerBase function definitions **************/

void HandlerBase::AppendNewStrategy(const std::string& name,
                                    const HloSharding& output_spec,
                                    absl::Span<const HloSharding> input_specs,
                                    double compute_cost,
                                    double communication_cost) {
  ReshardingCosts communication_resharding_costs;
  ReshardingCosts memory_resharding_costs;

  for (int i = 0; i < ins_->operand_count(); ++i) {
    const HloInstruction* operand = ins_->operand(i);
    communication_resharding_costs.push_back(CommunicationReshardingCostVector(
        strategy_map_.at(operand).get(), operand->shape(), input_specs[i],
        cluster_env_));
    memory_resharding_costs.push_back(MemoryReshardingCostVector(
        strategy_map_.at(operand).get(), operand->shape(), input_specs[i],
        cluster_env_));
  }

  strategy_group_->strategies.push_back(ShardingStrategy({
      name,
      output_spec,
      compute_cost,
      communication_cost,
      static_cast<double>(
          ByteSizeOfShapeWithSharding(ins_->shape(), output_spec)),
      communication_resharding_costs,
      memory_resharding_costs,
      {input_specs.begin(), input_specs.end()},
  }));
}

// Given lhs and rhs dim maps, infers a sharding for the output by relying on
// the sharding_propagation pass. Given that this is a relatively new change
// (as of 11/2023), we also take an optional expected output dim map as an
// argument, to verify that sharding propagation in fact infers the sharding
// we expect (and to crash if it doesn't).
// TODO(b/309638633) As we build more confidence in this, we should remove
// this expected_output_dim_map argument and fully rely on sharding
// propagation.
void HandlerBase::MaybeAppendInternal(
    const std::string& name, const DimMap& lhs_dim_map,
    const DimMap& rhs_dim_map,
    const std::optional<DimMap>& expected_output_dim_map,
    const Array<int64_t>& device_mesh, double compute_cost,
    const std::optional<std::function<double(const HloSharding&)>>&
        communication_cost_fn) {
  HloSharding lhs_spec = CreateInputSpec(lhs_, lhs_dim_map, device_mesh);
  HloSharding rhs_spec = CreateInputSpec(rhs_, rhs_dim_map, device_mesh);
  std::optional<HloSharding> output_spec =
      GetShardingFromUser(lhs_spec, rhs_spec);
  if (output_spec.has_value()) {
    if (expected_output_dim_map.has_value()) {
      HloSharding expected_output_spec =
          CreateInputSpec(ins_, *expected_output_dim_map, device_mesh);
      // TODO(b/308687597) Once the bug is resolved, we ideally either want
      // have a CHECK statement verifying that the sharding inferred by
      // sharding propagation is in fact what we expect, or we trust sharding
      // propagation's results without the check. b/308687597 currently
      // prevents us from doing so. AutoShardingTest.LargeSize in
      // //third_party/tensorflow/compiler/xla/hlo/experimental/auto_sharding:auto_sharding_test
      // currently fails due to the issue.
      if (ins_->opcode() == HloOpcode::kDot &&
          *output_spec != expected_output_spec) {
        output_spec = expected_output_spec;
        LOG(ERROR)
            << "The sharding inferred by sharding propagation in this case "
               "does not match the expected sharding for the dot "
               "instruction. This may be related to b/308687597. Given this "
               "mismatch, we continue with the expected sharding";
      }
    }
  } else {
    CHECK(expected_output_dim_map.has_value());
    output_spec = CreateInputSpec(ins_, *expected_output_dim_map, device_mesh);
    LOG(WARNING)
        << "Sharding propagation could not infer output sharding for:\n  "
        << ins_->ToString() << "\n  LHS Spec: " << lhs_spec
        << "\n  RHS Spec: " << rhs_spec << "\n  Output sharding name: " << name;
  }

  double communication_cost = 0;
  if (communication_cost_fn.has_value()) {
    communication_cost = communication_cost_fn.value()(*output_spec);
  }
  AppendNewStrategy(name, *output_spec, {lhs_spec, rhs_spec}, compute_cost,
                    communication_cost);
}

void HandlerBase::MaybeAppend(
    const std::string& name, const DimMap& lhs_dim_map,
    const DimMap& rhs_dim_map,
    const std::optional<DimMap>& expected_output_dim_map,
    const Array<int64_t>& device_mesh, double compute_cost,
    const std::optional<std::function<double(const HloSharding&)>>&
        communication_cost_fn) {
  MaybeAppendInternal(name, lhs_dim_map, rhs_dim_map, expected_output_dim_map,
                      device_mesh, compute_cost, communication_cost_fn);
  if (!option_.generate_windowed_einsum_strategies ||
      !expected_output_dim_map.has_value()) {
    return;
  }
  if (absl::StrContains(name, "allreduce")) {
    CHECK(communication_cost_fn.has_value());
    AppendReduceScatterWindowedEinsumStrategy(name, lhs_dim_map, rhs_dim_map,
                                              *expected_output_dim_map,
                                              device_mesh, compute_cost);
  } else {
    CHECK(!communication_cost_fn.has_value());
    AppendAllGatherWindowedEinsumStrategyForOperand(
        0, name, lhs_dim_map, rhs_dim_map, *expected_output_dim_map,
        device_mesh, compute_cost);
    AppendAllGatherWindowedEinsumStrategyForOperand(
        1, name, lhs_dim_map, rhs_dim_map, *expected_output_dim_map,
        device_mesh, compute_cost);
  }
}

std::optional<HloSharding> HandlerBase::GetShardingFromUser(
    const HloSharding& lhs_spec, const HloSharding& rhs_spec) {
  std::unique_ptr<HloInstruction> ins_clone = ins_->Clone();
  std::unique_ptr<HloInstruction> lhs_clone = lhs_->Clone();
  std::unique_ptr<HloInstruction> rhs_clone = rhs_->Clone();
  ins_clone->clear_sharding();
  lhs_clone->set_sharding(lhs_spec);
  rhs_clone->set_sharding(rhs_spec);
  CHECK_OK(ins_clone->ReplaceOperandWith(0, lhs_clone.get()));
  CHECK_OK(ins_clone->ReplaceOperandWith(1, rhs_clone.get()));
  if (ins_->opcode() == HloOpcode::kConvolution) {
    xla::InferConvolutionShardingFromOperands(
        ins_clone.get(), call_graph_, 10,
        /* may_combine_partial_sharding */ true, /* is_spmd */ true);
  } else {
    xla::InferDotShardingFromOperands(
        ins_clone.get(), call_graph_,
        dot_as_convolution_util::ParseDotGeneralFromDot(ins_clone.get()),
        /* may_combine_partial_sharding/ */ true, /* is_spmd */ true);
  }
  if (!ins_clone->has_sharding()) {
    return std::nullopt;
  }
  return ins_clone->sharding();
}

void HandlerBase::SortStrategies() {
  absl::c_sort(strategy_group_->strategies,
               [](const ShardingStrategy& s1, const ShardingStrategy& s2) {
                 if (s1.memory_cost == s2.memory_cost) {
                   return s1.name < s2.name;
                 } else {
                   return s1.memory_cost < s2.memory_cost;
                 }
               });
}

/************** DotHandler function definitions **************/

DotHandler::DotHandler(std::unique_ptr<StrategyGroup>& strategy_group,
                       StrategyMap& strategy_map, const HloDotInstruction* ins,
                       const int64_t instruction_id,
                       const HloInstructionSequence& instruction_sequence,
                       const HloCostAnalysis& hlo_cost_analysis,
                       const ClusterEnvironment& cluster_env,
                       const InstructionBatchDimMap& batch_map,
                       const AutoShardingOption& option,
                       const CallGraph& call_graph)
    : HandlerBase(strategy_group, strategy_map, ins, instruction_id,
                  instruction_sequence, hlo_cost_analysis, cluster_env,
                  batch_map, option, call_graph),
      is_dot_(true),
      space_base_dim_(ins->dot_dimension_numbers().lhs_batch_dimensions_size()),
      lhs_con_dims_(ins->dot_dimension_numbers().lhs_contracting_dimensions()),
      rhs_con_dims_(ins->dot_dimension_numbers().rhs_contracting_dimensions()),
      lhs_batch_dims_(ins->dot_dimension_numbers().lhs_batch_dimensions()),
      rhs_batch_dims_(ins->dot_dimension_numbers().rhs_batch_dimensions()),
      out_batch_dims_(
          ins->dot_dimension_numbers().rhs_batch_dimensions().size()) {
  std::tie(lhs_space_dims_, rhs_space_dims_) =
      GetSpaceDims(lhs_->shape(), rhs_->shape(), ins->dot_dimension_numbers());
  for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
    out_lhs_space_dims_.Add(space_base_dim_ + i);
  }
  for (int64_t i = 0; i < rhs_space_dims_.size(); ++i) {
    out_rhs_space_dims_.Add(space_base_dim_ + lhs_space_dims_.size() + i);
  }
  std::iota(out_batch_dims_.begin(), out_batch_dims_.end(), 0);
  CHECK_EQ(lhs_con_dims_.size(), rhs_con_dims_.size());
  CHECK_EQ(lhs_batch_dims_.size(), rhs_batch_dims_.size());
}

DotHandler::DotHandler(
    std::unique_ptr<StrategyGroup>& strategy_group, StrategyMap& strategy_map,
    const HloConvolutionInstruction* ins, const int64_t instruction_id,
    const HloInstructionSequence& instruction_sequence,
    const HloCostAnalysis& hlo_cost_analysis,
    const dot_as_convolution_util::DotConvolutionDimsInfo& conv_as_dot_dims,
    const ClusterEnvironment& cluster_env,
    const InstructionBatchDimMap& batch_map, const AutoShardingOption& option,
    const CallGraph& call_graph)
    : HandlerBase(strategy_group, strategy_map, ins, instruction_id,
                  instruction_sequence, hlo_cost_analysis, cluster_env,
                  batch_map, option, call_graph),
      is_dot_(false),
      space_base_dim_(-1) {
  CHECK(conv_as_dot_dims.conv_spatial_dims.empty());

  for (auto dim_idx : conv_as_dot_dims.batch_dims) {
    if (dim_idx.lhs >= 0) lhs_batch_dims_.Add(dim_idx.lhs);
    if (dim_idx.rhs >= 0) rhs_batch_dims_.Add(dim_idx.rhs);
    if (dim_idx.output >= 0) out_batch_dims_.push_back(dim_idx.output);
  }

  for (auto dim_idx : conv_as_dot_dims.contracting_dims) {
    if (dim_idx.lhs >= 0) lhs_con_dims_.Add(dim_idx.lhs);
    if (dim_idx.rhs >= 0) rhs_con_dims_.Add(dim_idx.rhs);
  }

  for (auto dim_idx : conv_as_dot_dims.lhs_non_contracting_dims) {
    if (dim_idx.lhs >= 0) lhs_space_dims_.Add(dim_idx.lhs);
    if (dim_idx.output >= 0) out_lhs_space_dims_.Add(dim_idx.output);
  }

  for (auto dim_idx : conv_as_dot_dims.rhs_non_contracting_dims) {
    if (dim_idx.rhs >= 0) rhs_space_dims_.Add(dim_idx.rhs);
    if (dim_idx.output >= 0) out_rhs_space_dims_.Add(dim_idx.output);
  }
}

std::string DotHandler::GenerateNameForDotSharding(const DimMap& output_dim_map,
                                                   const DimMap& lhs_dim_map) {
  std::string name;

  auto append_shardings_for_dims = [&name](absl::Span<const int64_t> out_dims,
                                           const DimMap& dim_map,
                                           absl::string_view identifier) {
    for (size_t i = 0; i < out_dims.size(); ++i) {
      int output_batch_dim = out_dims[i];
      int mesh_dim = -1;
      auto it = dim_map.find(output_batch_dim);
      if (it != dim_map.end() && it->second >= 0) {
        mesh_dim = it->second;
      }
      absl::StrAppend(&name, identifier, mesh_dim);
    }
  };

  // Output batch dims
  append_shardings_for_dims(out_batch_dims_, output_dim_map,
                            /*identifier=*/"b");
  // LHS space dims
  append_shardings_for_dims(out_lhs_space_dims_, output_dim_map,
                            /*identifier=*/"ls");
  // RHS space dims
  append_shardings_for_dims(out_rhs_space_dims_, output_dim_map,
                            /*identifier=*/"rs");
  // Contraction dims
  append_shardings_for_dims(lhs_con_dims_, lhs_dim_map,
                            /*identifier=*/"r");

  bool contraction_dim_sharded = false;
  for (size_t i = 0; i < lhs_con_dims_.size(); ++i) {
    if (auto it = lhs_dim_map.find(lhs_con_dims_[i]);
        it != lhs_dim_map.end() && it->second >= 0) {
      contraction_dim_sharded =
          contraction_dim_sharded || (GetDeviceMesh().dim(it->second) > 1);
    }
  }

  if (contraction_dim_sharded) {
    absl::StrAppend(&name, "|allreduce");
  }
  return name;
}

bool IsFullyReplicatedSharding(const DimMap& dim_map,
                               const Array<int64_t>& device_mesh) {
  if (dim_map.empty()) {
    return true;
  }
  for (const auto& [_, mesh_dim] : dim_map) {
    if (device_mesh.dim(mesh_dim) > 1) {
      return false;
    }
  }
  return true;
}

bool IsFullyReplicatedStrategy(const DimMap& output_dim_map,
                               const DimMap& lhs_dim_map,
                               const DimMap& rhs_dim_map,
                               const Array<int64_t>& device_mesh) {
  return IsFullyReplicatedSharding(output_dim_map, device_mesh) &&
         IsFullyReplicatedSharding(lhs_dim_map, device_mesh) &&
         IsFullyReplicatedSharding(rhs_dim_map, device_mesh);
}

bool IsFullySharded(const DimMap& dim_map, int num_mesh_dims) {
  return dim_map.size() >= num_mesh_dims;
}

void DotHandler::GenerateDotShardingStrategiesFromOutputSharding(
    const DimMap& output_dim_map) {
  DimMap lhs_dim_map, rhs_dim_map;
  absl::flat_hash_set<int> used_mesh_dims;

  // Propagate shardings for batch dimensions
  for (size_t i = 0; i < out_batch_dims_.size(); ++i) {
    int output_batch_dim = out_batch_dims_[i];
    int lhs_batch_dim = lhs_batch_dims_[i];
    int rhs_batch_dim = rhs_batch_dims_[i];
    auto it = output_dim_map.find(output_batch_dim);
    if (it != output_dim_map.end() && it->second >= 0) {
      int mesh_dim = it->second;
      used_mesh_dims.insert(mesh_dim);
      lhs_dim_map[lhs_batch_dim] = mesh_dim;
      rhs_dim_map[rhs_batch_dim] = mesh_dim;
    }
  }

  // Propagate shardings for spatial dimensions
  // - LHS space dims
  for (size_t i = 0; i < lhs_space_dims_.size(); ++i) {
    int lhs_space_dim = lhs_space_dims_[i];
    int output_space_dim = out_lhs_space_dims_[i];
    auto it = output_dim_map.find(output_space_dim);
    if (it != output_dim_map.end() && it->second >= 0) {
      int mesh_dim = it->second;
      used_mesh_dims.insert(mesh_dim);
      lhs_dim_map[lhs_space_dim] = mesh_dim;
    }
  }

  // - RHS space dims
  for (size_t i = 0; i < rhs_space_dims_.size(); ++i) {
    int rhs_space_dim = rhs_space_dims_[i];
    int output_space_dim = out_rhs_space_dims_[i];
    auto it = output_dim_map.find(output_space_dim);
    if (it != output_dim_map.end() && it->second >= 0) {
      int mesh_dim = it->second;
      used_mesh_dims.insert(mesh_dim);
      rhs_dim_map[rhs_space_dim] = mesh_dim;
    }
  }

  // Skip fully the replicated strategy here as we add that outside of
  // HandleDot in auto_sharding_strategy.
  // TODO(b/348372403): Consolidate the generation of all dot strategies
  // (including replicated strategies) in one place.
  if (!IsFullyReplicatedStrategy(output_dim_map, lhs_dim_map, rhs_dim_map,
                                 GetDeviceMesh()) &&
      // This second condition is added to ensure parity with the older strategy
      // generation code. Removing it will only increase the search space.
      IsFullySharded(output_dim_map, GetDeviceMesh().num_dimensions())) {
    MaybeAppend(GenerateNameForDotSharding(output_dim_map, lhs_dim_map),
                lhs_dim_map, rhs_dim_map, output_dim_map, GetDeviceMesh());
  }

  // Generate shardings for contraction dimensions
  if (used_mesh_dims.size() == GetDeviceMesh().num_dimensions()) {
    return;
  }

  absl::flat_hash_set<int> unused_mesh_dims;
  for (size_t i = 0; i < GetDeviceMesh().num_dimensions(); ++i) {
    if (!used_mesh_dims.contains(i) && GetDeviceMesh().dim(i) > 1) {
      unused_mesh_dims.insert(i);
    }
  }

  if (unused_mesh_dims.empty()) {
    return;
  }

  std::vector<int> reduction_dims(lhs_con_dims_.size());
  std::iota(reduction_dims.begin(), reduction_dims.end(), 0);

  auto split_func = [&](const DimMap& reduction_dim_map) {
    if (reduction_dim_map.empty()) {
      return;
    }

    DimMap lhs_dim_map_with_contractions = lhs_dim_map;
    DimMap rhs_dim_map_with_contractions = rhs_dim_map;
    for (const auto& [reducton_dim_index, mesh_dim] : reduction_dim_map) {
      lhs_dim_map_with_contractions
          [lhs_con_dims_[reduction_dims[reducton_dim_index]]] = mesh_dim;
      rhs_dim_map_with_contractions
          [rhs_con_dims_[reduction_dims[reducton_dim_index]]] = mesh_dim;
    }
    // Skip fully the replicated strategy here as we add that outside of
    // HandleDot in auto_sharding_strategy.
    // TODO: Fix the above
    if (IsFullyReplicatedStrategy(output_dim_map, lhs_dim_map_with_contractions,
                                  rhs_dim_map_with_contractions,
                                  GetDeviceMesh())) {
      return;
    }
    CHECK(!lhs_dim_map_with_contractions.empty());
    CHECK(!rhs_dim_map_with_contractions.empty());

    auto communication_cost_fn = [&](const HloSharding& output_sharding) {
      double memory_cost =
          ByteSizeOfShapeWithSharding(ins_->shape(), output_sharding);
      double total_cost = 0;
      for (const auto& [_, mesh_dim] : reduction_dim_map) {
        total_cost += cluster_env_.AllReduceCost(memory_cost, mesh_dim);
      }
      return total_cost;
    };

    MaybeAppend(GenerateNameForDotSharding(output_dim_map,
                                           lhs_dim_map_with_contractions),
                lhs_dim_map_with_contractions, rhs_dim_map_with_contractions,
                output_dim_map, GetDeviceMesh(),
                /*compute_cost=*/0, communication_cost_fn);
  };

  EnumerateGeneral(split_func, reduction_dims.size(),
                   /*current_tensor_dim=*/0, unused_mesh_dims,
                   /*current_dim_map=*/{});
}

void DotHandler::AppendAllGatherWindowedEinsumStrategyForOperand(
    int operand_num, const std::string& name, const DimMap& lhs_dim_map,
    const DimMap& rhs_dim_map, const DimMap& output_dim_map,
    const Array<int64_t>& device_mesh, double compute_cost) {
  const HloInstruction* operand = ins_->operand(operand_num);
  const DimMap& operand_dim_map = operand_num == 0 ? lhs_dim_map : rhs_dim_map;
  absl::flat_hash_set<int64_t> sharded_tensor_dims;
  absl::flat_hash_set<int64_t> used_mesh_dims;
  for (const auto [tensor_dim, mesh_dim] : operand_dim_map) {
    if (device_mesh.dim(mesh_dim) == 1) {
      continue;
    }
    sharded_tensor_dims.insert(tensor_dim);
    used_mesh_dims.insert(mesh_dim);
  }
  if (used_mesh_dims.size() == GetDeviceMesh().num_dimensions() ||
      sharded_tensor_dims.size() == operand->shape().rank()) {
    return;
  }

  for (int64_t tensor_dim = 0; tensor_dim < operand->shape().rank();
       ++tensor_dim) {
    if (sharded_tensor_dims.contains(tensor_dim)) {
      continue;
    }
    for (int64_t mesh_dim = 0; mesh_dim < GetDeviceMesh().num_dimensions();
         ++mesh_dim) {
      if (used_mesh_dims.contains(mesh_dim) ||
          (device_mesh.dim(mesh_dim) == 1)) {
        continue;
      }
      DimMap further_sharded_dim_map = operand_dim_map;
      further_sharded_dim_map[tensor_dim] = mesh_dim;

      auto updated_communication_cost_fn =
          [](const HloSharding& output_sharding) -> double {
        // TODO(331684721): Model costs for windowed einsum
        return 100.0;
      };

      std::string updated_name =
          absl::StrCat(absl::StrFormat("WindowedEinsum @ {%d,%d,%d}",
                                       operand_num, tensor_dim, mesh_dim),
                       name);
      MaybeAppendInternal(
          updated_name,
          operand_num == 0 ? further_sharded_dim_map : lhs_dim_map,
          operand_num == 1 ? further_sharded_dim_map : rhs_dim_map,
          output_dim_map, device_mesh, compute_cost,
          updated_communication_cost_fn);
    }
  }
}

void DotHandler::AppendReduceScatterWindowedEinsumStrategy(
    const std::string& name, const DimMap& lhs_dim_map,
    const DimMap& rhs_dim_map, const DimMap& output_dim_map,
    const Array<int64_t>& device_mesh, double compute_cost) {
  absl::flat_hash_set<int64_t> sharded_tensor_dims;
  absl::flat_hash_set<int64_t> used_mesh_dims;
  for (const auto [tensor_dim, mesh_dim] : output_dim_map) {
    if (device_mesh.dim(mesh_dim) == 1) {
      continue;
    }
    sharded_tensor_dims.insert(tensor_dim);
    used_mesh_dims.insert(mesh_dim);
  }
  if (used_mesh_dims.size() == GetDeviceMesh().num_dimensions() ||
      sharded_tensor_dims.size() == ins_->shape().rank()) {
    return;
  }

  for (int64_t tensor_dim = 0; tensor_dim < ins_->shape().rank();
       ++tensor_dim) {
    if (sharded_tensor_dims.contains(tensor_dim)) {
      continue;
    }
    for (int64_t mesh_dim = 0; mesh_dim < GetDeviceMesh().num_dimensions();
         ++mesh_dim) {
      if (used_mesh_dims.contains(mesh_dim) ||
          (device_mesh.dim(mesh_dim) == 1)) {
        continue;
      }
      DimMap further_sharded_dim_map = output_dim_map;
      further_sharded_dim_map[tensor_dim] = mesh_dim;

      auto updated_communication_cost_fn =
          [](const HloSharding& output_sharding) -> double {
        // TODO(331684721): Model costs for windowed einsum
        return 100.0;
      };

      std::string updated_name = absl::StrCat(
          absl::StrFormat("WindowedEinsum @ {%d,%d}", tensor_dim, mesh_dim),
          name);
      MaybeAppendInternal(updated_name, lhs_dim_map, rhs_dim_map,
                          further_sharded_dim_map, device_mesh, compute_cost,
                          updated_communication_cost_fn);
    }
  }
}

absl::Status DotHandler::RegisterStrategies() {
  auto generate_strategies_for_mesh = [&]() {
    absl::flat_hash_set<int> all_mesh_dims;
    for (int i = 0; i < GetDeviceMesh().num_dimensions(); ++i) {
      all_mesh_dims.insert(i);
    }
    EnumerateGeneral(
        /*split_func=*/
        [&](const DimMap& output_dim_map) {
          GenerateDotShardingStrategiesFromOutputSharding(output_dim_map);
        },
        ins_->shape().rank(), /*current_tensor_dim=*/0, all_mesh_dims,
        /*current_dim_map=*/{});
  };

  // TODO(b/356201483): Remove this when EnumerateGeneral can handle a tensor
  // dims sharded across multiple mesh dims
  is_generating_strategies_1d_ = false;
  generate_strategies_for_mesh();
  is_generating_strategies_1d_ = true;
  generate_strategies_for_mesh();

  SortStrategies();
  return absl::OkStatus();
}

/************** ConvHandler function definitions **************/

ConvHandler::ConvHandler(std::unique_ptr<StrategyGroup>& strategy_group,
                         StrategyMap& strategy_map, const HloInstruction* ins,
                         const int64_t instruction_id,
                         const HloInstructionSequence& instruction_sequence,
                         const HloCostAnalysis& hlo_cost_analysis,
                         const ClusterEnvironment& cluster_env,
                         const InstructionBatchDimMap& batch_map,
                         const AutoShardingOption& option,
                         const CallGraph& call_graph)
    : HandlerBase(strategy_group, strategy_map, ins, instruction_id,
                  instruction_sequence, hlo_cost_analysis, cluster_env,
                  batch_map, option, call_graph),
      conv_dnums_(ins->convolution_dimension_numbers()) {
  lhs_batch_dim_ = conv_dnums_.input_batch_dimension();
  lhs_in_channel_dim_ = conv_dnums_.input_feature_dimension();
  rhs_in_channel_dim_ = conv_dnums_.kernel_input_feature_dimension();
  rhs_out_channel_dim_ = conv_dnums_.kernel_output_feature_dimension();
  out_batch_dim_ = conv_dnums_.output_batch_dimension();
  out_out_channel_dim_ = conv_dnums_.output_feature_dimension();
}

void ConvHandler::GenerateConvolutionShardingStrategiesFromOutputSharding(
    const DimMap& output_dim_map) {
  DimMap lhs_dim_map;
  DimMap rhs_dim_map;
  absl::flat_hash_set<int> used_mesh_dims;
  std::string name;

  // Propagate batch dim sharding
  auto it = output_dim_map.find(out_batch_dim_);
  if (it != output_dim_map.end() && GetDeviceMesh().dim(it->second) > 1) {
    int mesh_dim = it->second;
    lhs_dim_map[lhs_batch_dim_] = mesh_dim;
    used_mesh_dims.insert(mesh_dim);
    absl::StrAppend(&name, "b", mesh_dim);
  } else {
    absl::StrAppend(&name, "b-1");
  }

  // Propagate out channel dim sharding
  it = output_dim_map.find(out_out_channel_dim_);
  if (it != output_dim_map.end() && GetDeviceMesh().dim(it->second) > 1) {
    int mesh_dim = it->second;
    lhs_dim_map[rhs_out_channel_dim_] = mesh_dim;
    used_mesh_dims.insert(mesh_dim);
    absl::StrAppend(&name, "oc", mesh_dim);
  } else {
    absl::StrAppend(&name, "oc-1");
  }

  MaybeAppend(name, lhs_dim_map, rhs_dim_map, output_dim_map, GetDeviceMesh());

  // Generate shardings for contraction dimensions
  if (used_mesh_dims.size() == GetDeviceMesh().num_dimensions()) {
    return;
  }

  absl::flat_hash_set<int> unused_mesh_dims;
  for (size_t i = 0; i < GetDeviceMesh().num_dimensions(); ++i) {
    if (!used_mesh_dims.contains(i) && GetDeviceMesh().dim(i) > 1) {
      unused_mesh_dims.insert(i);
    }
  }

  if (unused_mesh_dims.empty()) {
    return;
  }

  for (int64_t mesh_dim : unused_mesh_dims) {
    DimMap lhs_dim_map_with_contractions = lhs_dim_map;
    DimMap rhs_dim_map_with_contractions = rhs_dim_map;

    lhs_dim_map_with_contractions[lhs_in_channel_dim_] = mesh_dim;
    rhs_dim_map_with_contractions[rhs_in_channel_dim_] = mesh_dim;
    absl::StrAppend(&name, "ic", mesh_dim, "@allreduce");

    auto communication_cost_fn = [&](const HloSharding& output_sharding) {
      return cluster_env_.AllReduceCost(
          ByteSizeOfShapeWithSharding(ins_->shape(), output_sharding),
          mesh_dim);
    };

    MaybeAppend(name, lhs_dim_map_with_contractions,
                rhs_dim_map_with_contractions, output_dim_map, GetDeviceMesh(),
                /*compute_cost=*/0, communication_cost_fn);
  }
}

absl::Status ConvHandler::RegisterStrategies() {
  // For 1D sharding
  if ((ins_->feature_group_count() ==
           lhs_->shape().dimensions(lhs_in_channel_dim_) &&
       ins_->feature_group_count() ==
           rhs_->shape().dimensions(rhs_out_channel_dim_))) {
    // for depthwise conv
    // SS = SS x S
    // Split batch dim and channel dim
    SplitDepthwise(true);
  } else if ((ins_->batch_group_count() ==
                  lhs_->shape().dimensions(lhs_batch_dim_) &&
              ins_->batch_group_count() ==
                  rhs_->shape().dimensions(rhs_out_channel_dim_))) {
    // for depthwise conv filter_backward
    // SS = SS x S
    // Split batch dim and channel dim
    SplitDepthwise(false);
  }

  auto generate_strategies_for_mesh = [&]() {
    absl::flat_hash_set<int> all_mesh_dims;
    for (int i = 0; i < GetDeviceMesh().num_dimensions(); ++i) {
      all_mesh_dims.insert(i);
    }
    EnumerateGeneral(
        [&](const DimMap& output_dim_map) {
          GenerateConvolutionShardingStrategiesFromOutputSharding(
              output_dim_map);
        },
        2, /*current_tensor_dim=*/0, all_mesh_dims,
        /*current_dim_map=*/{});
  };

  // TODO(b/356201483): Remove this when EnumerateGeneral can handle a tensor
  // dims sharded across multiple mesh dims
  is_generating_strategies_1d_ = false;
  generate_strategies_for_mesh();
  is_generating_strategies_1d_ = true;
  generate_strategies_for_mesh();

  // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
  // and only keep the data parallel strategies.
  if (option_.force_batch_dim_to_mesh_dim >= 0 &&
      batch_map_.contains(GetBatchDimMapKey(ins_))) {
    TF_RETURN_IF_ERROR(FilterStrategy(ins_, ins_->shape(), strategy_group_,
                                      cluster_env_, batch_map_, option_));
  }

  SortStrategies();
  return absl::OkStatus();
}

void ConvHandler::SplitDepthwise(bool forward) {
  std::function<void(const DimMap&)> split_func =
      [&](const DimMap& output_dim_map) {
        int out_batch_mesh_dim = -1;
        int out_out_channel_mesh_dim = -1;
        if (auto it = output_dim_map.find(out_batch_dim_);
            it != output_dim_map.end()) {
          out_batch_mesh_dim = it->second;
        }
        if (auto it = output_dim_map.find(out_out_channel_dim_);
            it != output_dim_map.end()) {
          out_out_channel_mesh_dim = it->second;
        }
        if (out_batch_mesh_dim == -1 || out_out_channel_mesh_dim == -1) {
          return;
        }

        DimMap lhs_dim_map, rhs_dim_map;
        lhs_dim_map[lhs_batch_dim_] =
            forward ? out_batch_mesh_dim : out_out_channel_mesh_dim;
        lhs_dim_map[lhs_in_channel_dim_] =
            forward ? out_out_channel_mesh_dim : out_batch_mesh_dim;

        rhs_dim_map[rhs_out_channel_dim_] = out_out_channel_mesh_dim;

        MaybeAppend(absl::StrCat("b", out_batch_mesh_dim, "oc",
                                 out_out_channel_mesh_dim, "@depthwise"),
                    lhs_dim_map, rhs_dim_map, output_dim_map, GetDeviceMesh());
      };
  absl::flat_hash_set<int> all_mesh_dims;
  for (int i = 0; i < GetDeviceMesh().num_dimensions(); ++i) {
    all_mesh_dims.insert(i);
  }
  EnumerateGeneral(split_func, 2, /*current_tensor_dim=*/0, all_mesh_dims,
                   /*current_dim_map=*/{});
}

}  // namespace

// Register strategies for dot instructions.
absl::Status HandleDot(std::unique_ptr<StrategyGroup>& strategy_group,
                       StrategyGroups& strategy_groups,
                       StrategyMap& strategy_map, const HloInstruction* ins,
                       size_t instruction_id,
                       const HloInstructionSequence& instruction_sequence,
                       const HloCostAnalysis& hlo_cost_analysis,
                       const ClusterEnvironment& cluster_env,
                       const InstructionBatchDimMap& batch_map,
                       const AutoShardingOption& option,
                       const CallGraph& call_graph) {
  strategy_group = CreateLeafStrategyGroup(instruction_id, ins, strategy_map,
                                           strategy_groups);

  DotHandler handler(strategy_group, strategy_map, Cast<HloDotInstruction>(ins),
                     instruction_id, instruction_sequence, hlo_cost_analysis,
                     cluster_env, batch_map, option, call_graph);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return absl::OkStatus();
}

// Register strategies for convolution instructions.
absl::Status HandleConv(std::unique_ptr<StrategyGroup>& strategy_group,
                        StrategyGroups& strategy_groups,
                        StrategyMap& strategy_map, const HloInstruction* ins,
                        size_t instruction_id,
                        const HloInstructionSequence& instruction_sequence,
                        const HloCostAnalysis& hlo_cost_analysis,
                        const ClusterEnvironment& cluster_env,
                        const InstructionBatchDimMap& batch_map,
                        const AutoShardingOption& option,
                        const CallGraph& call_graph) {
  strategy_group = CreateLeafStrategyGroup(instruction_id, ins, strategy_map,
                                           strategy_groups);

  auto conv_as_dot_dims =
      dot_as_convolution_util::ParseConvolutionDimsInfo(ins);
  if (conv_as_dot_dims.conv_spatial_dims.empty()) {
    DotHandler handler(
        strategy_group, strategy_map, Cast<HloConvolutionInstruction>(ins),
        instruction_id, instruction_sequence, hlo_cost_analysis,
        conv_as_dot_dims, cluster_env, batch_map, option, call_graph);
    TF_RETURN_IF_ERROR(handler.RegisterStrategies());

  } else {
    ConvHandler handler(strategy_group, strategy_map, ins, instruction_id,
                        instruction_sequence, hlo_cost_analysis, cluster_env,
                        batch_map, option, call_graph);
    TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  }
  return absl::OkStatus();
}

}  // namespace spmd
}  // namespace xla

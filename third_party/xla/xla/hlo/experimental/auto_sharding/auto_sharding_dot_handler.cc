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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_solver_option.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/status.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace spmd {

using DimMap = StableHashMap</*tensor dim*/ int, /* mesh dim*/ int>;
using MeshDims = absl::Span<const int64_t>;

struct Enumeration {
  MeshDims mesh_dims;
  int64_t i;
  int64_t j;
};

// Contains base functionality common to both DotHandler and ConvHandler.
class HandlerBase {
 protected:
  HandlerBase(std::unique_ptr<StrategyVector>& strategies,
              StrategyMap& strategy_map, const HloInstruction* ins,
              const ClusterEnvironment& cluster_env,
              const InstructionBatchDimMap& batch_map,
              const AutoShardingSolverOption& solver_option,
              const CallGraph& call_graph)
      : strategies_(strategies),
        strategy_map_(strategy_map),
        ins_(ins),
        cluster_env_(cluster_env),
        batch_map_(batch_map),
        solver_option_(solver_option),
        call_graph_(call_graph),
        device_mesh_(cluster_env.device_mesh_),
        device_mesh_1d_(cluster_env.device_mesh_1d_),
        lhs_(ins->operand(0)),
        rhs_(ins->operand(1)) {}

  void AppendNewStrategy(const std::string& name,
                         const HloSharding& output_spec,
                         absl::Span<const HloSharding> input_specs,
                         double compute_cost, double communication_cost) {
    std::vector<std::vector<double>> resharding_costs;

    for (int i = 0; i < ins_->operand_count(); ++i) {
      const HloInstruction* operand = ins_->operand(i);
      resharding_costs.push_back(
          ReshardingCostVector(strategy_map_.at(operand).get(),
                               operand->shape(), input_specs[i], cluster_env_));
    }

    strategies_->leaf_vector.push_back(ShardingStrategy({
        name,
        output_spec,
        compute_cost,
        communication_cost,
        GetBytes(ins_->shape()) / output_spec.NumTiles(),
        resharding_costs,
        {input_specs.begin(), input_specs.end()},
    }));
  }

  bool CheckDims(const HloInstruction* ins, const DimMap& dim_map) const {
    for (const auto& [tensor_dim, mesh_dim] : dim_map) {
      auto shape_dim = ins->shape().dimensions().at(tensor_dim);
      auto device_mesh_dim = device_mesh_.dim(mesh_dim);
      if (shape_dim < device_mesh_dim) return false;
      if (solver_option_.only_allow_divisible_intermediate &&
          !IsDivisible(shape_dim, device_mesh_dim))
        return false;
    }
    return true;
  }

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

  HloSharding CreateInputSpecUsingShardingPropagation(
      int operand_index, const HloSharding& output_spec) const {
    std::optional<HloSharding> operand_sharding =
        GetInputSharding(ins_, ins_->operand(operand_index), operand_index,
                         output_spec, call_graph_, cluster_env_.NumDevices());
    CHECK(operand_sharding.has_value());
    return operand_sharding.value();
  }

  void MaybeAppend(const std::string& name, const HloSharding& output_spec,
                   const DimMap& lhs_dim_map, const DimMap& rhs_dim_map,
                   const Array<int64_t>& device_mesh, double compute_cost = 0,
                   double communication_cost = 0,
                   bool use_sharding_propagation = true) {
    if (!CheckDims(lhs_, lhs_dim_map) || !CheckDims(rhs_, rhs_dim_map)) return;
    HloSharding lhs_spec =
        use_sharding_propagation
            ? CreateInputSpecUsingShardingPropagation(0, output_spec)
            : CreateInputSpec(lhs_, lhs_dim_map, device_mesh);
    HloSharding rhs_spec =
        use_sharding_propagation
            ? CreateInputSpecUsingShardingPropagation(1, output_spec)
            : CreateInputSpec(rhs_, rhs_dim_map, device_mesh);
    AppendNewStrategy(name, output_spec, {lhs_spec, rhs_spec}, compute_cost,
                      communication_cost);
  }

  // Enumerates combinations of the given mesh + tensor dimensions.
  void Enumerate(std::function<void(const Enumeration&)> split_func,
                 size_t num_outer_dims = 2, size_t num_inner_dims = 2,
                 bool half = false) {
    auto mesh_shape = device_mesh_.dimensions();
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

  // Enumerates *half* of the combinations (if inner & outer dims are the same).
  void EnumerateHalf(std::function<void(const Enumeration&)> split_func,
                     size_t num_outer_dims = 2, size_t num_inner_dims = 2) {
    Enumerate(split_func, num_outer_dims, num_inner_dims, true);
  }

  std::unique_ptr<StrategyVector>& strategies_;
  StrategyMap& strategy_map_;
  const HloInstruction* ins_;
  const ClusterEnvironment& cluster_env_;
  const InstructionBatchDimMap& batch_map_;
  const AutoShardingSolverOption& solver_option_;
  const CallGraph& call_graph_;

  const Array<int64_t>& device_mesh_;
  const Array<int64_t>& device_mesh_1d_;
  const HloInstruction* lhs_;
  const HloInstruction* rhs_;
};

class DotHandler : public HandlerBase {
 public:
  DotHandler(std::unique_ptr<StrategyVector>& strategies,
             StrategyMap& strategy_map, const HloInstruction* ins,
             const ClusterEnvironment& cluster_env,
             const InstructionBatchDimMap& batch_map,
             const AutoShardingSolverOption& solver_option,
             const CallGraph& call_graph)
      : HandlerBase(strategies, strategy_map, ins, cluster_env, batch_map,
                    solver_option, call_graph),
        space_base_dim_(
            ins->dot_dimension_numbers().lhs_batch_dimensions_size()),
        lhs_con_dims_(
            ins->dot_dimension_numbers().lhs_contracting_dimensions()),
        rhs_con_dims_(
            ins->dot_dimension_numbers().rhs_contracting_dimensions()),
        lhs_batch_dims_(ins->dot_dimension_numbers().lhs_batch_dimensions()),
        rhs_batch_dims_(ins->dot_dimension_numbers().rhs_batch_dimensions()) {
    std::tie(lhs_space_dims_, rhs_space_dims_) = GetSpaceDims(
        lhs_->shape(), rhs_->shape(), ins->dot_dimension_numbers());
    CHECK_EQ(lhs_con_dims_.size(), rhs_con_dims_.size());
    CHECK_EQ(lhs_batch_dims_.size(), rhs_batch_dims_.size());
  }

  void SplitLhsSpaceRhsSpace() {
    auto func = [this](const Enumeration& e) {
      const DimMap lhs_dim_map = {{lhs_space_dims_[e.i], e.mesh_dims[0]}};
      const DimMap rhs_dim_map = {{rhs_space_dims_[e.j], e.mesh_dims[1]}};
      std::string name = absl::StrFormat("SS = SR x RS @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      HloSharding output_spec =
          Tile(ins_->shape(),
               {space_base_dim_ + e.i,
                space_base_dim_ + static_cast<int64_t>(lhs_space_dims_.size()) +
                    e.j},
               e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_);
    };
    Enumerate(func, lhs_space_dims_.size(), rhs_space_dims_.size());
  }

  void SplitLhsSpaceOnly() {
    auto func = [this](const Enumeration& e) {
      const DimMap lhs_dim_map = {{lhs_space_dims_[e.i], e.mesh_dims[0]},
                                  {lhs_space_dims_[e.j], e.mesh_dims[1]}};
      std::string name = absl::StrFormat("SSR = SSR x RR @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      HloSharding output_spec =
          Tile(ins_->shape(), {space_base_dim_ + e.i, space_base_dim_ + e.j},
               e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, {}, device_mesh_);
    };
    EnumerateHalf(func, lhs_space_dims_.size(), lhs_space_dims_.size());
  }

  void SplitRhsSpaceOnly() {
    auto func = [this](const Enumeration& e) {
      const DimMap rhs_dim_map = {{rhs_space_dims_[e.i], e.mesh_dims[0]},
                                  {rhs_space_dims_[e.j], e.mesh_dims[1]}};
      std::string name = absl::StrFormat("RSS = RR x RSS @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      HloSharding output_spec = Tile(
          ins_->shape(),
          {space_base_dim_ + static_cast<int64_t>(lhs_space_dims_.size()) + e.i,
           space_base_dim_ + static_cast<int64_t>(lhs_space_dims_.size()) +
               e.j},
          e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, {}, rhs_dim_map, device_mesh_);
    };
    EnumerateHalf(func, rhs_space_dims_.size(), rhs_space_dims_.size());
  }

  void SplitLhsSpaceBothContract() {
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%s} (allreduce @ %d)",
                          absl::StrJoin(e.mesh_dims, ","), e.mesh_dims[1]);
      const DimMap lhs_dim_map = {{lhs_space_dims_[e.i], e.mesh_dims[0]},
                                  {lhs_con_dims_[e.j], e.mesh_dims[1]}};
      const DimMap rhs_dim_map = {{rhs_con_dims_[e.j], e.mesh_dims[1]}};
      HloSharding output_spec = Tile(ins_->shape(), {space_base_dim_ + e.i},
                                     {e.mesh_dims[0]}, device_mesh_);
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, e.mesh_dims[1]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  communication_cost, /* use_sharding_propagation */ false);
    };
    Enumerate(func, lhs_space_dims_.size(), lhs_con_dims_.size());
  }

  void SplitRhsSpaceBothContract() {
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1) return;
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%s} (allreduce @ %d)",
                          absl::StrJoin(e.mesh_dims, ","), e.mesh_dims[0]);
      const DimMap rhs_dim_map = {{rhs_space_dims_[e.i], e.mesh_dims[1]},
                                  {rhs_con_dims_[e.j], e.mesh_dims[0]}};
      const DimMap lhs_dim_map = {{lhs_con_dims_[e.j], e.mesh_dims[0]}};
      HloSharding output_spec =
          Tile(ins_->shape(),
               {space_base_dim_ + static_cast<int64_t>(lhs_space_dims_.size()) +
                e.i},
               {e.mesh_dims[1]}, device_mesh_);
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, e.mesh_dims[0]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  communication_cost, /* use_sharding_propagation */ false);
    };
    Enumerate(func, rhs_space_dims_.size(), lhs_con_dims_.size());
  }

  void SplitOneBatchDim() {
    if (absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) != 1) {
      return;
    }
    auto func = [this](const Enumeration& e) {
      const DimMap lhs_dim_map = {{lhs_batch_dims_[e.i], e.j}};
      const DimMap rhs_dim_map = {{rhs_batch_dims_[e.i], e.j}};
      std::string name = absl::StrFormat("Sb_%d = Sb x Sb @ {%d}", e.i, e.j);
      HloSharding output_spec = Tile(ins_->shape(), {e.i}, {e.j}, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_);
    };
    Enumerate(func, lhs_batch_dims_.size(), device_mesh_.num_dimensions());
  }

  void SplitTwoBatchDims() {
    if (lhs_batch_dims_.size() != 2) return;
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      const DimMap lhs_dim_map = {{lhs_batch_dims_[0], e.mesh_dims[0]},
                                  {lhs_batch_dims_[1], e.mesh_dims[1]}};
      const DimMap rhs_dim_map = {{rhs_batch_dims_[0], e.mesh_dims[0]},
                                  {rhs_batch_dims_[1], e.mesh_dims[1]}};
      std::string name = absl::StrFormat("Sb = Sb x Sb @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      HloSharding output_spec =
          Tile(ins_->shape(), {0, 1}, e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_);
    };
    EnumerateHalf(func, lhs_batch_dims_.size(), lhs_batch_dims_.size());
  }

  void SplitBatchDimLhsSpace() {
    if (lhs_batch_dims_.empty()) return;
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      std::string name = absl::StrFormat("SbSi = SbSi x SbR @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      const DimMap lhs_dim_map = {{lhs_space_dims_[e.i], e.mesh_dims[1]},
                                  {lhs_batch_dims_[e.j], e.mesh_dims[0]}};
      const DimMap rhs_dim_map = {{rhs_batch_dims_[e.j], e.mesh_dims[0]}};
      HloSharding output_spec =
          Tile(ins_->shape(), {e.j, space_base_dim_ + e.i}, e.mesh_dims,
               device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_);
    };
    Enumerate(func, lhs_space_dims_.size(), lhs_batch_dims_.size());
  }

  void SplitBatchDimRhsSpace() {
    if (lhs_batch_dims_.empty()) return;
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      std::string name = absl::StrFormat("SbSj = SbR x SbSj @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      const DimMap rhs_dim_map = {{rhs_space_dims_[e.i], e.mesh_dims[1]},
                                  {rhs_batch_dims_[e.j], e.mesh_dims[0]}};
      const DimMap lhs_dim_map = {{lhs_batch_dims_[e.j], e.mesh_dims[0]}};
      HloSharding output_spec =
          Tile(ins_->shape(),
               {e.j, space_base_dim_ +
                         static_cast<int64_t>(lhs_space_dims_.size()) + e.i},
               e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_);
    };
    Enumerate(func, rhs_space_dims_.size(), lhs_batch_dims_.size());
  }

  void SplitBatchDimBothContract() {
    if (lhs_batch_dims_.empty()) return;
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      std::string name =
          absl::StrFormat("SbR = SbSk x SbSk @ {%s} (allreduce @ %d}",
                          absl::StrJoin(e.mesh_dims, ","), e.mesh_dims[1]);
      const DimMap lhs_dim_map = {{lhs_con_dims_[e.i], e.mesh_dims[1]},
                                  {lhs_batch_dims_[e.j], e.mesh_dims[0]}};
      const DimMap rhs_dim_map = {{rhs_batch_dims_[e.j], e.mesh_dims[0]}};
      HloSharding output_spec =
          Tile(ins_->shape(), {e.j}, {e.mesh_dims[0]}, device_mesh_);
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, e.mesh_dims[1]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  communication_cost, /* use_sharding_propagation */ false);
    };
    Enumerate(func, lhs_con_dims_.size(), lhs_batch_dims_.size());
  }

  void SplitBothContractTwoDims() {
    if (lhs_con_dims_.size() < 2 || rhs_con_dims_.size() < 2) return;
    auto func = [this](const Enumeration& e) {
      // Applies when there are more than one contracting dimension.
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      std::string name = absl::StrFormat(
          "RR = SS x SS @ {%s} (allreduce @ {%s}}",
          absl::StrJoin(e.mesh_dims, ","), absl::StrJoin(e.mesh_dims, ", "));
      const DimMap lhs_dim_map = {{lhs_con_dims_[e.i], e.mesh_dims[0]},
                                  {lhs_con_dims_[e.j], e.mesh_dims[1]}};
      const DimMap rhs_dim_map = {{rhs_con_dims_[e.i], e.mesh_dims[0]},
                                  {rhs_con_dims_[e.j], e.mesh_dims[1]}};
      HloSharding output_spec = HloSharding::Replicate();
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost = cluster_env_.AllReduceCost(
          memory_cost, e.mesh_dims[0], e.mesh_dims[1]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  communication_cost, /* use_sharding_propagation */ false);
    };
    EnumerateHalf(func, lhs_con_dims_.size(), lhs_con_dims_.size());
  }

  void RecomputeSplitBothContract() {
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      std::string name = absl::StrFormat("RR = RS x SR @ {%d} (allreduce @ %d)",
                                         e.mesh_dims[0], e.mesh_dims[0]);
      const DimMap lhs_dim_map = {{lhs_con_dims_[e.i], e.mesh_dims[0]}};
      const DimMap rhs_dim_map = {{rhs_con_dims_[e.i], e.mesh_dims[0]}};
      HloSharding output_spec = HloSharding::Replicate();
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double compute_cost = cluster_env_.DotCost(lhs_->shape(), rhs_->shape());
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, e.mesh_dims[0]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_,
                  compute_cost, communication_cost,
                  /* use_sharding_propagation */ false);
    };
    Enumerate(func, lhs_con_dims_.size(), 1);
  }

  void Add1DDataParallel() {
    if (device_mesh_.dim(0) > 1 &&
        absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) > 1) {
      int mesh_dim = 0;
      int64_t num_devices = device_mesh_1d_.dim(mesh_dim);

      // Si = Si x R @ 0
      for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
        const DimMap lhs_dim_map = {{lhs_space_dims_[i], mesh_dim}};
        if (lhs_->shape().dimensions(lhs_space_dims_[i]) < num_devices) {
          continue;
        }
        if (solver_option_.only_allow_divisible_intermediate &&
            !IsDivisible(lhs_->shape().dimensions(lhs_space_dims_[i]),
                         num_devices)) {
          continue;
        }
        std::string name = absl::StrFormat("Si = Si x R @ %d", mesh_dim);
        HloSharding output_spec = Tile(ins_->shape(), {space_base_dim_ + i},
                                       {mesh_dim}, device_mesh_1d_);
        MaybeAppend(name, output_spec, lhs_dim_map, {}, device_mesh_1d_);
      }

      // R = Sk x Sk @ (allreduce @ 0)
      for (int64_t i = 0; i < lhs_con_dims_.size(); ++i) {
        const DimMap lhs_dim_map = {{lhs_con_dims_[i], mesh_dim}};
        const DimMap rhs_dim_map = {{rhs_con_dims_[i], mesh_dim}};
        if (lhs_->shape().dimensions(lhs_con_dims_[i]) < num_devices) {
          continue;
        }
        if (solver_option_.only_allow_divisible_intermediate &&
            !IsDivisible(lhs_->shape().dimensions(lhs_con_dims_[i]),
                         num_devices)) {
          continue;
        }
        std::string name = absl::StrFormat("R = Sk x Sk @ %d (allreduce @ %d)",
                                           mesh_dim, mesh_dim);
        HloSharding output_spec = HloSharding::Replicate();
        double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
        double communication_cost =
            cluster_env_.AllReduceCost(memory_cost, mesh_dim);
        MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map,
                    device_mesh_1d_, 0, communication_cost,
                    /* use_sharding_propagation */ false);
      }
    }
  }

  void Add1DBatchSplit() {
    if (device_mesh_.dim(0) > 1 &&
        absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) > 1) {
      int mesh_dim = 0;
      for (int64_t i = 0; i < lhs_batch_dims_.size(); ++i) {
        const DimMap lhs_dim_map = {{lhs_batch_dims_[i], mesh_dim}};
        const DimMap rhs_dim_map = {{rhs_batch_dims_[i], mesh_dim}};
        std::string name =
            absl::StrFormat("Sb_%d = Sb x Sb @ {%d} 1d", i, mesh_dim);
        HloSharding output_spec =
            Tile(ins_->shape(), {i}, {mesh_dim}, device_mesh_1d_);
        MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map,
                    device_mesh_1d_);
      }
    }
  }

  Status RegisterStrategies() {
    // SS = SR x RS
    // Split lhs space dim and rhs space dim.
    SplitLhsSpaceRhsSpace();

    // SSR = SSR x RR
    // Split lhs space dims only if it has more than 1 space dims.
    if (lhs_space_dims_.size() > 1) {
      SplitLhsSpaceOnly();
    }
    // RSS = RR x RSS
    // Split rhs space dims only if it has more than 1 space dims.
    if (rhs_space_dims_.size() > 1) {
      SplitRhsSpaceOnly();
    }

    // SR = SS x SR
    // Split lhs space dim and both contracting dims.
    SplitLhsSpaceBothContract();

    // RS = RS x SS
    // Split rhs space dim and both contracting dims.
    SplitRhsSpaceBothContract();

    // RR = SS x SS
    // Split two contracting dims on lhs and rhs.
    SplitBothContractTwoDims();

    // RR = RS x SR
    // This is a special case where we allow spliting only one dim in the
    // multi-dimensional mesh case. This allows some recomputation
    // (e.g., the dense layer in the LM_head of BERT).
    RecomputeSplitBothContract();

    // Add 1d data parallel in multi-dimensional mesh
    if (solver_option_.allow_mixed_mesh_shape) {
      Add1DDataParallel();
    }

    if (solver_option_.batch_matmul_always_split_batch &&
        !lhs_batch_dims_.empty() &&
        cluster_env_.non_zero_mesh_dims_.size() > 1) {
      // If there is a batch dim and the device mesh is multi-dimensional,
      // always split on batch dim. Clear all old strategies.
      strategies_->leaf_vector.clear();
    }

    // Sb = Sb x Sb
    // Split one batch dim. Only used for 1d mesh
    SplitOneBatchDim();

    // SbSi = SbSi x SbR
    // Split batch dim and lhs space dim
    SplitBatchDimLhsSpace();

    // SbSj = SbR x SbSj
    // Split batch dim and rhs space dim
    SplitBatchDimRhsSpace();

    // SbSj = SbR x SbSj
    // Split batch dim and contracting dim
    SplitBatchDimBothContract();

    if (solver_option_.batch_matmul_always_split_batch &&
        lhs_batch_dims_.size() == 2 &&
        absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) > 1) {
      // If there are two batch dims, always split on these two dims.
      // Clear all old strategies.
      strategies_->leaf_vector.clear();
    }

    // Sb = Sb x Sb
    // Split batch dims.
    SplitTwoBatchDims();

    if (solver_option_.allow_mixed_mesh_shape) {
      Add1DBatchSplit();
    }

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option_.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map_.contains(GetBatchDimMapKey(ins_))) {
      TF_RETURN_IF_ERROR(FilterStrategy(ins_, ins_->shape(), strategies_,
                                        cluster_env_, batch_map_,
                                        solver_option_));
    }

    return OkStatus();
  }

  // Dimension information
  int64_t space_base_dim_;
  tsl::protobuf::RepeatedField<int64_t> lhs_space_dims_, rhs_space_dims_;
  const tsl::protobuf::RepeatedField<int64_t>& lhs_con_dims_;
  const tsl::protobuf::RepeatedField<int64_t>& rhs_con_dims_;
  const tsl::protobuf::RepeatedField<int64_t>& lhs_batch_dims_;
  const tsl::protobuf::RepeatedField<int64_t>& rhs_batch_dims_;
};

// Register strategies for dot instructions.
Status HandleDot(std::unique_ptr<StrategyVector>& strategies,
                 LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                 const HloInstruction* ins, size_t instruction_id,
                 const ClusterEnvironment& cluster_env,
                 const InstructionBatchDimMap& batch_map,
                 const AutoShardingSolverOption& solver_option,
                 const CallGraph& call_graph) {
  strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                        leaf_strategies);

  DotHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                     solver_option, call_graph);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return OkStatus();
}

class ConvHandler : public HandlerBase {
 public:
  ConvHandler(std::unique_ptr<StrategyVector>& strategies,
              StrategyMap& strategy_map, const HloInstruction* ins,
              const ClusterEnvironment& cluster_env,
              const InstructionBatchDimMap& batch_map,
              const AutoShardingSolverOption& solver_option,
              const CallGraph& call_graph)
      : HandlerBase(strategies, strategy_map, ins, cluster_env, batch_map,
                    solver_option, call_graph),
        conv_dnums_(ins->convolution_dimension_numbers()) {
    lhs_batch_dim_ = conv_dnums_.input_batch_dimension();
    lhs_in_channel_dim_ = conv_dnums_.input_feature_dimension();
    rhs_in_channel_dim_ = conv_dnums_.kernel_input_feature_dimension();
    rhs_out_channel_dim_ = conv_dnums_.kernel_output_feature_dimension();
    out_batch_dim_ = conv_dnums_.output_batch_dimension();
    out_out_channel_dim_ = conv_dnums_.output_feature_dimension();
  }

  void SplitLhsBatchRhsOutchannel() {
    auto func = [this](const Enumeration& e) {
      const DimMap lhs_dim_map = {{lhs_batch_dim_, e.mesh_dims[0]}};
      const DimMap rhs_dim_map = {{rhs_out_channel_dim_, e.mesh_dims[1]}};
      std::string name = absl::StrFormat("SS = SR x RS @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      HloSharding output_spec =
          Tile(ins_->shape(), {out_batch_dim_, out_out_channel_dim_},
               e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  0, /* use_sharding_propagation */ false);
    };
    EnumerateHalf(func);
  }

  void SplitLhsBatchBothInchannel() {
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1 ||
          device_mesh_.dim(e.mesh_dims[1]) <= 1)
        return;
      const DimMap lhs_dim_map = {{lhs_batch_dim_, e.mesh_dims[0]},
                                  {lhs_in_channel_dim_, e.mesh_dims[1]}};
      const DimMap rhs_dim_map = {{rhs_in_channel_dim_, e.mesh_dims[1]}};
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%s} (allreduce @ %d)",
                          absl::StrJoin(e.mesh_dims, ","), e.mesh_dims[1]);
      HloSharding output_spec =
          Tile(ins_->shape(), {out_batch_dim_}, {e.mesh_dims[0]}, device_mesh_);
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, e.mesh_dims[1]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  communication_cost, /* use_sharding_propagation */ false);
    };
    EnumerateHalf(func);
  }

  void SplitRhsOutchannelBothInchannel() {
    auto func = [this](const Enumeration& e) {
      if (device_mesh_.dim(e.mesh_dims[0]) <= 1) return;
      const DimMap lhs_dim_map = {{lhs_in_channel_dim_, e.mesh_dims[0]}};
      const DimMap rhs_dim_map = {{rhs_in_channel_dim_, e.mesh_dims[0]},
                                  {rhs_out_channel_dim_, e.mesh_dims[1]}};
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%s} (allreduce @ %d)",
                          absl::StrJoin(e.mesh_dims, ","), e.mesh_dims[0]);
      HloSharding output_spec = Tile(ins_->shape(), {out_out_channel_dim_},
                                     {e.mesh_dims[1]}, device_mesh_);
      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, e.mesh_dims[0]);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  communication_cost, /* use_sharding_propagation */ false);
    };
    EnumerateHalf(func);
  }

  void Add1DDataParallel() {
    if (device_mesh_.dim(0) > 1 &&
        absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) > 1) {
      int mesh_dim = 0;
      int64_t num_devices = device_mesh_1d_.dim(mesh_dim);

      // Si = Si x R @ 0
      if (lhs_->shape().dimensions(lhs_batch_dim_) % num_devices == 0) {
        const DimMap lhs_dim_map = {{lhs_batch_dim_, mesh_dim}};
        std::string name = absl::StrFormat("Si = Si x R @ 0");
        HloSharding output_spec =
            Tile(ins_->shape(), {out_batch_dim_}, {mesh_dim}, device_mesh_1d_);
        MaybeAppend(name, output_spec, lhs_dim_map, {}, device_mesh_1d_, 0, 0,
                    /* use_sharding_propagation */ false);
      }

      // R = Sk x Sk @ (allreduce @ 0)
      if (lhs_->shape().dimensions(lhs_in_channel_dim_) % num_devices == 0 &&
          rhs_->shape().dimensions(rhs_in_channel_dim_) % num_devices == 0) {
        const DimMap lhs_dim_map = {{lhs_in_channel_dim_, mesh_dim}};
        const DimMap rhs_dim_map = {{rhs_in_channel_dim_, mesh_dim}};
        std::string name = absl::StrFormat("R = Sk x Sk @ %d (allreduce @ %d)",
                                           mesh_dim, mesh_dim);
        HloSharding output_spec = HloSharding::Replicate();
        double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
        double communication_cost = cluster_env_.AllReduceCost(memory_cost, 0) +
                                    cluster_env_.AllReduceCost(memory_cost, 1);
        MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map,
                    device_mesh_1d_, 0, communication_cost,
                    /* use_sharding_propagation */ false);
      }
    }
  }

  void SplitDepthwise(bool forward) {
    auto func = [this, forward](const Enumeration& e) {
      const DimMap lhs_dim_map = {
          {lhs_batch_dim_, e.mesh_dims[forward ? 0 : 1]},
          {lhs_in_channel_dim_, e.mesh_dims[forward ? 1 : 0]}};
      const DimMap rhs_dim_map = {{rhs_out_channel_dim_, e.mesh_dims[1]}};
      std::string name = absl::StrFormat("SS = SS x RS @ {%s}",
                                         absl::StrJoin(e.mesh_dims, ","));
      HloSharding output_spec =
          Tile(ins_->shape(), {out_batch_dim_, out_out_channel_dim_},
               e.mesh_dims, device_mesh_);
      MaybeAppend(name, output_spec, lhs_dim_map, rhs_dim_map, device_mesh_, 0,
                  0, /* use_sharding_propagation */ false);
    };
    EnumerateHalf(func);
  }

  Status RegisterStrategies() {
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

    // SS = SR x RS
    // Split lhs batch dim and rhs out_channel dim.
    SplitLhsBatchRhsOutchannel();

    // SR = SS x SR
    // Split lhs batch dim and both in_channel dims.
    SplitLhsBatchBothInchannel();

    // RS = RS x SS
    // Split rhs out_channel dim and both in_channel dims.
    SplitRhsOutchannelBothInchannel();

    // Add 1d data parallel in multi-dimensional mesh
    if (solver_option_.allow_mixed_mesh_shape) {
      Add1DDataParallel();
    }

    // If force_batch_dim_to_mesh_dim is set, filter out invalid strategies
    // and only keep the data parallel strategies.
    if (solver_option_.force_batch_dim_to_mesh_dim >= 0 &&
        batch_map_.contains(GetBatchDimMapKey(ins_))) {
      TF_RETURN_IF_ERROR(FilterStrategy(ins_, ins_->shape(), strategies_,
                                        cluster_env_, batch_map_,
                                        solver_option_));
    }

    return OkStatus();
  }

  // Dimension information
  const ConvolutionDimensionNumbers& conv_dnums_;
  int64_t lhs_batch_dim_, lhs_in_channel_dim_;
  int64_t rhs_in_channel_dim_, rhs_out_channel_dim_;
  int64_t out_batch_dim_, out_out_channel_dim_;
};

// Register strategies for dot instructions.
Status HandleConv(std::unique_ptr<StrategyVector>& strategies,
                  LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                  const HloInstruction* ins, size_t instruction_id,
                  const ClusterEnvironment& cluster_env,
                  const InstructionBatchDimMap& batch_map,
                  const AutoShardingSolverOption& solver_option,
                  const CallGraph& call_graph) {
  strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                        leaf_strategies);

  ConvHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                      solver_option, call_graph);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return OkStatus();
}

}  // namespace spmd
}  // namespace xla

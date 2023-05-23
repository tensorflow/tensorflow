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

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_solver_option.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_strategy.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/cluster_environment.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {
namespace spmd {

void AppendNewStrategy(const HloInstruction* ins, const std::string& name,
                       const HloSharding& output_spec,
                       absl::Span<const HloSharding> input_specs,
                       double compute_cost, double communication_cost,
                       const ClusterEnvironment& cluster_env,
                       const StrategyMap& strategy_map,
                       std::unique_ptr<StrategyVector>& strategies) {
  std::vector<std::vector<double>> resharding_costs;

  for (int i = 0; i < ins->operand_count(); ++i) {
    const HloInstruction* operand = ins->operand(i);
    resharding_costs.push_back(
        ReshardingCostVector(strategy_map.at(operand).get(), operand->shape(),
                             input_specs[i], cluster_env));
  }

  strategies->leaf_vector.push_back(ShardingStrategy({
      name,
      output_spec,
      compute_cost,
      communication_cost,
      GetBytes(ins->shape()) / output_spec.NumTiles(),
      resharding_costs,
      {input_specs.begin(), input_specs.end()},
  }));
}

class DotHandler {
 public:
  DotHandler(std::unique_ptr<StrategyVector>& strategies,
             StrategyMap& strategy_map, const HloInstruction* ins,
             const ClusterEnvironment& cluster_env,
             const InstructionBatchDimMap& batch_map,
             const AutoShardingSolverOption& solver_option)
      : strategies_(strategies),
        strategy_map_(strategy_map),
        ins_(ins),
        cluster_env_(cluster_env),
        batch_map_(batch_map),
        solver_option_(solver_option),
        device_mesh_(cluster_env.device_mesh_),
        device_mesh_1d_(cluster_env.device_mesh_1d_),
        lhs_(ins->operand(0)),
        rhs_(ins->operand(1)),
        dot_dnums_(ins->dot_dimension_numbers()),
        space_base_dim_(dot_dnums_.lhs_batch_dimensions_size()),
        lhs_con_dims_(
            ins->dot_dimension_numbers().lhs_contracting_dimensions()),
        rhs_con_dims_(
            ins->dot_dimension_numbers().rhs_contracting_dimensions()),
        lhs_batch_dims_(ins->dot_dimension_numbers().lhs_batch_dimensions()),
        rhs_batch_dims_(ins->dot_dimension_numbers().rhs_batch_dimensions()) {
    std::tie(lhs_space_dims_, rhs_space_dims_) =
        GetSpaceDims(lhs_->shape(), rhs_->shape(), dot_dnums_);
    CHECK_EQ(lhs_con_dims_.size(), rhs_con_dims_.size());
    CHECK_EQ(lhs_batch_dims_.size(), rhs_batch_dims_.size());
  }

  void SplitLhsSpaceRhsSpace(int mesh_dim0, int mesh_dim1) {
    for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
      for (int64_t j = 0; j < rhs_space_dims_.size(); ++j) {
        if (lhs_->shape().dimensions().at(lhs_space_dims_.at(i)) <
                device_mesh_.dim(mesh_dim0) ||
            rhs_->shape().dimensions().at(rhs_space_dims_.at(j)) <
                device_mesh_.dim(mesh_dim1)) {
          continue;
        }
        if (solver_option_.only_allow_divisible_intermediate &&
            (!IsDivisible(lhs_->shape().dimensions().at(lhs_space_dims_.at(i)),
                          device_mesh_.dim(mesh_dim0)) ||
             !IsDivisible(rhs_->shape().dimensions().at(rhs_space_dims_.at(j)),
                          device_mesh_.dim(mesh_dim1)))) {
          continue;
        }
        std::string name =
            absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1);
        HloSharding output_spec =
            Tile(ins_->shape(),
                 {space_base_dim_ + i,
                  space_base_dim_ +
                      static_cast<int64_t>(lhs_space_dims_.size()) + j},
                 {mesh_dim0, mesh_dim1}, device_mesh_);
        HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_space_dims_[i]},
                                    {mesh_dim0}, device_mesh_);
        HloSharding rhs_spec = Tile(rhs_->shape(), {rhs_space_dims_[j]},
                                    {mesh_dim1}, device_mesh_);

        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env_, strategy_map_, strategies_);
      }
    }
  }

  void SplitLhsSpaceOnly(int mesh_dim0, int mesh_dim1) {
    for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
      for (int64_t j = i + 1; j < lhs_space_dims_.size(); ++j) {
        if (lhs_->shape().dimensions().at(lhs_space_dims_.at(i)) <
                device_mesh_.dim(mesh_dim0) ||
            lhs_->shape().dimensions().at(lhs_space_dims_.at(j)) <
                device_mesh_.dim(mesh_dim1)) {
          continue;
        }
        if (solver_option_.only_allow_divisible_intermediate &&
            (!IsDivisible(lhs_->shape().dimensions().at(lhs_space_dims_.at(i)),
                          device_mesh_.dim(mesh_dim0)) ||
             !IsDivisible(lhs_->shape().dimensions().at(lhs_space_dims_.at(j)),
                          device_mesh_.dim(mesh_dim1)))) {
          continue;
        }
        std::string name =
            absl::StrFormat("SSR = SSR x RR @ {%d,%d}", mesh_dim0, mesh_dim1);
        HloSharding output_spec =
            Tile(ins_->shape(), {space_base_dim_ + i, space_base_dim_ + j},
                 {mesh_dim0, mesh_dim1}, device_mesh_);
        HloSharding lhs_spec =
            Tile(lhs_->shape(), {lhs_space_dims_[i], lhs_space_dims_[j]},
                 {mesh_dim0, mesh_dim1}, device_mesh_);
        HloSharding rhs_spec = HloSharding::Replicate();

        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env_, strategy_map_, strategies_);
      }
    }
  }

  void SplitRhsSpaceOnly(int mesh_dim0, int mesh_dim1) {
    for (int64_t i = 0; i < rhs_space_dims_.size(); ++i) {
      for (int64_t j = i + 1; j < rhs_space_dims_.size(); ++j) {
        if (rhs_->shape().dimensions().at(rhs_space_dims_.at(i)) <
                device_mesh_.dim(mesh_dim0) ||
            rhs_->shape().dimensions().at(rhs_space_dims_.at(j)) <
                device_mesh_.dim(mesh_dim1)) {
          continue;
        }
        if (solver_option_.only_allow_divisible_intermediate &&
            (!IsDivisible(rhs_->shape().dimensions().at(rhs_space_dims_.at(i)),
                          device_mesh_.dim(mesh_dim0)) ||
             !IsDivisible(rhs_->shape().dimensions().at(rhs_space_dims_.at(j)),
                          device_mesh_.dim(mesh_dim1)))) {
          continue;
        }
        std::string name =
            absl::StrFormat("RSS = RR x RSS @ {%d,%d}", mesh_dim0, mesh_dim1);
        HloSharding output_spec = Tile(
            ins_->shape(),
            {space_base_dim_ + static_cast<int64_t>(lhs_space_dims_.size()) + i,
             space_base_dim_ + static_cast<int64_t>(lhs_space_dims_.size()) +
                 j},
            {mesh_dim0, mesh_dim1}, device_mesh_);
        HloSharding lhs_spec = HloSharding::Replicate();
        HloSharding rhs_spec =
            Tile(rhs_->shape(), {rhs_space_dims_[i], rhs_space_dims_[j]},
                 {mesh_dim0, mesh_dim1}, device_mesh_);

        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env_, strategy_map_, strategies_);
      }
    }
  }

  void SplitLhsSpaceBothContract(int mesh_dim0, int mesh_dim1) {
    if (device_mesh_.dim(mesh_dim0) > 1 && device_mesh_.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim1);
      for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
        for (int64_t j = 0; j < lhs_con_dims_.size(); ++j) {
          if (lhs_->shape().dimensions().at(lhs_space_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim0) ||
              lhs_->shape().dimensions().at(lhs_con_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim1)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              (!IsDivisible(
                   lhs_->shape().dimensions().at(lhs_space_dims_.at(i)),
                   device_mesh_.dim(mesh_dim0)) ||
               !IsDivisible(lhs_->shape().dimensions().at(lhs_con_dims_.at(j)),
                            device_mesh_.dim(mesh_dim1)))) {
            continue;
          }

          HloSharding output_spec = Tile(ins_->shape(), {space_base_dim_ + i},
                                         {mesh_dim0}, device_mesh_);
          HloSharding lhs_spec =
              Tile(lhs_->shape(), {lhs_space_dims_[i], lhs_con_dims_[j]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          HloSharding rhs_spec = Tile(rhs_->shape(), {rhs_con_dims_[j]},
                                      {mesh_dim1}, device_mesh_);

          double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
          double communication_cost =
              cluster_env_.AllReduceCost(memory_cost, mesh_dim1);
          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                            communication_cost, cluster_env_, strategy_map_,
                            strategies_);
        }
      }
    }
  }

  void SplitRhsSpaceBothContract(int mesh_dim0, int mesh_dim1) {
    if (device_mesh_.dim(mesh_dim0) > 1) {
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim0);
      for (int64_t i = 0; i < rhs_space_dims_.size(); ++i) {
        for (int64_t j = 0; j < lhs_con_dims_.size(); ++j) {
          if (rhs_->shape().dimensions().at(rhs_space_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim1) ||
              lhs_->shape().dimensions().at(lhs_con_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim0)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              (!IsDivisible(
                   rhs_->shape().dimensions().at(rhs_space_dims_.at(i)),
                   device_mesh_.dim(mesh_dim1)) ||
               !IsDivisible(lhs_->shape().dimensions().at(lhs_con_dims_.at(j)),
                            device_mesh_.dim(mesh_dim0)))) {
            continue;
          }
          HloSharding output_spec =
              Tile(ins_->shape(),
                   {space_base_dim_ +
                    static_cast<int64_t>(lhs_space_dims_.size()) + i},
                   {mesh_dim1}, device_mesh_);
          HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_con_dims_[j]},
                                      {mesh_dim0}, device_mesh_);
          HloSharding rhs_spec =
              Tile(rhs_->shape(), {rhs_con_dims_[j], rhs_space_dims_[i]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
          double communication_cost =
              cluster_env_.AllReduceCost(memory_cost, mesh_dim0);

          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                            communication_cost, cluster_env_, strategy_map_,
                            strategies_);
        }
      }
    }
  }

  void SplitOneBatchDim() {
    if (absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) == 1) {
      for (int64_t i = 0; i < lhs_batch_dims_.size(); ++i) {
        for (int64_t j = 0; j < device_mesh_.num_dimensions(); ++j) {
          if (lhs_->shape().dimensions().at(lhs_batch_dims_.at(i)) <
              device_mesh_.dim(j)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              !IsDivisible(lhs_->shape().dimensions().at(lhs_batch_dims_.at(i)),
                           device_mesh_.dim(j))) {
            continue;
          }
          std::string name = absl::StrFormat("Sb_%d = Sb x Sb @ {%d}", i, j);
          HloSharding output_spec = Tile(ins_->shape(), {i}, {j}, device_mesh_);
          HloSharding lhs_spec =
              Tile(lhs_->shape(), {lhs_batch_dims_[i]}, {j}, device_mesh_);
          HloSharding rhs_spec =
              Tile(rhs_->shape(), {rhs_batch_dims_[i]}, {j}, device_mesh_);

          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                            cluster_env_, strategy_map_, strategies_);
        }
      }
    }
  }

  void SplitTwoBatchDims(int mesh_dim0, int mesh_dim1) {
    if (lhs_batch_dims_.size() == 2 && device_mesh_.dim(mesh_dim0) > 1 &&
        device_mesh_.dim(mesh_dim1) > 1) {
      if (lhs_->shape().dimensions().at(lhs_batch_dims_.at(0)) <
              device_mesh_.dim(mesh_dim0) ||
          lhs_->shape().dimensions().at(lhs_batch_dims_.at(1)) <
              device_mesh_.dim(mesh_dim1)) {
        return;
      }
      if (solver_option_.only_allow_divisible_intermediate &&
          (!IsDivisible(lhs_->shape().dimensions().at(lhs_batch_dims_.at(0)),
                        device_mesh_.dim(mesh_dim0)) ||
           !IsDivisible(lhs_->shape().dimensions().at(lhs_batch_dims_.at(1)),
                        device_mesh_.dim(mesh_dim1)))) {
        return;
      }
      std::string name =
          absl::StrFormat("Sb = Sb x Sb @ {%d,%d}", mesh_dim0, mesh_dim1);
      HloSharding output_spec =
          Tile(ins_->shape(), {0, 1}, {mesh_dim0, mesh_dim1}, device_mesh_);
      HloSharding lhs_spec =
          Tile(lhs_->shape(), {lhs_batch_dims_[0], lhs_batch_dims_[1]},
               {mesh_dim0, mesh_dim1}, device_mesh_);
      HloSharding rhs_spec =
          Tile(rhs_->shape(), {rhs_batch_dims_[0], rhs_batch_dims_[1]},
               {mesh_dim0, mesh_dim1}, device_mesh_);
      AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                        cluster_env_, strategy_map_, strategies_);
    }
  }

  void SplitBatchDimLhsSpace(int mesh_dim0, int mesh_dim1) {
    if (!lhs_batch_dims_.empty() && device_mesh_.dim(mesh_dim0) > 1 &&
        device_mesh_.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SbSi = SbSi x SbR @ {%d,%d}", mesh_dim0, mesh_dim1);
      for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
        for (int64_t j = 0; j < lhs_batch_dims_.size(); ++j) {
          if (lhs_->shape().dimensions().at(lhs_space_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim0) ||
              lhs_->shape().dimensions().at(lhs_batch_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim1)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              (!IsDivisible(
                   lhs_->shape().dimensions().at(lhs_space_dims_.at(i)),
                   device_mesh_.dim(mesh_dim0)) ||
               !IsDivisible(
                   lhs_->shape().dimensions().at(lhs_batch_dims_.at(j)),
                   device_mesh_.dim(mesh_dim1)))) {
            continue;
          }
          HloSharding output_spec =
              Tile(ins_->shape(), {j, space_base_dim_ + i},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          HloSharding lhs_spec =
              Tile(lhs_->shape(), {lhs_batch_dims_[j], lhs_space_dims_[i]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          HloSharding rhs_spec = Tile(rhs_->shape(), {rhs_batch_dims_[j]},
                                      {mesh_dim0}, device_mesh_);

          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                            cluster_env_, strategy_map_, strategies_);
        }
      }
    }
  }

  void SplitBatchDimRhsSpace(int mesh_dim0, int mesh_dim1) {
    if (!lhs_batch_dims_.empty() && device_mesh_.dim(mesh_dim0) > 1 &&
        device_mesh_.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SbSj = SbR x SbSj @ {%d,%d}", mesh_dim0, mesh_dim1);
      for (int64_t i = 0; i < rhs_space_dims_.size(); ++i) {
        for (int64_t j = 0; j < lhs_batch_dims_.size(); ++j) {
          if (rhs_->shape().dimensions().at(rhs_space_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim1) ||
              lhs_->shape().dimensions().at(lhs_batch_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim0)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              (!IsDivisible(
                   rhs_->shape().dimensions().at(rhs_space_dims_.at(i)),
                   device_mesh_.dim(mesh_dim1)) ||
               !IsDivisible(
                   lhs_->shape().dimensions().at(lhs_batch_dims_.at(j)),
                   device_mesh_.dim(mesh_dim0)))) {
            continue;
          }
          HloSharding output_spec =
              Tile(ins_->shape(),
                   {j, space_base_dim_ +
                           static_cast<int64_t>(lhs_space_dims_.size()) + i},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_batch_dims_[j]},
                                      {mesh_dim0}, device_mesh_);
          HloSharding rhs_spec =
              Tile(rhs_->shape(), {rhs_batch_dims_[j], rhs_space_dims_[i]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);

          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                            cluster_env_, strategy_map_, strategies_);
        }
      }
    }
  }

  void SplitBatchDimBothContract(int mesh_dim0, int mesh_dim1) {
    if (!lhs_batch_dims_.empty() && device_mesh_.dim(mesh_dim0) > 1 &&
        device_mesh_.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SbR = SbSk x SbSk @ {%d,%d} (allreduce @ %d}",
                          mesh_dim0, mesh_dim1, mesh_dim1);
      for (int64_t i = 0; i < lhs_con_dims_.size(); ++i) {
        for (int64_t j = 0; j < lhs_batch_dims_.size(); ++j) {
          if (lhs_->shape().dimensions().at(lhs_con_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim1) ||
              lhs_->shape().dimensions().at(lhs_batch_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim0)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              (!IsDivisible(lhs_->shape().dimensions().at(lhs_con_dims_.at(i)),
                            device_mesh_.dim(mesh_dim1)) ||
               !IsDivisible(
                   lhs_->shape().dimensions().at(lhs_batch_dims_.at(j)),
                   device_mesh_.dim(mesh_dim0)))) {
            continue;
          }
          HloSharding output_spec =
              Tile(ins_->shape(), {j}, {mesh_dim0}, device_mesh_);
          HloSharding lhs_spec =
              Tile(lhs_->shape(), {lhs_batch_dims_[j], lhs_con_dims_[i]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          HloSharding rhs_spec =
              Tile(rhs_->shape(), {rhs_batch_dims_[j], rhs_con_dims_[i]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
          double communication_cost =
              cluster_env_.AllReduceCost(memory_cost, mesh_dim1);

          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                            communication_cost, cluster_env_, strategy_map_,
                            strategies_);
        }
      }
    }
  }

  void SplitBothContractTwoDims(int mesh_dim0, int mesh_dim1) {
    // Applies when there are more than one contracting dimension.
    if (lhs_con_dims_.size() >= 2 && rhs_con_dims_.size() >= 2 &&
        device_mesh_.dim(mesh_dim0) > 1 && device_mesh_.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("RR = SS x SS @ {%d,%d} (allreduce @ {%d, %d}}",
                          mesh_dim0, mesh_dim1, mesh_dim0, mesh_dim1);
      for (int64_t i = 0; i < lhs_con_dims_.size(); ++i) {
        for (int64_t j = i + 1; j < lhs_con_dims_.size(); ++j) {
          if (lhs_->shape().dimensions().at(lhs_con_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim0) ||
              lhs_->shape().dimensions().at(lhs_con_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim1) ||
              rhs_->shape().dimensions().at(rhs_con_dims_.at(i)) <
                  device_mesh_.dim(mesh_dim0) ||
              rhs_->shape().dimensions().at(rhs_con_dims_.at(j)) <
                  device_mesh_.dim(mesh_dim1)) {
            continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              (!IsDivisible(lhs_->shape().dimensions().at(lhs_con_dims_.at(i)),
                            device_mesh_.dim(mesh_dim0)) ||
               !IsDivisible(lhs_->shape().dimensions().at(lhs_con_dims_.at(j)),
                            device_mesh_.dim(mesh_dim1)) ||
               !IsDivisible(rhs_->shape().dimensions().at(rhs_con_dims_.at(i)),
                            device_mesh_.dim(mesh_dim0)) ||
               !IsDivisible(rhs_->shape().dimensions().at(rhs_con_dims_.at(j)),
                            device_mesh_.dim(mesh_dim1)))) {
            continue;
          }
          HloSharding output_spec = HloSharding::Replicate();
          HloSharding lhs_spec =
              Tile(lhs_->shape(), {lhs_con_dims_[i], lhs_con_dims_[j]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          HloSharding rhs_spec =
              Tile(rhs_->shape(), {rhs_con_dims_[i], rhs_con_dims_[j]},
                   {mesh_dim0, mesh_dim1}, device_mesh_);
          double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
          double communication_cost =
              cluster_env_.AllReduceCost(memory_cost, mesh_dim0, mesh_dim1);
          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                            communication_cost, cluster_env_, strategy_map_,
                            strategies_);
        }
      }
    }
  }

  void RecomputeSplitBothContract(int mesh_dim0, int mesh_dim1) {
    if (device_mesh_.dim(mesh_dim0) > 1 && device_mesh_.dim(mesh_dim1) > 1) {
      std::string name = absl::StrFormat("RR = RS x SR @ {%d} (allreduce @ %d)",
                                         mesh_dim0, mesh_dim0);
      for (int64_t i = 0; i < lhs_con_dims_.size(); ++i) {
        if (lhs_->shape().dimensions().at(lhs_con_dims_.at(i)) <
            device_mesh_.dim(mesh_dim0)) {
          continue;
        }
        if (solver_option_.only_allow_divisible_intermediate &&
            !IsDivisible(lhs_->shape().dimensions().at(lhs_con_dims_.at(i)),
                         device_mesh_.dim(mesh_dim0))) {
          continue;
        }
        HloSharding output_spec = HloSharding::Replicate();
        HloSharding lhs_spec =
            Tile(lhs_->shape(), {lhs_con_dims_[i]}, {mesh_dim0}, device_mesh_);
        HloSharding rhs_spec =
            Tile(rhs_->shape(), {rhs_con_dims_[i]}, {mesh_dim0}, device_mesh_);
        double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
        double compute_cost =
            cluster_env_.DotCost(lhs_->shape(), rhs_->shape(), dot_dnums_);
        double communication_cost =
            cluster_env_.AllReduceCost(memory_cost, mesh_dim0);

        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec},
                          compute_cost, communication_cost, cluster_env_,
                          strategy_map_, strategies_);
      }
    }
  }

  void Add1DDataParallel() {
    if (device_mesh_.dim(0) > 1 &&
        absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) > 1) {
      int mesh_dim = 0;
      int64_t num_devices = device_mesh_1d_.dim(mesh_dim);

      // Si = Si x R @ 0
      for (int64_t i = 0; i < lhs_space_dims_.size(); ++i) {
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
          HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_space_dims_[i]},
                                      {mesh_dim}, device_mesh_1d_);
          HloSharding rhs_spec = HloSharding::Replicate();
          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                            cluster_env_, strategy_map_, strategies_);
      }

      // R = Sk x Sk @ (allreduce @ 0)
      for (int64_t i = 0; i < lhs_con_dims_.size(); ++i) {
          if (lhs_->shape().dimensions(lhs_con_dims_[i]) < num_devices) {
          continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              !IsDivisible(lhs_->shape().dimensions(lhs_con_dims_[i]),
                           num_devices)) {
          continue;
          }
          std::string name = absl::StrFormat(
              "R = Sk x Sk @ %d (allreduce @ %d)", mesh_dim, mesh_dim);
          HloSharding output_spec = HloSharding::Replicate();
          HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_con_dims_[i]},
                                      {mesh_dim}, device_mesh_1d_);
          HloSharding rhs_spec = Tile(rhs_->shape(), {rhs_con_dims_[i]},
                                      {mesh_dim}, device_mesh_1d_);
          double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
          double communication_cost =
              cluster_env_.AllReduceCost(memory_cost, mesh_dim);

          AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                            communication_cost, cluster_env_, strategy_map_,
                            strategies_);
        }
    }
  }

  void Add1DBatchSplit() {
    if (device_mesh_.dim(0) > 1 &&
        absl::c_count_if(device_mesh_.dimensions(),
                         [](int64_t size) { return size > 1; }) > 1) {
      int mesh_dim = 0;
      for (int64_t i = 0; i < lhs_batch_dims_.size(); ++i) {
          if (rhs_->shape().dimensions().at(lhs_batch_dims_.at(i)) <
              device_mesh_.dim(mesh_dim)) {
          continue;
          }
          if (solver_option_.only_allow_divisible_intermediate &&
              !IsDivisible(rhs_->shape().dimensions().at(lhs_batch_dims_.at(i)),
                           device_mesh_.dim(mesh_dim))) {
          continue;
          }
        std::string name =
            absl::StrFormat("Sb_%d = Sb x Sb @ {%d} 1d", i, mesh_dim);
        HloSharding output_spec =
            Tile(ins_->shape(), {i}, {mesh_dim}, device_mesh_1d_);
        HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_batch_dims_[i]},
                                    {mesh_dim}, device_mesh_1d_);
        HloSharding rhs_spec = Tile(rhs_->shape(), {rhs_batch_dims_[i]},
                                    {mesh_dim}, device_mesh_1d_);
        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env_, strategy_map_, strategies_);
      }
    }
  }

  Status RegisterStrategies() {
    std::vector<int64_t> shardable_mesh_dims =
        VectorGreaterThanOneElementIndices(device_mesh_.dimensions());
    // For 1D sharding
    if (shardable_mesh_dims.size() == 1) {
      shardable_mesh_dims.push_back((shardable_mesh_dims.at(0) + 1) %
                                    device_mesh_.num_dimensions());
    }

    // SS = SR x RS
    // Split lhs space dim and rhs space dim.
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitLhsSpaceRhsSpace(shardable_mesh_dims[i], shardable_mesh_dims[j]);
        SplitLhsSpaceRhsSpace(shardable_mesh_dims[j], shardable_mesh_dims[i]);
      }
    }

    // SSR = SSR x RR
    // Split lhs space dims only if it has more than 1 space dims.
    if (lhs_space_dims_.size() > 1) {
      for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
        for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
          SplitLhsSpaceOnly(shardable_mesh_dims[i], shardable_mesh_dims[j]);
          SplitLhsSpaceOnly(shardable_mesh_dims[j], shardable_mesh_dims[i]);
        }
      }
    }
    // RSS = RR x RSS
    // Split rhs space dims only if it has more than 1 space dims.
    if (rhs_space_dims_.size() > 1) {
      for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
        for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
          SplitRhsSpaceOnly(shardable_mesh_dims[i], shardable_mesh_dims[j]);
          SplitRhsSpaceOnly(shardable_mesh_dims[j], shardable_mesh_dims[i]);
        }
      }
    }

    // SR = SS x SR
    // Split lhs space dim and both contracting dims.
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitLhsSpaceBothContract(shardable_mesh_dims[i],
                                  shardable_mesh_dims[j]);
        SplitLhsSpaceBothContract(shardable_mesh_dims[j],
                                  shardable_mesh_dims[i]);
      }
    }

    // RS = RS x SS
    // Split rhs space dim and both contracting dims.
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitRhsSpaceBothContract(shardable_mesh_dims[i],
                                  shardable_mesh_dims[j]);
        SplitRhsSpaceBothContract(shardable_mesh_dims[j],
                                  shardable_mesh_dims[i]);
      }
    }

    // RR = SS x SS
    // Split two contracting dims on lhs and rhs.
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitBothContractTwoDims(shardable_mesh_dims[i],
                                 shardable_mesh_dims[j]);
        SplitBothContractTwoDims(shardable_mesh_dims[j],
                                 shardable_mesh_dims[i]);
      }
    }

    // RR = RS x SR
    // This is a special case where we allow spliting only one dim in the
    // multi-dimensional mesh case. This allows some recomputation
    // (e.g., the dense layer in the LM_head of BERT).
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        RecomputeSplitBothContract(shardable_mesh_dims[i],
                                   shardable_mesh_dims[j]);
        RecomputeSplitBothContract(shardable_mesh_dims[j],
                                   shardable_mesh_dims[i]);
      }
    }

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
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitBatchDimLhsSpace(shardable_mesh_dims[i], shardable_mesh_dims[j]);
        SplitBatchDimLhsSpace(shardable_mesh_dims[j], shardable_mesh_dims[i]);
      }
    }

    // SbSj = SbR x SbSj
    // Split batch dim and rhs space dim
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitBatchDimRhsSpace(shardable_mesh_dims[i], shardable_mesh_dims[j]);
        SplitBatchDimRhsSpace(shardable_mesh_dims[j], shardable_mesh_dims[i]);
      }
    }

    // SbSj = SbR x SbSj
    // Split batch dim and contracting dim
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitBatchDimBothContract(shardable_mesh_dims[i],
                                  shardable_mesh_dims[j]);
        SplitBatchDimBothContract(shardable_mesh_dims[j],
                                  shardable_mesh_dims[i]);
      }
    }

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
    for (int64_t i = 0; i < shardable_mesh_dims.size(); ++i) {
      for (int64_t j = (i + 1); j < shardable_mesh_dims.size(); ++j) {
        SplitTwoBatchDims(shardable_mesh_dims[i], shardable_mesh_dims[j]);
        SplitTwoBatchDims(shardable_mesh_dims[j], shardable_mesh_dims[i]);
      }
    }

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

  std::unique_ptr<StrategyVector>& strategies_;
  StrategyMap& strategy_map_;
  const HloInstruction* ins_;
  const ClusterEnvironment& cluster_env_;
  const InstructionBatchDimMap& batch_map_;
  const AutoShardingSolverOption& solver_option_;

  const Array<int64_t>& device_mesh_;
  const Array<int64_t>& device_mesh_1d_;
  const HloInstruction* lhs_;
  const HloInstruction* rhs_;

  // Dimension information
  const DotDimensionNumbers& dot_dnums_;
  int64_t space_base_dim_;
  std::vector<int64_t> lhs_space_dims_, rhs_space_dims_;
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
                 const AutoShardingSolverOption& solver_option) {
  strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                        leaf_strategies);

  DotHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                     solver_option);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return OkStatus();
}

class ConvHandler {
 public:
  ConvHandler(std::unique_ptr<StrategyVector>& strategies,
              StrategyMap& strategy_map, const HloInstruction* ins,
              const ClusterEnvironment& cluster_env,
              const InstructionBatchDimMap& batch_map,
              const AutoShardingSolverOption& solver_option)
      : strategies_(strategies),
        strategy_map_(strategy_map),
        ins_(ins),
        cluster_env_(cluster_env),
        batch_map_(batch_map),
        solver_option_(solver_option),
        device_mesh_(cluster_env.device_mesh_),
        device_mesh_1d_(cluster_env.device_mesh_1d_),
        lhs_(ins->operand(0)),
        rhs_(ins->operand(1)),
        conv_dnums_(ins->convolution_dimension_numbers()) {
    lhs_batch_dim_ = conv_dnums_.input_batch_dimension();
    lhs_in_channel_dim_ = conv_dnums_.input_feature_dimension();
    rhs_in_channel_dim_ = conv_dnums_.kernel_input_feature_dimension();
    rhs_out_channel_dim_ = conv_dnums_.kernel_output_feature_dimension();
    out_batch_dim_ = conv_dnums_.output_batch_dimension();
    out_out_channel_dim_ = conv_dnums_.output_feature_dimension();

    // Only support 2 dimensional device mesh
    CHECK_EQ(device_mesh_.num_dimensions(), 2);
  }

  void SplitLhsBatchRhsOutchannel(int mesh_dim0, int mesh_dim1) {
    std::string name =
        absl::StrFormat("SS = SR x RS @ {%d,%d}", mesh_dim0, mesh_dim1);
    HloSharding output_spec =
        Tile(ins_->shape(), {out_batch_dim_, out_out_channel_dim_},
             {mesh_dim0, mesh_dim1}, device_mesh_);
    HloSharding lhs_spec =
        Tile(lhs_->shape(), {lhs_batch_dim_}, {mesh_dim0}, device_mesh_);
    HloSharding rhs_spec =
        Tile(rhs_->shape(), {rhs_out_channel_dim_}, {mesh_dim1}, device_mesh_);

    AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                      cluster_env_, strategy_map_, strategies_);
  }

  void SplitLhsBatchBothInchannel(int mesh_dim0, int mesh_dim1) {
    if (device_mesh_.dim(mesh_dim0) > 1 && device_mesh_.dim(mesh_dim1) > 1) {
      std::string name =
          absl::StrFormat("SR = SS x SR @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim1);
      HloSharding output_spec =
          Tile(ins_->shape(), {out_batch_dim_}, {mesh_dim0}, device_mesh_);
      HloSharding lhs_spec =
          Tile(lhs_->shape(), {lhs_batch_dim_, lhs_in_channel_dim_},
               {mesh_dim0, mesh_dim1}, device_mesh_);
      HloSharding rhs_spec =
          Tile(rhs_->shape(), {rhs_in_channel_dim_}, {mesh_dim1}, device_mesh_);

      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, mesh_dim1);

      AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env_, strategy_map_,
                        strategies_);
    }
  }

  void SplitRhsOutchannelBothInchannel(int mesh_dim0, int mesh_dim1) {
    if (device_mesh_.dim(mesh_dim0) > 1) {
      std::string name =
          absl::StrFormat("RS = RS x SS @ {%d,%d} (allreduce @ %d)", mesh_dim0,
                          mesh_dim1, mesh_dim0);
      HloSharding output_spec = Tile(ins_->shape(), {out_out_channel_dim_},
                                     {mesh_dim1}, device_mesh_);
      HloSharding lhs_spec =
          Tile(lhs_->shape(), {lhs_in_channel_dim_}, {mesh_dim0}, device_mesh_);
      HloSharding rhs_spec =
          Tile(rhs_->shape(), {rhs_in_channel_dim_, rhs_out_channel_dim_},
               {mesh_dim0, mesh_dim1}, device_mesh_);

      double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
      double communication_cost =
          cluster_env_.AllReduceCost(memory_cost, mesh_dim0);

      AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                        communication_cost, cluster_env_, strategy_map_,
                        strategies_);
    }
  }

  void Add1DDataParallel() {
    if (device_mesh_.dim(0) > 1 && device_mesh_.dim(1) > 1) {
      int mesh_dim = 0;
      int64_t num_devices = device_mesh_1d_.dim(mesh_dim);

      // Si = Si x R @ 0
      if (lhs_->shape().dimensions(lhs_batch_dim_) % num_devices == 0) {
        std::string name = absl::StrFormat("Si = Si x R @ 0");
        HloSharding output_spec =
            Tile(ins_->shape(), {out_batch_dim_}, {mesh_dim}, device_mesh_1d_);
        HloSharding lhs_spec =
            Tile(lhs_->shape(), {lhs_batch_dim_}, {mesh_dim}, device_mesh_1d_);
        HloSharding rhs_spec = HloSharding::Replicate();

        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                          cluster_env_, strategy_map_, strategies_);
      }

      // R = Sk x Sk @ (allreduce @ 0)
      if (lhs_->shape().dimensions(lhs_in_channel_dim_) % num_devices == 0 &&
          rhs_->shape().dimensions(rhs_in_channel_dim_) % num_devices == 0) {
        std::string name = absl::StrFormat("R = Sk x Sk @ %d (allreduce @ %d)",
                                           mesh_dim, mesh_dim);
        HloSharding output_spec = HloSharding::Replicate();
        HloSharding lhs_spec = Tile(lhs_->shape(), {lhs_in_channel_dim_},
                                    {mesh_dim}, device_mesh_1d_);
        HloSharding rhs_spec = Tile(rhs_->shape(), {rhs_in_channel_dim_},
                                    {mesh_dim}, device_mesh_1d_);
        double memory_cost = GetBytes(ins_->shape()) / output_spec.NumTiles();
        double communication_cost = cluster_env_.AllReduceCost(memory_cost, 0) +
                                    cluster_env_.AllReduceCost(memory_cost, 1);

        AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0,
                          communication_cost, cluster_env_, strategy_map_,
                          strategies_);
      }
    }
  }

  void SplitDepthwise(int mesh_dim0, int mesh_dim1, bool forward) {
    std::string name =
        absl::StrFormat("SS = SS x RS @ {%d,%d}", mesh_dim0, mesh_dim1);
    HloSharding output_spec =
        Tile(ins_->shape(), {out_batch_dim_, out_out_channel_dim_},
             {mesh_dim0, mesh_dim1}, device_mesh_);
    HloSharding lhs_spec =
        forward ? Tile(lhs_->shape(), {lhs_batch_dim_, lhs_in_channel_dim_},
                       {mesh_dim0, mesh_dim1}, device_mesh_)
                : Tile(lhs_->shape(), {lhs_batch_dim_, lhs_in_channel_dim_},
                       {mesh_dim1, mesh_dim0}, device_mesh_);

    HloSharding rhs_spec =
        Tile(rhs_->shape(), {rhs_out_channel_dim_}, {mesh_dim1}, device_mesh_);

    AppendNewStrategy(ins_, name, output_spec, {lhs_spec, rhs_spec}, 0, 0,
                      cluster_env_, strategy_map_, strategies_);
  }

  Status RegisterStrategies() {
    if (device_mesh_.num_dimensions() > 2) {
      return tsl::errors::Internal(
          "This function does not support 3D mesh shape with convolution ops "
          "yet.");
    }
    if ((ins_->feature_group_count() ==
             lhs_->shape().dimensions(lhs_in_channel_dim_) &&
         ins_->feature_group_count() ==
             rhs_->shape().dimensions(rhs_out_channel_dim_))) {
      // for depthwise conv
      // SS = SS x S
      // Split batch dim and channel dim
      SplitDepthwise(0, 1, true);
      SplitDepthwise(1, 0, true);
    } else if ((ins_->batch_group_count() ==
                    lhs_->shape().dimensions(lhs_batch_dim_) &&
                ins_->batch_group_count() ==
                    rhs_->shape().dimensions(rhs_out_channel_dim_))) {
      // for depthwise conv filter_backward
      // SS = SS x S
      // Split batch dim and channel dim
      SplitDepthwise(0, 1, false);
      SplitDepthwise(1, 0, false);
    }

    // SS = SR x RS
    // Split lhs batch dim and rhs out_channel dim.
    SplitLhsBatchRhsOutchannel(0, 1);
    SplitLhsBatchRhsOutchannel(1, 0);

    // SR = SS x SR
    // Split lhs batch dim and both in_channel dims.
    SplitLhsBatchBothInchannel(0, 1);
    SplitLhsBatchBothInchannel(1, 0);

    // RS = RS x SS
    // Split rhs out_channel dim and both in_channel dims.
    SplitRhsOutchannelBothInchannel(0, 1);
    SplitRhsOutchannelBothInchannel(1, 0);

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

  std::unique_ptr<StrategyVector>& strategies_;
  StrategyMap& strategy_map_;
  const HloInstruction* ins_;
  const ClusterEnvironment& cluster_env_;
  const InstructionBatchDimMap& batch_map_;
  const AutoShardingSolverOption& solver_option_;

  const Array<int64_t>& device_mesh_;
  const Array<int64_t>& device_mesh_1d_;
  const HloInstruction* lhs_;
  const HloInstruction* rhs_;

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
                  const AutoShardingSolverOption& solver_option) {
  strategies = CreateLeafStrategyVector(instruction_id, ins, strategy_map,
                                        leaf_strategies);

  ConvHandler handler(strategies, strategy_map, ins, cluster_env, batch_map,
                      solver_option);
  TF_RETURN_IF_ERROR(handler.RegisterStrategies());
  return OkStatus();
}

}  // namespace spmd
}  // namespace xla

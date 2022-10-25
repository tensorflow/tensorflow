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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_
#define TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding_util.h"
#include "tensorflow/compiler/xla/service/hlo_live_range.h"

namespace xla {
namespace spmd {

// A constant to represent infinity cost.
constexpr double kInfinityCost = 1e13;

// Options for the auto-sharding solver.
struct AutoShardingSolverOption {
  // Forcibly split the batch dimension and map it to a mesh dimension.
  // This can force the auto-sharding pass to generate the data parallel
  // strategy.
  int force_batch_dim_to_mesh_dim;

  // If true, override the cost of all-gather with the given value.
  bool override_all_gather_cost;
  double all_gather_cost;

  // If true, override the cost of all-reduce with the given value.
  bool override_all_reduce_cost;
  double all_reduce_cost;

  // If true, override the cost of reduce-scatter with the given value.
  bool override_reduce_scatter_cost;
  double reduce_scatter_cost;

  // If true, override the cost of all-to-all with the given value.
  bool override_all_to_all_cost;
  double all_to_all_cost;

  // If true, allow replicated parameters.
  bool allow_replicated_parameters;

  // If true, prefer reduce-scatter + all-gather over all-reduce.
  // A post process will be applied to replace all-reduce with reduce-scater +
  // all-gather if no communication overhead is introduced.
  bool prefer_reduce_scatter;

  // If True, generate a gradient-accumulation friendly variant of
  // reduce-scatter
  bool reduce_scatter_grad_acc_friendly;

  // If true, aggressively partition more tensors when generating
  // reduce-scatter, even if it introduces more communication.
  bool reduce_scatter_aggressive_partition;

  // If true, the batch matmul will always be parallelized on the batch dim in
  // 2d mesh case.
  bool batch_matmul_always_split_batch;

  // If true, allow strategies that recompute heavy operators (e.g., dot)
  // to reduce communication.
  bool allow_recompute_heavy_op;

  // If true, allow adding 1d strategies in 2d logical mesh.
  bool allow_mixed_mesh_shape;

  // The number of micro batches if gradient accumulation is used.
  // If this is not 1, the cost of all-reduce for gradient synchronization
  // is divided by this number.
  int grad_acc_num_micro_batches;

  // If true, load solution vector from PassContext
  bool load_solution_vector;

  // If it is not empty, forcibly use simple heuristic strategies
  // instead of the ILP solver. This is used for ablation study.
  std::string force_simple_heuristic;

  // If true, forcibly set the strategy of some instructions
  bool force_strategy;
  std::vector<int64_t> force_strategy_inst_indices;
  std::vector<std::string> force_strategy_stra_names;
};

// One sharding strategy
struct ShardingStrategy {
  std::string name;
  HloSharding output_sharding;
  double compute_cost;
  double communication_cost;
  double memory_cost;
  // resharding_costs[i][j] is the resharding cost from the output of
  // i-th operand's j-th strategy to this strategy.
  // If there is only one tuple operand,resharding_costs[i][j] is the resharding
  // cost from i-th tuple element's j-th strategy.
  std::vector<std::vector<double>> resharding_costs;
  // Optional: the required shardings of operands.
  // This is used to guide the SPMD partitioner.
  std::vector<HloSharding> input_shardings;

  std::string ToString() const {
    return absl::StrCat(name, ", ", output_sharding.ToString());
  }

  std::string ToStringLong() const {
    std::vector<std::string> resharding_vector_strings;
    resharding_vector_strings.reserve(resharding_costs.size());
    for (const auto& v : resharding_costs) {
      resharding_vector_strings.push_back(
          absl::StrCat("[", absl::StrJoin(v, ", "), "]"));
    }
    std::string resharding_cost_str =
        absl::StrCat("{", absl::StrJoin(resharding_vector_strings, ", "), "}");
    std::string input_sharding_str = "{";
    for (const auto& s : input_shardings) {
      if (s.IsReplicated()) {
        input_sharding_str += "[R],";
      } else {
        if (s.ReplicateOnLastTileDim()) {
          input_sharding_str +=
              "[" + absl::StrJoin(s.tile_assignment().dimensions(), ", ") +
              "]last_tile_dim_replicate,";
        } else {
          input_sharding_str +=
              "[" + absl::StrJoin(s.tile_assignment().dimensions(), ", ") +
              "],";
        }
      }
    }
    input_sharding_str += "}\n";
    return absl::StrCat(name, ", ", output_sharding.ToString(),
                        ", compute_cost=", compute_cost,
                        ", communication_cost=", communication_cost,
                        ", memory_cost=", memory_cost,
                        ", resharding_costs=", resharding_cost_str,
                        ", input_shardings=", input_sharding_str);
  }
};

// The strategy choices for each instruction.
struct StrategyVector {
  bool is_tuple;
  // The index used in the solver. For non-leaf nodes, this is set to -1.
  int64_t id;
  // The index of the HLO instruction that this strategy vector belongs to.
  size_t instruction_id;
  // The connected nodes used for resharding costs;
  // The size must be the same as the size of resharding cost
  // each element in leaf_vector's resharding_costs.size() needs to be the same
  // as strategies->in_nodes.size()
  std::vector<const StrategyVector*> in_nodes;
  // The followed strategy. Used for merging nodes.
  const StrategyVector* following = nullptr;
  // Used when is_tuple == False. Leaf strategy vector.
  // A vector of strategy choices for the non-tuple output.
  std::vector<ShardingStrategy> leaf_vector;
  // Used when is_tuple == True. A vector of pointers, each pointer is one
  // StrategyVector for one value in the output Tuple
  std::vector<std::unique_ptr<StrategyVector>> childs;

  std::string ToString(size_t indention = 0) const {
    std::string str;
    absl::StrAppend(&str, std::string(indention, ' '), "id: ", id, "\n");
    absl::StrAppend(&str, std::string(indention, ' '),
                    "instruction id: ", instruction_id, "\n");
    absl::StrAppend(&str, std::string(indention, ' '), "is_tuple: ", is_tuple,
                    "\n");
    if (following != nullptr) {
      absl::StrAppend(&str, std::string(indention, ' '),
                      "following instruction: ", following->instruction_id,
                      "\n");
    } else {
      absl::StrAppend(&str, std::string(indention, ' '),
                      "source instruction\n");
    }
    for (auto i : in_nodes) {
      absl::StrAppend(&str, std::string(indention, ' '), "in nodes: id=", i->id,
                      " instruction_id=", i->instruction_id, "\n");
    }
    if (is_tuple) {
      for (size_t i = 0; i < childs.size(); ++i) {
        absl::StrAppend(&str, std::string(indention, ' '), "Tuple element #", i,
                        ":\n");
        absl::StrAppend(&str, childs[i]->ToString(indention + 2));
      }
    } else {
      for (const auto& strategy : leaf_vector) {
        absl::StrAppend(&str, std::string(indention, ' '), "Strategy ",
                        strategy.ToStringLong());
      }
    }
    return str;
  }
};

// Type aliases.
using LivenessSet = std::vector<std::vector<const HloValue*>>;
// Map an instruction to its strategy vector.
using StrategyMap =
    StableHashMap<const HloInstruction*, std::unique_ptr<StrategyVector>>;
// The list of all leaf strategies.
using LeafStrategies = std::vector<StrategyVector*>;
// The list of all dot instruction pairs that can be optimized by
// AllReduceReassociate pass.
using AssociativeDotPairs =
    std::vector<std::pair<const StrategyVector*, const StrategyVector*>>;
// The set of all alias pairs
using AliasSet = StableHashSet<std::pair<int64_t, int64_t>>;

// Store the profiling results of communication and computation.
class ProfilingResult {
 public:
  // TODO (zhuohan): loading the profiling result.
  ProfilingResult() {
    if (all_reduce_cost_dict_.empty()) {
      enabled_ = false;
    } else {
      enabled_ = true;
    }
  }

  bool Enabled() const { return enabled_; }

  double EstimateAllGatherCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    if (all_gather_cost_dict_.empty()) {
      // Use all-reduce to approximate all-gather.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            all_gather_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_gather_cost_dict_);
  }

  double EstimateAllReduceCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    return EstimateInternal(replica_groups, size, dtype,
                            all_reduce_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype, all_reduce_cost_dict_);
  }

  double EstimateReduceScatterCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    if (reduce_scatter_cost_dict_.empty()) {
      // Use all-reduce to approximate reduce-scatter.
      return EstimateAllReduceCost(replica_groups, size, dtype) / 2;
    }

    return EstimateInternal(replica_groups, size, dtype,
                            reduce_scatter_cost_dict_) -
           EstimateInternal(replica_groups, 0, dtype,
                            reduce_scatter_cost_dict_);
  }

  double EstimateAllToAllCost(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype) const {
    // A penalty factor to make the theoretical cost match the
    // empirical cost on v100 + nvlink.
    int64_t num_devices = replica_groups.front().size();
    double penalty_factor = static_cast<double>(num_devices) / 2.0;
    // Use all-gather to approximate all-to-all.
    return EstimateAllGatherCost(replica_groups, size / num_devices, dtype) *
           penalty_factor;
  }

  std::string ToString() {
    std::string str;
    for (const auto& item : all_reduce_cost_dict_) {
      absl::StrAppend(&str, item.first.first, " ", item.first.second, "\n");
    }
    return str;
  }

 private:
  // pair<group, dtype>
  using Key = std::pair<std::string, std::string>;
  // vector<pair<size, time>>
  using Value = std::vector<std::pair<int64_t, double>>;

  // Estimate the cost by linear interpolation between the two closest points.
  double EstimateInternal(
      const std::vector<std::vector<int64_t>>& replica_groups, int64_t size,
      const std::string& dtype,
      const StableHashMap<Key, Value>& cost_dict) const {
    Key key(Group2Str(replica_groups), dtype);
    Value cost_list = cost_dict.at(key);

    CHECK(!cost_list.empty());

    size_t i;
    if (size > cost_list.back().first) {
      i = cost_list.size() - 2;
    } else if (size < cost_list.front().first) {
      i = 0;
    } else {
      for (i = 0; i < cost_list.size() - 1; ++i) {
        if (cost_list[i].first <= size && size <= cost_list[i + 1].first) {
          break;
        }
      }
    }

    int64_t left_size = cost_list[i].first;
    double left_cost = cost_list[i].second;
    int64_t right_size = cost_list[i + 1].first;
    double right_cost = cost_list[i + 1].second;

    return 1.0 * (size - left_size) / (right_size - left_size) *
               (right_cost - left_cost) +
           left_cost;
  }

  // Make a string key of a replica_groups.
  std::string Group2Str(
      const std::vector<std::vector<int64_t>>& replica_groups) const {
    std::string str;
    absl::StrAppend(&str, "(");
    for (const auto& group : replica_groups) {
      absl::StrAppend(&str, "(", absl::StrJoin(group, ","), ")");
    }
    absl::StrAppend(&str, ")");

    return str;
  }

  bool enabled_;
  StableHashMap<Key, Value> all_reduce_cost_dict_;
  StableHashMap<Key, Value> all_gather_cost_dict_;
  StableHashMap<Key, Value> reduce_scatter_cost_dict_;
};

// The cluster has a multi-dimensional device mesh topology.
// Each mesh dimension has its own latency and bandwidth.
// We use alpha-beta model to model the communication cost.
// If profiling result is provided, we always prefer to use
// the real profiling result.
class ClusterEnvironment {
 public:
  ClusterEnvironment(const Array<int64_t>& original_device_mesh,
                     const Array<int64_t>& device_mesh,
                     absl::Span<const double> mesh_alpha,
                     absl::Span<const double> mesh_beta,
                     const ProfilingResult& prof_result,
                     const AutoShardingSolverOption& solver_option)
      : original_device_mesh_(original_device_mesh),
        device_mesh_(device_mesh),
        mesh_alpha_(mesh_alpha.begin(), mesh_alpha.end()),
        mesh_beta_(mesh_beta.begin(), mesh_beta.end()),
        prof_result_(prof_result),
        total_devices_(device_mesh.num_elements()),
        device_mesh_1d_(original_device_mesh),
        solver_option_(solver_option) {
    // Build replica group for each dimension.
    non_zero_mesh_dims_ =
        VectorGreaterThanOneElementIndices(device_mesh.dimensions());
    GenerateCachedReplicaGroups();
    // TODO(yuemmawang) Find the largest dimension in original_device_mesh and
    // create 1d mesh on that dimension.
    device_mesh_1d_.Reshape({original_device_mesh.num_elements(), 1});
  }

  size_t NumDevices() const { return total_devices_; }

  bool IsDeviceMesh3D() const {
    return VectorGreaterThanOneElementCount(device_mesh_.dimensions()) == 3;
  }

  bool IsDeviceMesh2D() const {
    return VectorGreaterThanOneElementCount(device_mesh_.dimensions()) == 2;
  }

  bool IsDeviceMesh1D() const {
    return VectorGreaterThanOneElementCount(device_mesh_.dimensions()) == 1;
  }

  bool IsOriginalDeviceMesh2D() const {
    return VectorGreaterThanOneElementCount(
               original_device_mesh_.dimensions()) == 2;
  }

  double AllGatherCost(double num_bytes, int mesh_dim) const {
    if (solver_option_.override_all_gather_cost) {
      return solver_option_.all_gather_cost;
    }

    if (prof_result_.Enabled()) {
      return prof_result_.EstimateAllGatherCost(
          cached_replica_groups_[mesh_dim], num_bytes / 4, "float32");
    }

    if (solver_option_.force_batch_dim_to_mesh_dim == mesh_dim) {
      // if data-parallel is forced on this dim, we only allow all-reduce
      // in this dimension.
      return kInfinityCost;
    }

    int64_t num_devices = device_mesh_.dim(mesh_dim);
    return (round(mesh_alpha_[mesh_dim] + mesh_beta_[mesh_dim] *
                                              (num_devices - 1) / num_devices *
                                              num_bytes) +
            0.1);
  }

  // TODO(zhuohan): distinguish dtype and reduce_op.
  double AllReduceCost(double num_bytes, int32_t mesh_dim,
                       int32_t mesh_dim_another = -1) const {
    if (solver_option_.override_all_reduce_cost) {
      return solver_option_.all_reduce_cost;
    }

    if (prof_result_.Enabled()) {
      return prof_result_.EstimateAllReduceCost(
          cached_replica_groups_[mesh_dim], num_bytes / 4, "float32");
    }
    double alpha, beta;
    int64_t num_devices;
    if (mesh_dim_another == -1) {
      // Only communicating on one mesh dimension.
      alpha = mesh_alpha_[mesh_dim];
      beta = mesh_beta_[mesh_dim];
      num_devices = device_mesh_.dim(mesh_dim);
    } else {
      // Communicating through both mesh dimensions.
      alpha = std::max(mesh_alpha_[mesh_dim], mesh_alpha_[mesh_dim_another]);
      beta = std::max(mesh_beta_[mesh_dim], mesh_beta_[mesh_dim_another]);
      num_devices = device_mesh_.num_elements();
    }
    return (
        round(alpha + beta * 2 * (num_devices - 1) / num_devices * num_bytes) +
        0.01);
  }

  double ReduceScatterCost(double num_bytes, int mesh_dim) const {
    if (solver_option_.override_reduce_scatter_cost) {
      return solver_option_.reduce_scatter_cost;
    }

    if (prof_result_.Enabled()) {
      return prof_result_.EstimateReduceScatterCost(
          cached_replica_groups_[mesh_dim], num_bytes / 4, "float32");
    }

    int64_t num_devices = device_mesh_.dim(mesh_dim);
    return (round(mesh_alpha_[mesh_dim] + mesh_beta_[mesh_dim] *
                                              (num_devices - 1) / num_devices *
                                              num_bytes) +
            0.001);
  }

  double AllToAllCost(double num_bytes, int mesh_dim) const {
    if (solver_option_.override_all_to_all_cost) {
      return solver_option_.all_to_all_cost;
    }

    if (prof_result_.Enabled()) {
      return prof_result_.EstimateAllToAllCost(cached_replica_groups_[mesh_dim],
                                               num_bytes / 4, "float32");
    }

    if (solver_option_.force_batch_dim_to_mesh_dim == mesh_dim) {
      // if data-parallel is forced on this dim, we only allow all-reduce
      // in this dimension.
      return kInfinityCost;
    }

    int64_t num_devices = device_mesh_.dim(mesh_dim);
    return AllToAllCostUtil(num_bytes, mesh_dim, num_devices, mesh_alpha_,
                            mesh_beta_);
  }

  double DotCost(const Shape& lhs_shape, const Shape& rhs_shape,
                 const DotDimensionNumbers& dot_dnums) const {
    if (!solver_option_.allow_recompute_heavy_op) {
      return kInfinityCost;
    }

    // TODO(zhuohan): When profiling data is not available, it is not easy to
    // align the scale of compute cost and communication cost. Here we just use
    // a simple heuristic to compute the compute cost with communication cost.
    double num_bytes = GetBytes(lhs_shape) + GetBytes(rhs_shape);
    return AllReduceCost(num_bytes, 0) + AllReduceCost(num_bytes, 1);
  }

  // Get the corresponding mesh dimension for every tensor dimension.
  // -1 means replicated on that dimension
  std::vector<int64_t> GetTensorDimToMeshDimWrapper(
      const Shape& shape, const HloSharding& spec) const {
    int64_t n_dim = NumTileDimensions(spec);
    std::vector<int64_t> tensor_dim_to_mesh_dim =
        GetTensorDimToMeshDim(shape.rank(), spec, device_mesh_);
    AdjustTensorMeshDimMapping(tensor_dim_to_mesh_dim, n_dim);
    return tensor_dim_to_mesh_dim;
  }

  // The communication cost of resharding a tensor from src to dst
  // TODO(b/238210866) Do not use kInfinityCost.
  double ReshardingCost(const Shape& shape, const HloSharding& src_spec,
                        const HloSharding& dst_spec) const {
    // TODO(zhuohan): This function can be wrong and needs more tests.
    if (src_spec == dst_spec || IsUndefined(src_spec)) {
      return 0.0;
    }
    CHECK(!IsUndefined(dst_spec));
    int64_t src_n_dim = NumTileDimensions(src_spec);
    int64_t dst_n_dim = NumTileDimensions(dst_spec);
    // When src_spec and dst_spec are for arrays with different number of
    // dimensions, which could happen when an instruction follows the sharding
    // of an operand with a different shape, we need to use their
    // TiledDataRank().
    size_t src_rank = shape.rank();
    if (src_spec.IsTiled()) {
      src_rank = src_spec.TiledDataRank();
    }
    size_t dst_rank = shape.rank();
    if (dst_spec.IsTiled()) {
      dst_rank = dst_spec.TiledDataRank();
    }
    std::vector<int64_t> src_tensor_dim_to_mesh_dim;
    if (VectorGreaterThanOneElementCount(
            src_spec.tile_assignment().dimensions()) == 1 &&
        VectorGreaterThanOneElementCount(device_mesh_.dimensions()) > 1) {
      // src spec is 1D and device_mesh is 2D or 3D
      src_tensor_dim_to_mesh_dim =
          GetTensorDimToMeshDim(src_rank, src_spec, device_mesh_1d_);
    } else {
      src_tensor_dim_to_mesh_dim =
          GetTensorDimToMeshDim(src_rank, src_spec, device_mesh_);
    }
    std::vector<int64_t> dst_tensor_dim_to_mesh_dim;
    if (VectorGreaterThanOneElementCount(
            dst_spec.tile_assignment().dimensions()) == 1 &&
        VectorGreaterThanOneElementCount(device_mesh_.dimensions()) > 1) {
      // src spec is 1D and device_mesh is 2D or 3D
      dst_tensor_dim_to_mesh_dim =
          GetTensorDimToMeshDim(dst_rank, dst_spec, device_mesh_1d_);
    } else {
      dst_tensor_dim_to_mesh_dim =
          GetTensorDimToMeshDim(dst_rank, dst_spec, device_mesh_);
    }
    if (src_n_dim != dst_n_dim && src_n_dim != -1 && dst_n_dim != -1) {
      return ReshardingCostMixedMeshShape(
          shape, src_tensor_dim_to_mesh_dim, dst_tensor_dim_to_mesh_dim,
          device_mesh_.num_elements(), mesh_alpha_, mesh_beta_);
    }

    AdjustTensorMeshDimMapping(src_tensor_dim_to_mesh_dim, src_n_dim);
    AdjustTensorMeshDimMapping(dst_tensor_dim_to_mesh_dim, dst_n_dim);

    // Analyze the dims that need to dynamic-sliced or all-gather.
    std::vector<int> slice_dims;
    std::vector<int> all_gather_dims;
    for (int64_t i = 0; i < std::min(src_rank, dst_rank); ++i) {
      int src_mesh_dim = src_tensor_dim_to_mesh_dim[i];
      int dst_mesh_dim = dst_tensor_dim_to_mesh_dim[i];
      if (src_mesh_dim == dst_mesh_dim) {
        continue;
      }
      if (src_mesh_dim == -1) {
        slice_dims.push_back(src_mesh_dim);
        continue;
      }
      if (dst_mesh_dim == -1) {
        all_gather_dims.push_back(src_mesh_dim);
        continue;
      }
      // Do not allow other re-sharding patterns. (e.g., collective-permute)
      return kInfinityCost;
    }

    // Case 1: no communication is required. Only needs dynamic-slice.
    if (all_gather_dims.empty()) {
      return 0;
    }

    // Do not allow some strange re-sharding patterns.
    if (slice_dims.size() > 1 && all_gather_dims.size() > 1) {
      return kInfinityCost;
    }

    // Case 2: all-to-all
    if (slice_dims.size() == 1 && all_gather_dims.size() == 1) {
      if (device_mesh_.dim(0) > 1 && device_mesh_.dim(1) > 1) {
        return kInfinityCost;
      }

      double bytes = GetBytes(shape);
      return AllToAllCost(bytes, all_gather_dims.front());
    }

    // Case 3: all-gather
    double bytes = GetBytes(shape) / src_spec.NumTiles();
    double cost = 0.0;
    for (int dim : all_gather_dims) {
      if (dim >= device_mesh_.num_dimensions()) {
        return kInfinityCost;
      }
      bytes *= device_mesh_.dim(dim);
      cost += AllGatherCost(bytes, dim);
    }
    return cost;
  }

  // Print the information of this device mesh.
  std::string ToString() {
    std::string str;
    absl::StrAppend(&str, "device_mesh: ", device_mesh_.ToString(), "\n");
    absl::StrAppend(&str, "mesh_alpha: ", absl::StrJoin(mesh_alpha_, " "),
                    "\n");
    absl::StrAppend(&str, "mesh_beta: ", absl::StrJoin(mesh_beta_, " "), "\n");
    return str;
  }

  // The original, complete device mesh shape that describes the hardware.
  const Array<int64_t> original_device_mesh_;
  // When solve_nd_sharding_iteratively is true, it is a partial mesh shape from
  // the original_device_mesh_. When solve_nd_sharding_iteratively is false, it
  // is the same as original_device_mesh_.
  const Array<int64_t> device_mesh_;
  // Bandwidth of the device mesh
  const std::vector<double> mesh_alpha_;
  const std::vector<double> mesh_beta_;
  const ProfilingResult& prof_result_;
  std::vector<int64_t> non_zero_mesh_dims_;
  const int total_devices_;

  // Cache a flatten 1d version of the device mesh.
  // Used for mixed mesh shape strategies.
  Array<int64_t> device_mesh_1d_;

  // The solver option may override the cost of communication primitives
  const AutoShardingSolverOption& solver_option_;

  // Cached replica groups. Shape: [mesh_dim, group_id, ids in this group].
  std::vector<std::vector<std::vector<int64_t>>> cached_replica_groups_;

 private:
  void GenerateCachedReplicaGroups() {
    // One vector per device_mesh_ dimension.
    cached_replica_groups_.reserve(device_mesh_.num_dimensions());
    for (size_t i = 0; i < device_mesh_.num_dimensions(); i++) {
      cached_replica_groups_.push_back(
          GetReplicaGroupsAlongOneDimension(device_mesh_, i));
    }
  }

  void AdjustTensorMeshDimMapping(std::vector<int64_t>& mapping,
                                  int64_t n_dim) const {
    // Shift the non-zero dim for 1d mesh
    if (n_dim == 1 && non_zero_mesh_dims_.size() == 1) {
      for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] == 0) {
          mapping[i] = non_zero_mesh_dims_.front();
        }
      }
    }
  }
};

// Function declarations
// Their comments can be found in their definitions in *.cc files.
HloSharding Tile(const Shape& shape, absl::Span<const int64_t> tensor_dims,
                 absl::Span<const int64_t> mesh_dims,
                 const Array<int64_t>& device_mesh);

std::vector<double> ReshardingCostVector(const StrategyVector* strategies,
                                         const Shape& shape,
                                         const HloSharding& required_sharding,
                                         const ClusterEnvironment& cluster_env);

std::vector<double> FollowInsCostVector(int64_t source_len, int64_t index);

std::unique_ptr<StrategyVector> CreateLeafStrategyVector(
    size_t instruction_id, const HloInstruction* ins,
    const StrategyMap& strategy_map, LeafStrategies& leaf_strategies);

void SetInNodesWithInstruction(std::unique_ptr<StrategyVector>& strategies,
                               const HloInstruction* ins,
                               const StrategyMap& strategy_map);

void RemoveDuplicatedStrategy(std::unique_ptr<StrategyVector>& strategies);

Status FilterStrategy(const HloInstruction* ins, const Shape& shape,
                      std::unique_ptr<StrategyVector>& strategies,
                      const ClusterEnvironment& cluster_env,
                      const InstructionBatchDimMap& batch_map,
                      const AutoShardingSolverOption& solver_option);

Status HandleDot(std::unique_ptr<StrategyVector>& strategies,
                 LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                 const HloInstruction* ins, size_t instruction_id,
                 const ClusterEnvironment& cluster_env,
                 const InstructionBatchDimMap& batch_map,
                 const AutoShardingSolverOption& solver_option);

Status HandleConv(std::unique_ptr<StrategyVector>& strategies,
                  LeafStrategies& leaf_strategies, StrategyMap& strategy_map,
                  const HloInstruction* ins, size_t instruction_id,
                  const ClusterEnvironment& cluster_env,
                  const InstructionBatchDimMap& batch_map,
                  const AutoShardingSolverOption& solver_option);

void AnnotateShardingWithSimpleHeuristic(HloModule* module,
                                         const std::string& heuristic,
                                         const AliasMap& alias_map,
                                         const ClusterEnvironment& cluster_env);

// Handle alias: alias pairs must have the same HloSharding.
// To deal with alias, we do special process both before and after
// BuildStrategyAndCost. Because it is easier to handle elementwise
// instructions before BuildStrategyAndCost and it is easier to handle
// dot/conv instructions after BuildStrategyAndCost. Before
// BuildStrategyAndCost, we build an AliasMap to guide the generation of
// strategies. After BuildStrategyAndCost, we use AliasSet to add alias
// constraints in the ILP problem.
AliasMap BuildAliasMap(const HloModule* module);

AliasSet BuildAliasSet(const HloModule* module,
                       const StrategyMap& strategy_map);

void CheckAliasSetCompatibility(const AliasSet& alias_set,
                                const LeafStrategies& leaf_strategies,
                                const HloInstructionSequence& sequence);
}  // namespace spmd
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_AUTO_SHARDING_STRATEGY_H_

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_CONFIG_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_CONFIG_H_

#include <string>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

enum class FusionConfigCollection {
  kOff,      // Do not collect configuration.
  kPerEdge,  // Collect per-edge configuration.
  kPerNode,  // Collect per-node configuration.
};

// This class gathers all settings and values which affect the compiled
// executable outside of the HLO code itself. This include layouts of inputs and
// outputs to the module and settings such as HLO profiling. Together the
// HloModule and HloModuleConfig unambiguously determine a particular
// executable.
class HloModuleConfig {
 public:
  // Represents a pair of input and output of the entry computation that can be
  // considered as the original and updated values of a variable maintained by
  // the caller, and that can be transparently sharded by XLA as an internal
  // optimization. If sharded, XLA will create separate sharding/unsharding
  // programs, and the caller is responsible to call the XLA-generated
  // sharding/unsharding programs before and after the sharded main program.
  //
  // If the variable is not updated and there is not a corresponding output, use
  // {-1} as the output_shape_index.
  //
  // The sharding/unsharding programs will include all the input/output pairs in
  // shardable_value_update_pairs() as a flat tuple in their inputs/outputs,
  // sorted by (input_parameter_number, parameter_shape_index).
  //
  // A typical usage pattern is to shard the variables first, then repeatedly
  // invoke the main program, and finally invoke the unsharding program before
  // they are used in full-shape.
  struct ShardableValueUpdatePair {
    int64 input_parameter_number;
    ShapeIndex parameter_shape_index;
    ShapeIndex output_shape_index;
  };

  // A configuration can be created either with, or without an entry
  // ComputationLayout. The default ctor creates it without -- in this case
  // accessing entry_computation_layout will CHECK-fail. The ctor accepting a
  // ProgramShape creates a computation layout using this shape.
  // The layouts in the ProgramShape will be reset to default unless
  // ignore_layouts is set to false.
  HloModuleConfig() = default;

  explicit HloModuleConfig(const ProgramShape& program_shape,
                           bool ignore_layouts = true);

  explicit HloModuleConfig(ComputationLayout entry_computation_layout);

  // Checks if this config has an entry computation layout already.
  bool has_entry_computation_layout() const {
    return entry_computation_layout_.has_value();
  }

  // Sets the entry_computation_layout's parameter and result shapes for this
  // config, according to the given program shape. The parameters and result
  // are set to default layout.
  void SetDefaultComputationLayout(const ProgramShape& program_shape);

  // Same as above but if the given program contains layout for parameters or
  // result, the entry_computation_layout's layout is updated accordingly.
  void SetComputationLayoutIfExists(const ProgramShape& program_shape);

  // Returns a constant reference to the layout of the entry computation.
  // Assumes the layout was set.
  const ComputationLayout& entry_computation_layout() const {
    CHECK(entry_computation_layout_.has_value());
    return *entry_computation_layout_;
  }

  // Returns a mutable pointer to the layout of the entry computation.
  // Assumes the layout was set.
  ComputationLayout* mutable_entry_computation_layout() {
    CHECK(entry_computation_layout_.has_value());
    return &(*entry_computation_layout_);
  }

  // Returns whether to enable HLO-level profiling.
  bool hlo_profiling_enabled() const {
    return debug_options_.xla_hlo_profile();
  }

  bool cpu_traceme_enabled() const {
    return debug_options_.xla_cpu_enable_xprof_traceme();
  }

  // Sets/returns the module seed set during execution.
  void set_seed(uint64 seed) { seed_ = seed; }
  uint64 seed() const { return seed_; }

  // Set the launch id of the program. Launch id identifies a set of programs
  // that should be launched together.
  void set_launch_id(uint64 launch_id) { launch_id_ = launch_id; }

  int32 launch_id() const { return launch_id_; }

  void set_replica_count(int64 replica_count) {
    replica_count_ = replica_count;
  }
  int64 replica_count() const { return replica_count_; }

  void set_num_partitions(int64 num_partitions) {
    num_partitions_ = num_partitions;
  }
  int64 num_partitions() const { return num_partitions_; }

  const std::vector<bool> param_requires_broadcast_via_collectives() const {
    return param_requires_broadcast_via_collectives_;
  }
  void set_param_requires_broadcast_via_collectives(
      const std::vector<bool> require_broadcast) {
    param_requires_broadcast_via_collectives_ = std::move(require_broadcast);
  }

  void set_use_spmd_partitioning(bool use_spmd_partitioning) {
    use_spmd_partitioning_ = use_spmd_partitioning;
  }
  bool use_spmd_partitioning() const { return use_spmd_partitioning_; }

  // If enabled, deduplicate equivalent hlos into function calls to reduce code
  // size.
  void set_deduplicate_hlo(bool deduplicate_hlo) {
    deduplicate_hlo_ = deduplicate_hlo;
  }
  bool deduplicate_hlo() const { return deduplicate_hlo_; }

  // Return a string which unambiguously represents all the fields of this data
  // structure. Used for generating a cache key for storing the compiled
  // executable.
  string compilation_cache_key() const;

  const DebugOptions& debug_options() const { return debug_options_; }

  void set_debug_options(const DebugOptions& debug_options) {
    debug_options_ = debug_options;
  }

  // Sets/returns the number of intra op threads for this module.
  void set_intra_op_parallelism_threads(
      const int intra_op_parallelism_threads) {
    intra_op_parallelism_threads_ = intra_op_parallelism_threads;
  }
  int64 intra_op_parallelism_threads() const {
    return intra_op_parallelism_threads_;
  }

  // Checks if this config has a static device assignment.
  bool has_static_device_assignment() const {
    return static_device_assignment_.has_value();
  }

  // Getter and setter of the compile-time known device assignment.
  const DeviceAssignment& static_device_assignment() const {
    CHECK(static_device_assignment_.has_value());
    return *static_device_assignment_;
  }
  void set_static_device_assignment(const DeviceAssignment& device_assignment) {
    static_device_assignment_ = device_assignment;
  }

  const std::vector<ShardableValueUpdatePair> shardable_value_update_pairs()
      const {
    return shardable_value_update_pairs_;
  }
  void set_shardable_value_update_pairs(
      std::vector<ShardableValueUpdatePair> pairs) {
    shardable_value_update_pairs_ = std::move(pairs);
  }

  // Whether input and output buffers are aliased if the associated parameter is
  // passed-through XLA modules without being changed.
  bool alias_passthrough_params() const { return alias_passthrough_params_; }
  void set_alias_passthrough_params(bool alias_passthrough_params) {
    alias_passthrough_params_ = alias_passthrough_params;
  }

  bool content_aware_computation_sorting() const {
    return content_aware_computation_sorting_;
  }
  void set_content_aware_computation_sorting(
      bool content_aware_computation_sorting) {
    content_aware_computation_sorting_ = content_aware_computation_sorting;
  }

  FusionConfigCollection fusion_config_collection() const {
    return fusion_config_collection_;
  }
  void set_fusion_config_collection(
      FusionConfigCollection fusion_config_collection) {
    fusion_config_collection_ = fusion_config_collection;
  }

  const std::vector<std::vector<bool>>& fusion_config() const {
    return fusion_config_;
  }
  std::vector<std::vector<bool>>* mutable_fusion_config() {
    return &fusion_config_;
  }

  const std::vector<std::vector<int64>>& dot_config() const {
    return dot_config_;
  }

  std::vector<std::vector<int64>>* mutable_dot_config() { return &dot_config_; }

  const std::vector<std::vector<std::vector<int64>>>& layout_config() const {
    return layout_config_;
  }

  std::vector<std::vector<std::vector<int64>>>* mutable_layout_config() {
    return &layout_config_;
  }

 private:
  // If you add new members, be sure to update compilation_cache_key.

  absl::optional<ComputationLayout> entry_computation_layout_;

  // Module/graph-level seed handle.
  uint64 seed_ = 0;

  // Program id that identifies a set of program to be launched together.
  int32 launch_id_ = 0;

  // The number of replicas (data parallelism) to compile this binary for.
  int64 replica_count_ = 1;

  // The number of partitions (model parallelism) to compile this binary for.
  int64 num_partitions_ = 1;

  // Whether to broadcast args across all replicas. One entry per arg.
  std::vector<bool> param_requires_broadcast_via_collectives_;

  // Whether to use SPMD (true) or MPMD (false) when num_partitions_ > 0 and XLA
  // needs to partition the module.
  bool use_spmd_partitioning_ = false;

  // If enabled, deduplicate equivalent hlos into function calls to reduce code
  // size.
  bool deduplicate_hlo_ = false;

  // The target maximum parallelism at which to partition HLOs for parallel
  // execution on the CPU backend.
  int64 intra_op_parallelism_threads_ = -1;

  DebugOptions debug_options_;

  // Compile-time known device assignment.
  absl::optional<DeviceAssignment> static_device_assignment_;

  std::vector<ShardableValueUpdatePair> shardable_value_update_pairs_;

  bool alias_passthrough_params_ = false;

  bool content_aware_computation_sorting_ = false;

  FusionConfigCollection fusion_config_collection_ =
      FusionConfigCollection::kOff;

  // TODO(b/155665133): Consolidate fusion, dot, and layout config into a proto
  // similar to backend config.

  // Custom fusion configuration, where fusion_config_[c][v] control if node v
  // in computation c must be fused to all its consumers (true) or not (false).
  std::vector<std::vector<bool>> fusion_config_;

  // Custom dot canonicalization configuration, where dot_config_[v] control
  // how to convert dot operation v (sorted topologically and by computation) to
  // convolution.
  std::vector<std::vector<int64>> dot_config_;

  // Layout configuration, where layout_config_[v][i] controls the layout
  // decision i of operation v.
  std::vector<std::vector<std::vector<int64>>> layout_config_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_MODULE_CONFIG_H_

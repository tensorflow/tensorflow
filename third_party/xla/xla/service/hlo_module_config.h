/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_MODULE_CONFIG_H_
#define XLA_SERVICE_HLO_MODULE_CONFIG_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/schedule_config.h"
#include "xla/service/sharding_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

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
    int64_t input_parameter_number;
    ShapeIndex parameter_shape_index;
    ShapeIndex output_shape_index;
  };

  // A configuration can be created either with, or without an entry
  // ComputationLayout. The default ctor creates it without -- in this case
  // accessing entry_computation_layout will CHECK-fail. The ctor accepting a
  // ProgramShape creates a computation layout using this shape.
  // The layouts in the ProgramShape will be reset to default unless
  // ignore_layouts is set to false.
  HloModuleConfig() { debug_options_ = DefaultDebugOptionsIgnoringFlags(); }

  explicit HloModuleConfig(const ProgramShape& program_shape,
                           bool ignore_layouts = true);

  explicit HloModuleConfig(ComputationLayout entry_computation_layout);

  // Convert an HloModuleConfig to or from a proto.
  HloModuleConfigProto ToProto() const;
  static absl::StatusOr<std::unique_ptr<HloModuleConfig>> CreateFromProto(
      const HloModuleConfigProto& proto);

  // Assigns the repeated ShardableValueUpdatePairProto field to the given
  // values in 'update_pairs'.
  static void AssignProtoShardableValueUpdatePairs(
      tsl::protobuf::RepeatedPtrField<ShardableValueUpdatePairProto>*
          proto_update_pairs,
      const std::vector<HloModuleConfig::ShardableValueUpdatePair>&
          update_pairs);

  // Assigns shardable_value_update_pairs_ field in 'config' to the given values
  // in 'pairs'.
  static void AssignStructShardableValueUpdatePairs(
      HloModuleConfig& config,
      const tsl::protobuf::RepeatedPtrField<ShardableValueUpdatePairProto>&
          pairs);

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

  // Clears the entry computation layout.
  void clear_entry_computation_layout() {
    entry_computation_layout_ = std::nullopt;
  }

  // Returns whether to enable HLO-level profiling.
  bool hlo_profiling_enabled() const {
    return debug_options_.xla_hlo_profile();
  }

  bool cpu_traceme_enabled() const {
    return debug_options_.xla_cpu_enable_xprof_traceme();
  }

  // Sets/returns the module seed set during execution.
  void set_seed(uint64_t seed) { seed_ = seed; }
  uint64_t seed() const { return seed_; }

  // Set the launch id of the program. Launch id identifies a set of programs
  // that should be launched together.
  void set_launch_id(uint64_t launch_id) { launch_id_ = launch_id; }
  int32_t launch_id() const { return launch_id_; }

  void set_replica_count(int64_t replica_count) {
    replica_count_ = replica_count;
  }
  int64_t replica_count() const { return replica_count_; }

  void set_num_partitions(int64_t num_partitions) {
    num_partitions_ = num_partitions;
  }
  int64_t num_partitions() const { return num_partitions_; }

  const std::vector<bool>& param_requires_broadcast_via_collectives() const {
    return param_requires_broadcast_via_collectives_;
  }
  void set_param_requires_broadcast_via_collectives(
      std::vector<bool> require_broadcast) {
    param_requires_broadcast_via_collectives_ = std::move(require_broadcast);
  }

  void set_use_spmd_partitioning(bool use_spmd_partitioning) {
    use_spmd_partitioning_ = use_spmd_partitioning;
  }
  bool use_spmd_partitioning() const { return use_spmd_partitioning_; }

  void set_use_auto_spmd_partitioning(bool use_auto_spmd_partitioning) {
    use_auto_spmd_partitioning_ = use_auto_spmd_partitioning;
    if (use_auto_spmd_partitioning) {
      // TODO(yuemmawang) Remove this warning once auto sharding is thoroughly
      // tested with fleetwide models.
      LOG(WARNING) << "Warning: Using auto_spmd_partitioning. It is "
                      "experimental and may contain bugs!";
      LOG(INFO) << "Overwriting use_spmd_partitioning to true, because "
                   "use_auto_spmd_partitioning is true.";
      set_use_spmd_partitioning(true);
    }
  }
  bool use_auto_spmd_partitioning() const {
    return use_auto_spmd_partitioning_;
  }

  void set_auto_spmd_partitioning_mesh_shape(std::vector<int64_t> mesh_shape) {
    auto_spmd_partitioning_mesh_shape_ = std::move(mesh_shape);
  }
  const std::vector<int64_t>& auto_spmd_partitioning_mesh_shape() const {
    return auto_spmd_partitioning_mesh_shape_;
  }

  void set_auto_spmd_partitioning_mesh_ids(std::vector<int64_t> mesh_ids) {
    auto_spmd_partitioning_mesh_ids_ = std::move(mesh_ids);
  }
  const std::vector<int64_t>& auto_spmd_partitioning_mesh_ids() const {
    return auto_spmd_partitioning_mesh_ids_;
  }

  void set_exec_time_optimization_effort(float exec_time_optimization_effort) {
    exec_time_optimization_effort_ = exec_time_optimization_effort;
  }
  float exec_time_optimization_effort() const {
    return exec_time_optimization_effort_;
  }

  void set_memory_fitting_effort(float memory_fitting_effort) {
    memory_fitting_effort_ = memory_fitting_effort;
  }
  float memory_fitting_effort() const { return memory_fitting_effort_; }

  void set_optimization_level(
      ExecutionOptions::EffortLevel optimization_level) {
    optimization_level_ = optimization_level;
  }
  ExecutionOptions::EffortLevel optimization_level() const {
    return optimization_level_;
  }

  void set_memory_fitting_level(
      ExecutionOptions::EffortLevel memory_fitting_level) {
    memory_fitting_level_ = memory_fitting_level;
  }
  ExecutionOptions::EffortLevel memory_fitting_level() const {
    return memory_fitting_level_;
  }

  // If enabled, deduplicate equivalent hlos into function calls to reduce code
  // size.
  void set_deduplicate_hlo(bool deduplicate_hlo) {
    deduplicate_hlo_ = deduplicate_hlo;
  }
  bool deduplicate_hlo() const { return deduplicate_hlo_; }

  void set_device_type(const std::string& device_type) {
    device_type_ = device_type;
  }
  absl::string_view device_type() const { return device_type_; }

  // Return a string which unambiguously represents all the fields of this data
  // structure. Used for generating a cache key for storing the compiled
  // executable.
  std::string compilation_cache_key() const;

  const DebugOptions& debug_options() const { return debug_options_; }
  DebugOptions& mutable_debug_options() { return debug_options_; }
  void set_debug_options(const DebugOptions& debug_options) {
    debug_options_ = debug_options;
  }

  // Sets/returns the number of intra op threads for this module.
  void set_intra_op_parallelism_threads(
      const int intra_op_parallelism_threads) {
    intra_op_parallelism_threads_ = intra_op_parallelism_threads;
  }
  int64_t intra_op_parallelism_threads() const {
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
  void reset_static_device_assignment() {
    static_device_assignment_ = std::nullopt;
  }

  // Checks if this config has a simulated device assignment.
  bool has_pre_simulation_device_assignment() const {
    return pre_simulation_device_assignment_.has_value();
  }

  // Getter and setter of the compile-time known device assignment.
  const DeviceAssignment& pre_simulation_device_assignment() const {
    CHECK(pre_simulation_device_assignment_.has_value());
    return *pre_simulation_device_assignment_;
  }

  void set_pre_simulation_device_assignment(
      const DeviceAssignment& device_assignment) {
    pre_simulation_device_assignment_ = device_assignment;
  }

  bool allow_separate_sharding_programs() const {
    return allow_separate_sharding_programs_;
  }
  void set_allow_separate_sharding_programs(
      bool allow_separate_sharding_programs) {
    allow_separate_sharding_programs_ = allow_separate_sharding_programs;
  }

  const std::vector<ShardableValueUpdatePair>& shardable_value_update_pairs()
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
  void set_fusion_config(std::vector<std::vector<bool>> fusion_config) {
    fusion_config_ = std::move(fusion_config);
  }
  std::vector<std::vector<bool>>& mutable_fusion_config() {
    return fusion_config_;
  }

  const absl::flat_hash_map<std::string, std::vector<int64_t>>& dot_config()
      const {
    return dot_config_;
  }
  absl::flat_hash_map<std::string, std::vector<int64_t>>* mutable_dot_config() {
    return &dot_config_;
  }

  const std::vector<std::vector<std::vector<int64_t>>>& layout_config() const {
    return layout_config_;
  }
  std::vector<std::vector<std::vector<int64_t>>>* mutable_layout_config() {
    return &layout_config_;
  }

  const std::vector<std::vector<bool>>& phase_ordering_config() const {
    return phase_ordering_config_;
  }
  void set_phase_ordering_config(
      std::vector<std::vector<bool>> phase_ordering_config) {
    phase_ordering_config_ = std::move(phase_ordering_config);
  }
  std::vector<std::vector<bool>>& mutable_phase_ordering_config() {
    return phase_ordering_config_;
  }

  const ShardingConfig& sharding_config() const { return sharding_config_; }
  ShardingConfig* mutable_sharding_config() { return &sharding_config_; }

  const ScheduleConfig& schedule_config() const { return schedule_config_; }
  ScheduleConfig* mutable_schedule_config() { return &schedule_config_; }

  int phase_index() const { return phase_index_; }
  void set_phase_index(const int phase_index) { phase_index_ = phase_index; }

  absl::Span<const bool> allow_spmd_sharding_propagation_to_parameters() const {
    return allow_spmd_sharding_propagation_to_parameters_;
  }
  absl::Span<const bool> allow_spmd_sharding_propagation_to_output() const {
    return allow_spmd_sharding_propagation_to_output_;
  }
  void set_allow_spmd_sharding_propagation_to_parameters(
      absl::Span<const bool> data) {
    return allow_spmd_sharding_propagation_to_parameters_.assign(data.begin(),
                                                                 data.end());
  }
  void set_allow_spmd_sharding_propagation_to_output(
      absl::Span<const bool> data) {
    return allow_spmd_sharding_propagation_to_output_.assign(data.begin(),
                                                             data.end());
  }

  const std::vector<uint64_t>& memory_space_assignment_config() const {
    return memory_space_assignment_config_;
  }
  std::vector<uint64_t>* mutable_memory_space_assignment_config() {
    return &memory_space_assignment_config_;
  }

  int64_t GetAnalysisAllowance(absl::string_view pass_name) const {
    auto it = analysis_allowance_map_.find(pass_name);
    if (it == analysis_allowance_map_.end()) {
      return -1;
    }
    return (*it).second;
  }
  void SetAnalysisAllowance(absl::string_view pass_name, int64_t allowance) {
    analysis_allowance_map_[pass_name] = allowance;
  }

  PrecisionConfig::Precision matrix_unit_operand_precision() const {
    return matrix_unit_operand_precision_;
  }
  void set_matrix_unit_operand_precision(
      PrecisionConfig::Precision matrix_unit_operand_precision) {
    matrix_unit_operand_precision_ = matrix_unit_operand_precision;
  }

  absl::string_view fdo_profile() const { return fdo_profile_; }
  void set_fdo_profile(absl::string_view fdo_profile) {
    fdo_profile_ = fdo_profile;
  }

  int64_t device_memory_size() const { return device_memory_size_; }
  void set_device_memory_size(int64_t device_memory_size) {
    device_memory_size_ = device_memory_size;
  }

  bool use_shardy_partitioner() const { return use_shardy_partitioner_; }
  void set_use_shardy_partitioner(bool use_shardy_partitioner) {
    use_shardy_partitioner_ = use_shardy_partitioner;
  }

 private:
  // If you add new members, be sure to update compilation_cache_key and the
  // HloModuleConfigProto.
  // LINT.IfChange
  std::optional<ComputationLayout> entry_computation_layout_;

  // Module/graph-level seed handle.
  uint64_t seed_ = 0;

  // Program id that identifies a set of program to be launched together.
  int32_t launch_id_ = 0;

  // The number of replicas (data parallelism) to compile this binary for.
  int64_t replica_count_ = 1;

  // The number of partitions (model parallelism) to compile this binary for.
  int64_t num_partitions_ = 1;

  // Whether to broadcast args across all replicas. One entry per arg.
  std::vector<bool> param_requires_broadcast_via_collectives_;

  // Whether to use SPMD (true) or MPMD (false) when num_partitions_ > 0 and XLA
  // needs to partition the module.
  bool use_spmd_partitioning_ = false;

  // Whether to automatically generate XLA shardings for SPMD partitioner.
  bool use_auto_spmd_partitioning_ = false;

  // Mesh shape and mesh ids used by auto spmd partitioning.
  std::vector<int64_t> auto_spmd_partitioning_mesh_shape_;

  std::vector<int64_t> auto_spmd_partitioning_mesh_ids_;

  // The amount of effort to spend on optimizing for minimizing program
  // execution time, as a value in [-1.0, +1.0]. The baseline is 0.0, which
  // strongly prioritizes execution time at the cost of longer compile times,
  // suitable for production workloads. A value of -0.5 would be appropriate for
  // research use cases that prefer faster compilations to iterate more quickly.
  // Positive values, on the other hand, might enable costly optimizations that
  // are off by default.
  float exec_time_optimization_effort_ = 0.0f;

  // The amount of effort to spend on making the program fit in memory (where
  // "fit in memory" here has a backend-dependent meaning), as a value in [-1.0,
  // +1.0]. The baseline is 0.0, which expends significant effort on attempting
  // to make the program fit. A value of -1.0 would be appropriate for use cases
  // that wish to spend minimal effort here and fail as quickly as possible
  // instead. Positive values, on the other hand, might enable costly algorithms
  // to reduce memory usage that are off by default.
  float memory_fitting_effort_ = 0.0f;

  // The amount of effort to spend on optimizing for minimizing program
  // execution time. As a general guideline, O2 strongly prioritizes execution
  // time, and is typically suitable for production workloads. O3 may enable
  // costly or experimental optimizations that may greatly increase compile
  // time.
  ExecutionOptions::EffortLevel optimization_level_ =
      ExecutionOptions::EFFORT_UNKNOWN;

  // The amount of effort to spend on making the program fit in memory (where
  // "fit in memory" here has a backend-dependent meaning). As a general
  // guideline, O2 will expend significant effort on attempting to make the
  // program fit. O0 will spend minimal effort and fail as quickly as possible
  // instead. O3 might enable costly algorithms to reduce memory usage that may
  // greatly increase compile time.
  ExecutionOptions::EffortLevel memory_fitting_level_ =
      ExecutionOptions::EFFORT_O2;

  // If enabled, deduplicate equivalent hlos into function calls to reduce code
  // size.
  bool deduplicate_hlo_ = false;

  // The target maximum parallelism at which to partition HLOs for parallel
  // execution on the CPU backend.
  int64_t intra_op_parallelism_threads_ = -1;

  std::string device_type_;

  DebugOptions debug_options_;

  // Compile-time known device assignment.
  std::optional<DeviceAssignment> static_device_assignment_;

  // Compile-time known device assignment.
  std::optional<DeviceAssignment> pre_simulation_device_assignment_;

  bool allow_separate_sharding_programs_ = false;

  std::vector<ShardableValueUpdatePair> shardable_value_update_pairs_;

  bool alias_passthrough_params_ = false;

  bool content_aware_computation_sorting_ = false;

  FusionConfigCollection fusion_config_collection_ =
      FusionConfigCollection::kOff;

  // Custom fusion configuration, where fusion_config_[c][v] control if node v
  // in computation c must be fused to all its consumers (true) or not (false).
  std::vector<std::vector<bool>> fusion_config_;

  // Custom dot canonicalization configuration, where dot_config_[v] control
  // how to convert dot operation named 'v' to convolution.
  absl::flat_hash_map<std::string, std::vector<int64_t>> dot_config_;

  // Layout configuration, where layout_config_[v][i] controls the layout
  // decision i of operation v.
  std::vector<std::vector<std::vector<int64_t>>> layout_config_;

  // Memory Space Assignment configuration, where
  // memory_space_assignment_config_ controls the order of buffer intervals
  // of this hlo module.
  std::vector<uint64_t> memory_space_assignment_config_;

  // Phase ordering configuration, where phase_ordering_config[v][i] controls
  // whether a specific pass with index i (e.g. 0 = DCE, 1 = CSE, etc.) is
  // inserted after pass v in pipeline. See tuning::PhaseOrderingConfig for
  // details on what indices (i) correspond to which passes.
  std::vector<std::vector<bool>> phase_ordering_config_;
  // Index (v) corresponding to current passes being added for phase ordering.
  // This is the variable that stores state to allow us to use the same
  // config across functions during compilation.
  int phase_index_ = 0;

  // Allows sharding propagation to propagate to the parameters. This changes
  // the input shape of the computation (which is undesirable), but it can be
  // used to allow to run partial compilation to determine what would be the
  // input sharding of a computation if XLA would be allowed to propagate the
  // sharding which can be used by higher level framework as a way to query
  // intermediate sharding of operations when multiple computation would be
  // chained and merged together.
  // This is a vector of bool, because the user can control which parameters can
  // have the sharding substituted. If only one boolean value is passed in the
  // vector that is interpreted as the value to be applied for every parameter.
  absl::InlinedVector<bool, 1> allow_spmd_sharding_propagation_to_parameters_ =
      {false};
  // Allows sharding propagation to propagate to the outputs. This changes the
  // output shape of the computation (which is undesirable), but it can be used
  // to allow to run partial compilation to determine what would be the output
  // sharding of a computation if XLA would be allowed to propagate the sharding
  // which can be used by higher level framework as a way to query intermediate
  // sharding of operations when multiple computation would be chained and
  // merged together.
  // Each boolean in the vector specifies if the propagation is allowed to
  // change the sharding of a specific leaf in tuple output. One single boolean
  // in the vector means we are applying this to every value in the tuple
  // output. If the output is not a tuple then only a single value is valid
  // here.
  absl::InlinedVector<bool, 1> allow_spmd_sharding_propagation_to_output_ = {
      false};

  // Each Hlo analysis is allowed at least a constant number of
  // abstract cost units, before it is considered for early termination.
  absl::flat_hash_map<std::string, int64_t> analysis_allowance_map_;

  PrecisionConfig::Precision matrix_unit_operand_precision_ =
      PrecisionConfig::DEFAULT;

  // Profiling data for feedback directed optimizations. Note that this is not
  // the only way to feed FDO data into the compiler and individual backends
  // may choose to get FDO data by other means.
  std::string fdo_profile_;

  int64_t device_memory_size_ = 0;

  bool use_shardy_partitioner_ = false;

  // Sharding configuration, where sharding_config_.nodes[v] controls the
  // sharding of operation v.
  ShardingConfig sharding_config_;

  // Schedule configuration, where schedule_config_.sequence is the sequence of
  // instructions to be scheduled.
  ScheduleConfig schedule_config_;

  // LINT.ThenChange(//tensorflow/compiler/xla/xla.proto)
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_MODULE_CONFIG_H_

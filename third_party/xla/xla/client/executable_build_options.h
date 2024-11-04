/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
#define XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace stream_executor {

// Forward-declared to avoid StreamExecutor dependency.
class DeviceMemoryAllocator;

}  // namespace stream_executor

namespace xla {
class HloModule;

// Class containing options for building an LocalExecutable with
// LocalClient::Compile.
class ExecutableBuildOptions {
 public:
  // If set, this is the device to build the computation for. Valid
  // device_ordinal values are: 0 to # of devices - 1. These values are
  // identical to the device ordinal values used by StreamExecutor. The built
  // executable will be executable on any device equivalent to the specified
  // device as determined by Backend::devices_equivalent(). A value of -1
  // indicates this option has not been set.
  ExecutableBuildOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this specifies the layout of the result of the computation. If not
  // set, the service will chose the layout of the result. A Shape is used to
  // store the layout to accommodate tuple result shapes. A value of nullptr
  // indicates the option has not been set.
  ExecutableBuildOptions& set_result_layout(const Shape& shape_with_layout);
  const Shape* result_layout() const;

  // Expose access to the XLA compilation environments, which will be passed to
  // the compilation process. `comp_envs()` must not be called if
  // `has_comp_envs()` returns false.
  bool has_comp_envs() const { return comp_envs_.has_value(); }
  const CompilationEnvironments& comp_envs() const { return *comp_envs_; }
  CompilationEnvironments* mutable_comp_envs();

  // Expose access to the XLA debug options which will be passed to the
  // compilation process. `debug_options()` must not be called if
  // `has_debug_options()` returns false.
  bool has_debug_options() const { return debug_options_.has_value(); }
  const DebugOptions& debug_options() const { return *debug_options_; }
  DebugOptions* mutable_debug_options();

  // If set, this specifies an allocator that can be used to allocate temporary
  // space on the device during compilation.  For example, the compiler might
  // want to run various algorithms on the device and pick the fastest one -- it
  // might allocate buffers for use by these algorithms using this allocator.
  //
  // This does not need to be the same as the se::DeviceMemoryAllocator passed
  // when running the executable.
  ExecutableBuildOptions& set_device_allocator(
      se::DeviceMemoryAllocator* allocator);
  se::DeviceMemoryAllocator* device_allocator() const;

  // The number of replicas of this computation that are to be executed.
  // Defaults to 1.
  int num_replicas() const { return num_replicas_; }
  ExecutableBuildOptions& set_num_replicas(int num_replicas);

  // The number of partitions in this computation. Defaults to 1.
  int num_partitions() const { return num_partitions_; }
  ExecutableBuildOptions& set_num_partitions(int num_partitions);

  // Indicates whether to use SPMD (true) or MPMD (false) partitioning when
  // num_partitions > 1 and XLA is requested to partition the input program.
  bool use_spmd_partitioning() const { return use_spmd_partitioning_; }
  ExecutableBuildOptions& set_use_spmd_partitioning(bool use_spmd_partitioning);

  // Whether to automatically generate XLA shardings for SPMD partitioner.
  bool use_auto_spmd_partitioning() const {
    return use_auto_spmd_partitioning_;
  }
  ExecutableBuildOptions& set_use_auto_spmd_partitioning(
      bool use_auto_spmd_partitioning);

  std::vector<int64_t> auto_spmd_partitioning_mesh_shape() const {
    return auto_spmd_partitioning_mesh_shape_;
  }
  ExecutableBuildOptions& set_auto_spmd_partitioning_mesh_shape(
      std::vector<int64_t> mesh_shape);

  std::vector<int64_t> auto_spmd_partitioning_mesh_ids() const {
    return auto_spmd_partitioning_mesh_ids_;
  }
  ExecutableBuildOptions& set_auto_spmd_partitioning_mesh_ids(
      std::vector<int64_t> mesh_ids);

  float exec_time_optimization_effort() const {
    return exec_time_optimization_effort_;
  }
  ExecutableBuildOptions& set_exec_time_optimization_effort(
      float exec_time_optimization_effort) {
    exec_time_optimization_effort_ = exec_time_optimization_effort;
    return *this;
  }

  float memory_fitting_effort() const { return memory_fitting_effort_; }
  ExecutableBuildOptions& set_memory_fitting_effort(
      float memory_fitting_effort) {
    memory_fitting_effort_ = memory_fitting_effort;
    return *this;
  }

  bool deduplicate_hlo() const { return deduplicate_hlo_; }
  ExecutableBuildOptions& set_deduplicate_hlo(bool deduplicate_hlo);

  // If set, this specifies a static device assignment for the computation.
  // Otherwise, the computation will be compiled generically and can be run with
  // any device assignment compatible with the computation's replica and
  // partition counts.
  bool has_device_assignment() const { return device_assignment_.has_value(); }
  ExecutableBuildOptions& set_device_assignment(
      const DeviceAssignment& device_assignment);
  const DeviceAssignment& device_assignment() const {
    CHECK(device_assignment_.has_value());
    return device_assignment_.value();
  }
  void clear_device_assignment() { device_assignment_.reset(); }

  // Whether input and output buffers are aliased if the associated parameter is
  // passed-through XLA modules without being changed.
  bool alias_passthrough_params() const { return alias_passthrough_params_; }
  void set_alias_passthrough_params(bool alias_passthrough_params) {
    alias_passthrough_params_ = alias_passthrough_params;
  }

  bool run_backend_only() const { return run_backend_only_; }
  // By default, XLA builds an executable by invoking standard compilation, i.e,
  // running Compiler::Compile, or both Compiler::RunHloPasses and
  // Compiler::RunBackend. When run_backend_only is set to true, XLA builds an
  // executable by invoking only RunBackend and skip invoking RunHloPasses,
  // which can be used to compile post-optimizations HLO modules.
  ExecutableBuildOptions& set_run_backend_only(bool run_backend_only) {
    run_backend_only_ = run_backend_only;
    return *this;
  }

  absl::Span<const bool> allow_spmd_sharding_propagation_to_parameters() const {
    return allow_spmd_sharding_propagation_to_parameters_;
  }
  absl::Span<const bool> allow_spmd_sharding_propagation_to_output() const {
    return allow_spmd_sharding_propagation_to_output_;
  }
  bool any_allow_spmd_sharding_propagation_to_parameters() const {
    return absl::c_linear_search(allow_spmd_sharding_propagation_to_parameters_,
                                 true);
  }
  bool any_allow_spmd_sharding_propagation_to_output() const {
    return absl::c_linear_search(allow_spmd_sharding_propagation_to_output_,
                                 true);
  }
  // Allows sharding propagation to propagate to the inputs. This changes the
  // input shape of the computation (which is undesirable), but it can be used
  // to allow to run partial compilation to determine what would be the input
  // sharding of a computation if XLA would be allowed to propagate the sharding
  // which can be used by higher level framework as a way to query intermediate
  // sharding of operations when multiple computation would be chained and
  // merged together.
  ExecutableBuildOptions& set_allow_spmd_sharding_propagation_to_parameters(
      absl::Span<const bool> allow_spmd_sharding_propagation_to_parameters) {
    allow_spmd_sharding_propagation_to_parameters_.assign(
        allow_spmd_sharding_propagation_to_parameters.begin(),
        allow_spmd_sharding_propagation_to_parameters.end());
    return *this;
  }
  // Allows sharding propagation to propagate to the outputs. This changes the
  // output shape of the computation (which is undesirable), but it can be used
  // to allow to run partial compilation to determine what would be the output
  // sharding of a computation if XLA would be allowed to propagate the sharding
  // which can be used by higher level framework as a way to query intermediate
  // sharding of operations when multiple computation would be chained and
  // merged together.
  ExecutableBuildOptions& set_allow_spmd_sharding_propagation_to_output(
      absl::Span<const bool> allow_spmd_sharding_propagation_to_output) {
    allow_spmd_sharding_propagation_to_output_.assign(
        allow_spmd_sharding_propagation_to_output.begin(),
        allow_spmd_sharding_propagation_to_output.end());
    return *this;
  }

  // Thread pool for parallel compilation.
  tsl::thread::ThreadPool* compile_thread_pool() const {
    return compile_thread_pool_;
  }
  ExecutableBuildOptions& set_compile_thread_pool(
      tsl::thread::ThreadPool* compile_thread_pool) {
    compile_thread_pool_ = compile_thread_pool;
    return *this;
  }

  using LayoutCanonicalizationCallback =
      std::function<absl::StatusOr<std::pair<std::vector<Shape>, Shape>>(
          const HloModule& module)>;
  void set_layout_canonicalization_callback(
      LayoutCanonicalizationCallback callback) {
    layout_canonicalization_callback_ = std::move(callback);
  }
  LayoutCanonicalizationCallback layout_canonicalization_callback() const {
    return layout_canonicalization_callback_;
  }

  absl::string_view fdo_profile() const { return fdo_profile_; }
  void set_fdo_profile(std::string fdo_profile) {
    fdo_profile_ = std::move(fdo_profile);
  }
  std::string* mutable_fdo_profile() { return &fdo_profile_; }

  // The amount of device memory available for the executable.
  int64_t device_memory_size() const { return device_memory_size_; }
  ExecutableBuildOptions& set_device_memory_size(int64_t device_memory_size) {
    device_memory_size_ = device_memory_size;
    return *this;
  }

  bool use_shardy_partitioner() const { return use_shardy_partitioner_; }
  ExecutableBuildOptions& set_use_shardy_partitioner(
      bool use_shardy_partitioner) {
    use_shardy_partitioner_ = use_shardy_partitioner;
    return *this;
  }

  // Returns a string representation of the build options, suitable for
  // debugging.
  std::string ToString() const;

  absl::StatusOr<ExecutableBuildOptionsProto> ToProto() const;

  int process_index() const { return process_index_; }
  void set_process_index(const int process_index) {
    process_index_ = process_index;
  }
  int process_count() const { return process_count_; }
  void set_process_count(const int process_count) {
    process_count_ = process_count;
  }

  std::shared_ptr<KeyValueStoreInterface> key_value_store() const {
    return key_value_store_;
  }
  void set_key_value_store(std::shared_ptr<KeyValueStoreInterface> kv_store) {
    key_value_store_ = kv_store;
  }

 private:
  int device_ordinal_ = -1;
  Shape result_layout_;
  bool result_layout_set_ = false;
  std::optional<CompilationEnvironments> comp_envs_;
  std::optional<DebugOptions> debug_options_;
  se::DeviceMemoryAllocator* device_allocator_ = nullptr;
  int num_replicas_ = 1;
  int num_partitions_ = 1;
  bool use_spmd_partitioning_ = false;
  bool use_auto_spmd_partitioning_ = false;
  std::vector<int64_t> auto_spmd_partitioning_mesh_shape_;
  std::vector<int64_t> auto_spmd_partitioning_mesh_ids_;
  float exec_time_optimization_effort_ = 0.0f;
  float memory_fitting_effort_ = 0.0f;
  bool deduplicate_hlo_ = false;
  bool broadcast_replicated_params_ = false;
  std::optional<DeviceAssignment> device_assignment_;
  bool alias_passthrough_params_ = false;
  bool run_backend_only_ = false;
  absl::InlinedVector<bool, 1> allow_spmd_sharding_propagation_to_parameters_ =
      {false};
  absl::InlinedVector<bool, 1> allow_spmd_sharding_propagation_to_output_ = {
      false};
  tsl::thread::ThreadPool* compile_thread_pool_ = nullptr;
  LayoutCanonicalizationCallback layout_canonicalization_callback_;
  std::string fdo_profile_;
  int64_t device_memory_size_ = 0;
  bool use_shardy_partitioner_ = false;
  int process_index_ = 0;
  int process_count_ = 1;
  std::shared_ptr<KeyValueStoreInterface> key_value_store_;
};

absl::StatusOr<ExecutableBuildOptions> ExecutableBuildOptionsFromProto(
    const ExecutableBuildOptionsProto& input);

// Creates an ExecutionOptions based on a given ExecutableBuildOptions and
// ProgramShape.
ExecutionOptions CreateExecutionOptions(
    const ExecutableBuildOptions& build_options,
    const ProgramShape* program_shape);

}  // namespace xla

#endif  // XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_

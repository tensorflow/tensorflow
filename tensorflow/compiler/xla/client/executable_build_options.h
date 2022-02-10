/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/threadpool.h"

namespace stream_executor {

// Forward-declared to avoid StreamExecutor dependency.
class DeviceMemoryAllocator;

}  // namespace stream_executor

namespace xla {

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

  // Expose access to the XLA debug options which will be passed to the
  // compilation process.
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

  // Returns a string representation of the build options, suitable for
  // debugging.
  std::string ToString() const;

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

  bool allow_spmd_sharding_propagation_to_output() const {
    return allow_spmd_sharding_propagation_to_output_;
  }
  // Allows sharding propagation to propagate to the outputs. This changes the
  // output shape of the computation (which is undesirable), but it can be used
  // to allow to run partial compilation to determine what would be the output
  // sharding of a computation if XLA would be allowed to propagate the sharding
  // which can be used by higher level framework as a way to query intermediate
  // sharding of operations when multiple computation would be chained and
  // merged together.
  ExecutableBuildOptions& set_allow_spmd_sharding_propagation_to_output(
      bool allow_spmd_sharding_propagation_to_output) {
    allow_spmd_sharding_propagation_to_output_ =
        allow_spmd_sharding_propagation_to_output;
    return *this;
  }

  // Thread pool for parallel compilation.
  tensorflow::thread::ThreadPool* compile_thread_pool() const {
    return compile_thread_pool_;
  }
  ExecutableBuildOptions& set_compile_thread_pool(
      tensorflow::thread::ThreadPool* compile_thread_pool) {
    compile_thread_pool_ = compile_thread_pool;
    return *this;
  }

 private:
  int device_ordinal_ = -1;
  Shape result_layout_;
  bool result_layout_set_ = false;
  absl::optional<DebugOptions> debug_options_;
  se::DeviceMemoryAllocator* device_allocator_ = nullptr;
  int num_replicas_ = 1;
  int num_partitions_ = 1;
  bool use_spmd_partitioning_ = false;
  bool deduplicate_hlo_ = false;
  bool broadcast_replicated_params_ = false;
  absl::optional<DeviceAssignment> device_assignment_;
  bool alias_passthrough_params_ = false;
  bool run_backend_only_ = false;
  bool allow_spmd_sharding_propagation_to_output_ = false;
  tensorflow::thread::ThreadPool* compile_thread_pool_ = nullptr;
};

// Creates an ExecutionOptions based on a given ExecutableBuildOptions and
// ProgramShape.
ExecutionOptions CreateExecutionOptions(
    const ExecutableBuildOptions& build_options,
    const ProgramShape* program_shape);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_EXECUTABLE_BUILD_OPTIONS_H_

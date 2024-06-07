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

#include "xla/client/executable_build_options.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xla/debug_options_flags.h"
#include "xla/execution_options_util.h"
#include "xla/layout_util.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

ExecutableBuildOptions& ExecutableBuildOptions::set_device_allocator(
    se::DeviceMemoryAllocator* allocator) {
  device_allocator_ = allocator;
  return *this;
}

se::DeviceMemoryAllocator* ExecutableBuildOptions::device_allocator() const {
  return device_allocator_;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_ordinal(
    int device_ordinal) {
  CHECK_GE(device_ordinal, 0);
  device_ordinal_ = device_ordinal;
  return *this;
}

int ExecutableBuildOptions::device_ordinal() const { return device_ordinal_; }

DebugOptions* ExecutableBuildOptions::mutable_debug_options() {
  if (!has_debug_options()) {
    debug_options_ = GetDebugOptionsFromFlags();
  }
  return &debug_options_.value();
}

ExecutableBuildOptions& ExecutableBuildOptions::set_result_layout(
    const Shape& shape_with_layout) {
  result_layout_set_ = true;
  result_layout_ = shape_with_layout;
  return *this;
}

CompilationEnvironments* ExecutableBuildOptions::mutable_comp_envs() {
  if (!has_comp_envs()) {
    comp_envs_.emplace();
  }
  return &*comp_envs_;
}

const Shape* ExecutableBuildOptions::result_layout() const {
  return result_layout_set_ ? &result_layout_ : nullptr;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_num_replicas(
    int num_replicas) {
  num_replicas_ = num_replicas;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_num_partitions(
    int num_partitions) {
  num_partitions_ = num_partitions;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_use_spmd_partitioning(
    bool use_spmd_partitioning) {
  use_spmd_partitioning_ = use_spmd_partitioning;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_use_auto_spmd_partitioning(
    bool use_auto_spmd_partitioning) {
  use_auto_spmd_partitioning_ = use_auto_spmd_partitioning;
  return *this;
}

ExecutableBuildOptions&
ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_shape(
    std::vector<int64_t> mesh_shape) {
  auto_spmd_partitioning_mesh_shape_ = std::move(mesh_shape);
  return *this;
}

ExecutableBuildOptions&
ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_ids(
    std::vector<int64_t> mesh_ids) {
  auto_spmd_partitioning_mesh_ids_ = std::move(mesh_ids);
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_deduplicate_hlo(
    bool deduplicate_hlo) {
  deduplicate_hlo_ = deduplicate_hlo;
  return *this;
}

ExecutableBuildOptions& ExecutableBuildOptions::set_device_assignment(
    const DeviceAssignment& device_assignment) {
  device_assignment_ = device_assignment;
  return *this;
}

std::string ExecutableBuildOptions::ToString() const {
  std::string result_layout = "nullopt";
  if (result_layout_set_) {
    result_layout = ShapeUtil::HumanStringWithLayout(result_layout_);
  }
  return absl::StrFormat(
      "ExecutableBuildOptions{device_ordinal=%d, result_layout=%s, "
      "num_replicas=%d}",
      device_ordinal_, result_layout, num_replicas_);
}

absl::StatusOr<ExecutableBuildOptionsProto> ExecutableBuildOptions::ToProto()
    const {
  ExecutableBuildOptionsProto output;
  output.set_device_ordinal(device_ordinal());
  if (result_layout()) {
    *output.mutable_result_layout() = result_layout()->ToProto();
  }
  if (has_comp_envs()) {
    *output.mutable_comp_envs() = comp_envs().ToProto();
  }
  if (has_debug_options()) {
    *output.mutable_debug_options() = debug_options();
  }
  if (layout_canonicalization_callback_) {
    return InvalidArgument(
        "Cannot serialize "
        "ExecutableBuildOptions::layout_canonicalization_callback");
  }
  if (compile_thread_pool() != nullptr) {
    return InvalidArgument(
        "Cannot serialize ExecutableBuildOptions::compile_thread_pool");
  }
  output.set_num_replicas(num_replicas());
  output.set_num_partitions(num_partitions());
  output.set_use_spmd_partitioning(use_spmd_partitioning());
  output.set_use_auto_spmd_partitioning(use_auto_spmd_partitioning());
  output.set_deduplicate_hlo(deduplicate_hlo());
  if (has_device_assignment()) {
    device_assignment().Serialize(output.mutable_device_assignment());
  }
  output.set_alias_passthrough_params(alias_passthrough_params());
  output.set_run_backend_only(run_backend_only());
  if (!allow_spmd_sharding_propagation_to_parameters().empty()) {
    output.mutable_allow_spmd_sharding_propagation_to_parameters()->Clear();
    for (bool v : allow_spmd_sharding_propagation_to_parameters()) {
      output.mutable_allow_spmd_sharding_propagation_to_parameters()->Add(v);
    }
  }
  if (!allow_spmd_sharding_propagation_to_output().empty()) {
    output.mutable_allow_spmd_sharding_propagation_to_output()->Clear();
    for (bool v : allow_spmd_sharding_propagation_to_output()) {
      output.mutable_allow_spmd_sharding_propagation_to_output()->Add(v);
    }
  }
  *output.mutable_fdo_profile() = fdo_profile();
  output.set_device_memory_size(device_memory_size());
  for (int64_t s : auto_spmd_partitioning_mesh_shape()) {
    output.mutable_auto_spmd_partitioning_mesh_shape()->Add(s);
  }
  for (int64_t s : auto_spmd_partitioning_mesh_ids()) {
    output.mutable_auto_spmd_partitioning_mesh_ids()->Add(s);
  }
  return output;
}

absl::StatusOr<ExecutableBuildOptions> ExecutableBuildOptionsFromProto(
    const ExecutableBuildOptionsProto& input) {
  xla::ExecutableBuildOptions output;
  if (input.device_ordinal() != -1) {
    output.set_device_ordinal(input.device_ordinal());
  }
  if (input.has_result_layout()) {
    output.set_result_layout(xla::Shape(input.result_layout()));
  }
  if (input.has_comp_envs()) {
    TF_ASSIGN_OR_RETURN(
        auto comp_envs,
        xla::CompilationEnvironments::CreateFromProto(input.comp_envs()));
    *output.mutable_comp_envs() = std::move(*comp_envs);
  }
  if (input.has_debug_options()) {
    *output.mutable_debug_options() = input.debug_options();
  }
  output.set_num_replicas(input.num_replicas());
  output.set_num_partitions(input.num_partitions());
  output.set_use_spmd_partitioning(input.use_spmd_partitioning());
  output.set_use_auto_spmd_partitioning(input.use_auto_spmd_partitioning());
  output.set_deduplicate_hlo(input.deduplicate_hlo());
  if (input.has_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::DeviceAssignment> assignment,
        xla::DeviceAssignment::Deserialize(input.device_assignment()));
    output.set_device_assignment(*assignment);
  }
  output.set_alias_passthrough_params(input.alias_passthrough_params());
  output.set_run_backend_only(input.run_backend_only());
  output.set_allow_spmd_sharding_propagation_to_parameters(
      input.allow_spmd_sharding_propagation_to_parameters());
  output.set_allow_spmd_sharding_propagation_to_output(
      input.allow_spmd_sharding_propagation_to_output());
  *output.mutable_fdo_profile() = input.fdo_profile();
  output.set_device_memory_size(input.device_memory_size());
  output.set_auto_spmd_partitioning_mesh_shape(
      std::vector<int64_t>(input.auto_spmd_partitioning_mesh_shape().begin(),
                           input.auto_spmd_partitioning_mesh_shape().end()));
  output.set_auto_spmd_partitioning_mesh_ids(
      std::vector<int64_t>(input.auto_spmd_partitioning_mesh_ids().begin(),
                           input.auto_spmd_partitioning_mesh_ids().end()));
  return output;
}

ExecutionOptions CreateExecutionOptions(
    const ExecutableBuildOptions& build_options,
    const ProgramShape* program_shape) {
  ExecutionOptions execution_options = CreateDefaultExecutionOptions();
  if (build_options.has_debug_options()) {
    *execution_options.mutable_debug_options() = build_options.debug_options();
  }
  if (build_options.result_layout() != nullptr) {
    *execution_options.mutable_shape_with_output_layout() =
        build_options.result_layout()->ToProto();
  } else {
    Shape result_shape(program_shape->result());
    LayoutUtil::SetToDefaultLayout(&result_shape);
    *execution_options.mutable_shape_with_output_layout() =
        result_shape.ToProto();
  }
  execution_options.set_num_replicas(build_options.num_replicas());
  execution_options.set_num_partitions(build_options.num_partitions());
  execution_options.set_use_spmd_partitioning(
      build_options.use_spmd_partitioning());
  execution_options.set_use_auto_spmd_partitioning(
      build_options.use_auto_spmd_partitioning());
  for (auto t : build_options.auto_spmd_partitioning_mesh_shape()) {
    execution_options.mutable_auto_spmd_partitioning_mesh_shape()->Add(t);
  }
  for (auto t : build_options.auto_spmd_partitioning_mesh_ids()) {
    execution_options.mutable_auto_spmd_partitioning_mesh_ids()->Add(t);
  }
  execution_options.set_deduplicate_hlo(build_options.deduplicate_hlo());
  if (!build_options.allow_spmd_sharding_propagation_to_parameters().empty()) {
    execution_options.mutable_allow_spmd_sharding_propagation_to_parameters()
        ->Clear();
    for (bool v :
         build_options.allow_spmd_sharding_propagation_to_parameters()) {
      execution_options.mutable_allow_spmd_sharding_propagation_to_parameters()
          ->Add(v);
    }
  }
  if (!build_options.allow_spmd_sharding_propagation_to_output().empty()) {
    execution_options.mutable_allow_spmd_sharding_propagation_to_output()
        ->Clear();
    for (bool v : build_options.allow_spmd_sharding_propagation_to_output()) {
      execution_options.mutable_allow_spmd_sharding_propagation_to_output()
          ->Add(v);
    }
  }
  if (build_options.has_device_assignment()) {
    build_options.device_assignment().Serialize(
        execution_options.mutable_device_assignment());
  }
  execution_options.set_alias_passthrough_params(
      build_options.alias_passthrough_params());
  execution_options.set_fdo_profile(build_options.fdo_profile().data(),
                                    build_options.fdo_profile().size());
  execution_options.set_device_memory_size(build_options.device_memory_size());
  return execution_options;
}

}  // namespace xla

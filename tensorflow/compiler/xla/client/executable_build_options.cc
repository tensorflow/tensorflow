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

#include "tensorflow/compiler/xla/client/executable_build_options.h"

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

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
  auto_spmd_partitioning_mesh_shape_ = mesh_shape;
  return *this;
}

ExecutableBuildOptions&
ExecutableBuildOptions::set_auto_spmd_partitioning_mesh_ids(
    std::vector<int64_t> mesh_ids) {
  auto_spmd_partitioning_mesh_ids_ = mesh_ids;
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
  execution_options.set_allow_spmd_sharding_propagation_to_output(
      build_options.allow_spmd_sharding_propagation_to_output());
  if (build_options.has_device_assignment()) {
    TF_CHECK_OK(build_options.device_assignment().Serialize(
        execution_options.mutable_device_assignment()));
  }
  execution_options.set_alias_passthrough_params(
      build_options.alias_passthrough_params());
  return execution_options;
}

}  // namespace xla

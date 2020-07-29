/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_

#include <string>
#include <vector>

#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_module_group.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {
namespace tpu {

namespace se = ::stream_executor;

// Describes the position of an argument or return value after the computation
// has been partitioned into cores.
struct ShardingAndIndex {
  // Sharding across cores.
  ::xla::OpSharding sharding;
  // Argument/return value number. If sharding is single-core, `indices` has a
  // single element; otherwise, it has num_cores elements.
  std::vector<int> indices;
};

// TODO(b/158279168): Dedup with internal version.
// Return the per-device shape for a `shape` with a given `sharding`.
xla::Shape GetPerDeviceShape(const xla::Shape& shape,
                             const xla::HloSharding& sharding,
                             int64 device);

stream_executor::port::StatusOr<std::unique_ptr<xla::HloModuleConfig>>
CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const xla::Shape> argument_shapes,
    absl::optional<const xla::Shape> result_layout,
    absl::optional<const xla::DeviceAssignment> device_assignment,
    int replica_count, int num_partitions,
    const xla::DebugOptions* debug_options, const int* seed,
    const int* launch_id, const bool* alias_passthrough_params,
    const xla::FusionConfigCollection* fusion_config_collection,
    const std::vector<std::vector<bool>>* fusion_config);

stream_executor::port::StatusOr<std::unique_ptr<xla::HloModuleConfig>>
CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const xla::Shape> argument_shapes,
    absl::optional<const xla::Shape> result_layout,
    absl::optional<const xla::DeviceAssignment> device_assignment,
    int replica_count,
    int num_partitions, const xla::DebugOptions* debug_options);

xla::ShapeTree<xla::HloSharding> GetSubtree(
    const xla::ShapeTree<xla::HloSharding>& tuple_shape_tree,
    int element_index);

xla::Shape GetPerDeviceShape(const xla::Shape& shape,
                             const xla::HloSharding& sharding,
                             int64 device);

Status AddVariableUpdatesToCores(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::vector<ShardingAndIndex>& arg_core_mapping,
    std::vector<bool>* may_modify_variables,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes,
    std::vector<std::vector<std::pair<int, bool>>>* per_core_variable_indices);

se::port::Status ComputeOutputShapesForEachCore(
    const tpu::TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes);

se::port::Status CreateHloModules(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const absl::optional<xla::DeviceAssignment>& device_assignment,
    std::vector<std::unique_ptr<xla::HloModule>>* hlo_modules);

se::port::StatusOr<TpuAotCompilationRequestProto>
CreateTpuAotCompilationRequest(
    const xla::HloModuleGroup& module_group,
    const XlaCompiler::CompilationResult& compilation_result,
    const TPUCompileMetadataProto& metadata,
    const std::vector<std::vector<xla::Shape>>& per_core_arg_shapes,
    const std::vector<std::vector<xla::Shape>>& per_core_output_shapes,
    const std::vector<std::vector<std::pair<int, bool>>>&
        per_core_variable_indices,
    const absl::optional<xla::DeviceAssignment>& device_assignment);
}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_

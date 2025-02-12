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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"

namespace tensorflow {
namespace tpu {

// List of parameters for lowering Mlir to HLO IR.
// If mlir_module_op is set, it will be used instead of mlir_module.
struct MlirToHloArgs {
  absl::string_view mlir_module;
  ConfigProto::Experimental::MlirBridgeRollout rollout_state =
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  std::optional<mlir::ModuleOp> mlir_module_op;
};

// Variant of guaranteed constant tensors types.
using GuaranteedConsts = std::variant<absl::Span<const TensorProto* const>,
                                      const OpInputList* const>;

// List of parameters for lowering function library definition to HLO IR.
struct FunctionToHloArgs {
  const NameAttrList* const function;
  const FunctionLibraryDefinition* flib_def;
  int graph_def_version;
  GuaranteedConsts guaranteed_constants;
};

// Persistent cache for compiled TPU program and the related compiler metadata
// intended for TPU inference.
// TODO(henrytan): there is an opportunity to consolidate the interface with the
// `TpuCompilationCacheInterface` once `TpuPersistentCompilationCache` is
// converted into a ref count based class.
class TpuPersistentCompilationCacheInterface {
 public:
  virtual ~TpuPersistentCompilationCacheInterface() = default;

  // Returns the location where cache entries are stored.
  virtual std::string cache_location() const = 0;
};

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
                             const xla::HloSharding& sharding, int64_t device);

absl::StatusOr<std::unique_ptr<xla::HloModuleConfig>> CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const xla::Shape> argument_shapes,
    std::optional<const xla::Shape> result_layout,
    std::optional<const xla::DeviceAssignment> device_assignment,
    int replica_count, int num_partitions,
    const xla::DebugOptions* debug_options, const int* seed,
    const int* launch_id, const bool* alias_passthrough_params,
    const xla::FusionConfigCollection* fusion_config_collection,
    const std::vector<std::vector<bool>>* fusion_config);

absl::StatusOr<std::unique_ptr<xla::HloModuleConfig>> CreateModuleConfig(
    const xla::ProgramShape& program_shape,
    absl::Span<const xla::Shape> argument_shapes,
    std::optional<const xla::Shape> result_layout,
    std::optional<const xla::DeviceAssignment> device_assignment,
    int replica_count, int num_partitions,
    const xla::DebugOptions* debug_options);

xla::ShapeTree<xla::HloSharding> GetSubtree(
    const xla::ShapeTree<xla::HloSharding>& tuple_shape_tree,
    int element_index);

xla::Shape GetPerDeviceShape(const xla::Shape& shape,
                             const xla::HloSharding& sharding, int64_t device);

absl::Status AddVariableUpdatesToCores(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::vector<ShardingAndIndex>& arg_core_mapping,
    std::vector<bool>* may_modify_variables,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes,
    std::vector<std::vector<std::pair<int, bool>>>* per_core_variable_indices);

absl::Status ComputeOutputShapesForEachCore(
    const tpu::TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    std::vector<std::vector<xla::Shape>>* per_core_output_shapes);

absl::Status CreateHloModules(
    const TPUCompileMetadataProto& metadata,
    const XlaCompiler::CompilationResult& compilation_result,
    const std::optional<xla::DeviceAssignment>& device_assignment,
    std::vector<std::unique_ptr<xla::HloModule>>* hlo_modules);

absl::StatusOr<TpuCompilationRequestProto> CreateTpuCompilationRequest(
    const std::variant<MlirToHloArgs, FunctionToHloArgs>& computation,
    const TPUCompileMetadataProto& metadata,
    const std::vector<TensorShape>& arg_shapes);

absl::Status CompileOpMetadataFromContext(OpKernelConstruction* ctx,
                                          TPUCompileMetadataProto* metadata,
                                          NameAttrList* function_name,
                                          std::string* mlir_module);

// Computes shapes for each argument. Uses both the static shape from the
// metadata, and the dynamic shapes where the static shape is not
// defined. There must be one dynamic_shape for each argument with a
// partially defined shape, in index order.
absl::Status ComputeArgumentShapes(
    const TPUCompileMetadataProto& metadata,
    const std::vector<TensorShape>& dynamic_shapes,
    std::vector<TensorShape>* arg_shapes);

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_OP_SUPPORT_H_

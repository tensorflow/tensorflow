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

#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"

#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/service/hlo_parser.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tsl/lib/math/math_util.h"

namespace tensorflow {
namespace {

constexpr char kNumSplitAttr[] = "num_split";

bool IsShapeKnown(mlir::TensorType type) {
  if (!type.hasRank()) return false;

  bool shape_known = false;
  for (int i = 0; i < type.getRank(); ++i) {
    if (type.getShape()[i] == mlir::ShapedType::kDynamic) {
      shape_known = false;
      break;
    } else {
      shape_known = true;
    }
  }

  return shape_known;
}

int64_t GetPadding(const int split_dim, const int num_splits,
                   const PartialTensorShape& partial_tensor_shape) {
  // If dim dimension is not defined, no uneven sharding support.
  if (partial_tensor_shape.dim_size(split_dim) <= 0) {
    return 0;
  }
  int64_t per_split_size = tsl::MathUtil::CeilOfRatio<int64_t>(
      partial_tensor_shape.dim_size(split_dim), num_splits);
  int64_t total_padding =
      per_split_size * num_splits - partial_tensor_shape.dim_size(split_dim);
  return total_padding;
}

mlir::TF::XlaConcatNDOp CreateXlaConcatNDOp(
    const mlir::Location& location, mlir::ArrayRef<mlir::Value> inputs,
    const std::vector<int64_t>& num_concats,
    const std::vector<int64_t>& paddings, mlir::OpBuilder& builder) {
  llvm::SmallVector<int64_t, 4> output_shape;
  if (inputs.empty()) {
    mlir::emitError(location, "inputs list to concat ops is empty");
    return nullptr;
  }
  if (num_concats.size() != paddings.size()) {
    mlir::emitError(location,
                    "num_concats and paddings must be of the same length.");
    return nullptr;
  }
  // All input slices must have the same shape, otherwise XLAConcatND op will
  // raise an error.
  auto input_slice_type = mlir::cast<mlir::TensorType>(inputs[0].getType());
  auto element_type = input_slice_type.getElementType();
  mlir::Type output_type;
  if (input_slice_type.hasRank()) {
    const auto& slice_shape = input_slice_type.getShape();
    for (int i = 0; i < num_concats.size(); i++) {
      auto num_concat = num_concats[i];
      const int max_dim_size = slice_shape[i] * num_concat;

      output_shape.emplace_back(max_dim_size - paddings[i]);
    }
    VLOG(2) << "CreateXlaConcatNDOp. output_shape="
            << absl::StrJoin(output_shape, ",")
            << ", Padding=" << absl::StrJoin(paddings, ",");
    output_type = mlir::RankedTensorType::get(output_shape, element_type);
  } else {
    output_type = input_slice_type;
  }

  auto op = builder.create<mlir::TF::XlaConcatNDOp>(
      location, output_type, inputs, builder.getI64ArrayAttr(num_concats),
      builder.getI64ArrayAttr(paddings));
  return op;
}

mlir::LogicalResult CreateXlaSplitNDOp(
    const mlir::Location& location, mlir::Value src_input, const int64_t rank,
    const std::vector<int64_t>& num_splits,
    const std::vector<int64_t>& paddings, mlir::OpBuilder* builder,
    mlir::TF::XlaSplitNDOp* xla_split_op, bool is_ici_weight_dist_spmd) {
  auto input_type = mlir::cast<mlir::TensorType>(src_input.getType());
  mlir::Type output_type;
  const auto& input_shape = input_type.getShape();

  auto output_slice_shape = llvm::to_vector<4>(input_type.getShape());
  int num_tiles = 1;
  if (num_splits.size() != rank || num_splits.size() != paddings.size()) {
    return mlir::failure();
  }
  for (int i = 0; i < rank; ++i) {
    if (input_shape[i] == mlir::ShapedType::kDynamic) {
      output_slice_shape[i] = input_shape[i];
    } else {
      output_slice_shape[i] = ((input_shape[i] + paddings[i]) / num_splits[i]);
    }
    num_tiles *= num_splits[i];
  }
  output_type = mlir::RankedTensorType::get(output_slice_shape,
                                            input_type.getElementType());

  llvm::SmallVector<mlir::Type, 4> output_types(num_tiles, output_type);

  VLOG(2) << "CreateXlaSplitNDOp. input_shape="
          << absl::StrJoin(input_shape, ",")
          << ", Padding: " << absl::StrJoin(paddings, ",");

  *xla_split_op = builder->create<mlir::TF::XlaSplitNDOp>(
      location, output_types, src_input, builder->getI64ArrayAttr(num_splits),
      builder->getI64ArrayAttr(paddings));
  if (is_ici_weight_dist_spmd) {
    (*xla_split_op)
        ->setAttr(kICIWeightDistributionMlirBridgeMarker,
                  builder->getBoolAttr(true));
  }
  return mlir::success();
}

mlir::LogicalResult HandleTileShardedInputsUsingXlaSplitOps(
    const mlir::Location& location, const xla::OpSharding& input_sharding,
    const mlir::Value& original_source, mlir::OpBuilder* builder,
    llvm::SmallVectorImpl<mlir::Value>* tiled_inputs,
    bool is_ici_weight_dist_spmd) {
  std::vector<int64_t> num_splits(
      input_sharding.tile_assignment_dimensions().begin(),
      input_sharding.replicate_on_last_tile_dim()
          ? std::prev(input_sharding.tile_assignment_dimensions().end())
          : input_sharding.tile_assignment_dimensions().end());

  const int rank = input_sharding.replicate_on_last_tile_dim()
                       ? input_sharding.tile_assignment_dimensions_size() - 1
                       : input_sharding.tile_assignment_dimensions_size();
  std::vector<int64_t> paddings;
  paddings.reserve(rank);
  const auto input_type =
      mlir::cast<mlir::TensorType>(original_source.getType());
  if (IsShapeKnown(input_type)) {
    auto shape = llvm::to_vector<4>(
        original_source.getType().cast<mlir::TensorType>().getShape());
    for (int dim = 0; dim < rank; ++dim) {
      paddings.push_back(
          GetPadding(dim, input_sharding.tile_assignment_dimensions(dim),
                     PartialTensorShape(shape)));
    }
  } else {
    paddings.assign(rank, 0);
  }

  mlir::TF::XlaSplitNDOp xla_split_op;
  if (mlir::failed(CreateXlaSplitNDOp(
          location, original_source, rank, num_splits, paddings, builder,
          &xla_split_op, is_ici_weight_dist_spmd))) {
    return mlir::failure();
  }

  tiled_inputs->clear();
  tiled_inputs->reserve(input_sharding.tile_assignment_devices_size());
  int64_t repeat_count =
      input_sharding.replicate_on_last_tile_dim()
          ? *input_sharding.tile_assignment_dimensions().rbegin()
          : 1;
  for (int i = 0; i < xla_split_op.getResults().size(); i++) {
    auto split_op_output = xla_split_op.getResults()[i];
    for (int64_t j = 0; j < repeat_count; ++j) {
      tiled_inputs->push_back(split_op_output);
    }
  }

  return mlir::success();
}

bool UnsupportedPartitionedShardingType(xla::OpSharding::Type sharding) {
  return sharding != xla::OpSharding::REPLICATED &&
         sharding != xla::OpSharding::OTHER;
}

}  // namespace

absl::StatusOr<std::map<int, int>> GetDimensionIndicesAndNumSplitsFromSharding(
    const xla::OpSharding& sharding) {
  int64_t tensor_tile_rank = sharding.tile_assignment_dimensions_size();
  if (sharding.replicate_on_last_tile_dim()) {
    tensor_tile_rank--;
  }

  std::map<int, int> dimension_to_splits_map;
  for (int dim_index = 0; dim_index < tensor_tile_rank; ++dim_index) {
    if (sharding.tile_assignment_dimensions(dim_index) > 1) {
      dimension_to_splits_map.emplace(
          dim_index, sharding.tile_assignment_dimensions(dim_index));
    }
  }

  if (dimension_to_splits_map.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Arg has unnecessary tiled sharding: ", sharding.DebugString()));
  }

  return dimension_to_splits_map;
}

int GetDimsFromXLAShardingTiled(const xla::OpSharding& xla_sharding) {
  return xla_sharding.tile_assignment_dimensions_size() -
         (xla_sharding.replicate_on_last_tile_dim() ? 1 : 0) -
         xla_sharding.last_tile_dims_size();
}

bool IsOtherReplicatedSharding(const xla::OpSharding& xla_sharding) {
  int max_dim = GetDimsFromXLAShardingTiled(xla_sharding);
  for (int i = 0; i < max_dim; ++i) {
    if (xla_sharding.tile_assignment_dimensions(i) != 1) {
      return false;
    }
  }
  return xla_sharding.type() == xla::OpSharding::OTHER &&
         (xla_sharding.replicate_on_last_tile_dim() ||
          !xla_sharding.last_tile_dims().empty());
}

bool IsSplitSharding(const xla::OpSharding& sharding) {
  return sharding.type() == xla::OpSharding::OTHER &&
         !IsOtherReplicatedSharding(sharding);
}

bool IsReplicatedSharding(const xla::OpSharding& sharding) {
  return sharding.type() == xla::OpSharding::REPLICATED ||
         IsOtherReplicatedSharding(sharding);
}

mlir::LogicalResult DecodeShardingAttribute(const std::string& shard_str,
                                            xla::OpSharding& sharding,
                                            bool report_error) {
  if (sharding.ParseFromString(shard_str)) return mlir::success();
  // TODO(b/287299845) MLIR should only have human readable representation
  // going forward. So, remove parsing binary sharding.
  absl::StatusOr<xla::HloSharding> sharding_hlo = xla::ParseSharding(shard_str);
  if (sharding_hlo.ok()) {
    sharding = sharding_hlo->ToProto();
    return mlir::success();
  }
  if (report_error)
    llvm::errs() << std::string(sharding_hlo.status().message()) << "\n";
  return mlir::failure();
}

mlir::LogicalResult DecodeShardingAttribute(mlir::Attribute shard_attr,
                                            xla::OpSharding& sharding,
                                            bool report_error) {
  if (!mlir::isa<mlir::StringAttr>(shard_attr)) return mlir::failure();

  auto shard_str = mlir::cast<mlir::StringAttr>(shard_attr).getValue().str();
  return DecodeShardingAttribute(shard_str, sharding, report_error);
}

void EncodeSharding(mlir::Operation* op, llvm::StringRef shard_str) {
  if (!op->hasAttrOfType<mlir::StringAttr>(shard_str)) return;

  ::xla::OpSharding sharding;
  auto sharding_proto_str =
      op->getAttrOfType<mlir::StringAttr>(shard_str).getValue().str();
  if (!sharding.ParseFromString(sharding_proto_str)) return;

  auto hlosharding = xla::HloSharding::FromProto(sharding);
  if (!hlosharding.ok()) {
    op->emitError("Unable to encode sharding to human readable ")
        << hlosharding.status().message();
    return;
  }
  op->setAttr(shard_str,
              mlir::StringAttr::get(op->getContext(), hlosharding->ToString()));
}

mlir::LogicalResult ExtractInputsForLogicalDevices(
    const int num_cores_per_replica,
    mlir::tf_device::ClusterFuncOp cluster_func, mlir::OpBuilder* builder,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Value, 4>>* input_list) {
  // Initialize the input list for each logical devices.
  input_list->reserve(num_cores_per_replica);
  for (int i = 0; i < num_cores_per_replica; ++i)
    input_list->emplace_back(llvm::SmallVector<mlir::Value, 4>());

  llvm::SmallVector<mlir::Value, 4> cluster_func_inputs(
      cluster_func.getOperands());
  auto sharding_attrs =
      cluster_func.getOperation()->getAttrOfType<mlir::ArrayAttr>(
          kInputShardingAttr);
  // If sharding attribute does not exist, then all inputs are placed on 0th
  // logical core by default.
  if (!sharding_attrs) {
    (*input_list)[0] = cluster_func_inputs;
    return mlir::success();
  }

  // Enumerate sharding configuration for each inputs. If input has replicate
  // sharding then all logical devices take the value as input. If input has
  // maximal sharding then only the specified logical device take the value as
  // the input.
  for (const auto& sharding_attr_and_index : llvm::enumerate(sharding_attrs)) {
    const auto& sharding_attr = sharding_attr_and_index.value();
    const auto input_index = sharding_attr_and_index.index();
    const auto& input_value = cluster_func_inputs[input_index];

    xla::OpSharding sharding;
    if (DecodeShardingAttribute(
            mlir::cast<mlir::StringAttr>(sharding_attr).getValue().str(),
            sharding)
            .failed()) {
      return cluster_func.emitError("incorrect sharding format for inputs");
    }

    const auto input_sharding_type = sharding.type();
    auto tiled_sharding_mismatched = [&](int tiled_input_size) {
      return cluster_func.emitError(
          llvm::formatv("incorrect {0}-th tiled input sharding received. "
                        "Product of tile sharding splits({1}) must be equal to "
                        "number of logical devices : {2}",
                        input_index, tiled_input_size, num_cores_per_replica));
    };

    // If input is already partitioned using the `tf.TPUPartitionedInputV2` op,
    // only replicated sharding is supported where i-th operand to
    // `tf.TPUPartitionedInputV2` op is input to the i-th logical device.
    if (auto partitioned_input =
            llvm::dyn_cast_or_null<mlir::TF::TPUPartitionedInputV2Op>(
                input_value.getDefiningOp())) {
      if (UnsupportedPartitionedShardingType(input_sharding_type))
        return cluster_func->emitOpError()
               << "unsupported input sharding type "
               << OpSharding_Type_Name(input_sharding_type) << " for "
               << input_index << "-th input";

      if (input_sharding_type == xla::OpSharding::REPLICATED) {
        for (const auto& index_and_inputs : llvm::enumerate(*input_list)) {
          index_and_inputs.value().emplace_back(
              partitioned_input.getOperand(index_and_inputs.index()));
        }
      } else {
        assert(input_sharding_type == xla::OpSharding::OTHER);
        if (partitioned_input.getInputs().size() != num_cores_per_replica)
          return tiled_sharding_mismatched(
              partitioned_input.getInputs().size());

        for (int i = 0; i < sharding.tile_assignment_devices_size(); ++i) {
          const int assigned_logical_device =
              sharding.tile_assignment_devices(i);
          (*input_list)[assigned_logical_device].emplace_back(
              partitioned_input.getInputs()[i]);
        }
      }
      continue;
    }

    if (IsSplitSharding(sharding)) {
      bool is_ici_weight_dist_spmd =
          cluster_func.getOperand(input_index).getDefiningOp() &&
          cluster_func.getOperand(input_index)
              .getDefiningOp()
              ->hasAttr(kICIWeightDistributionMlirBridgeMarker);
      llvm::SmallVector<mlir::Value, 4> tiled_inputs;

      auto result = HandleTileShardedInputsUsingXlaSplitOps(
          cluster_func.getLoc(), sharding, input_value, builder, &tiled_inputs,
          is_ici_weight_dist_spmd);
      if (mlir::failed(result)) return mlir::failure();

      const int64_t tiled_inputs_size = tiled_inputs.size();
      if (tiled_inputs_size != num_cores_per_replica)
        return tiled_sharding_mismatched(tiled_inputs.size());

      for (int i = 0; i < sharding.tile_assignment_devices_size(); ++i) {
        const int assigned_logical_device = sharding.tile_assignment_devices(i);
        (*input_list)[assigned_logical_device].emplace_back(tiled_inputs[i]);
      }
    } else if (IsReplicatedSharding(sharding)) {
      for (auto& inputs : *input_list) inputs.emplace_back(input_value);
    } else {
      assert(input_sharding_type == xla::OpSharding::MAXIMAL);
      const int logical_device_id = sharding.tile_assignment_devices(0);
      (*input_list)[logical_device_id].emplace_back(input_value);
    }
  }
  return mlir::success();
}

mlir::LogicalResult ParseAndValidateOutputSharding(
    const int num_cores_per_replica,
    mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::SmallVector<xla::OpSharding, 4>* output_sharding_list) {
  output_sharding_list->reserve(cluster_func.getNumResults());

  const auto output_sharding_attrs =
      cluster_func.getOperation()->getAttrOfType<mlir::ArrayAttr>(
          kOutputShardingAttr);
  if (!output_sharding_attrs)
    return cluster_func.emitError(
        "output_sharding_configuration missing from cluster func");

  if (output_sharding_attrs.size() != cluster_func.getNumResults())
    return cluster_func.emitError("incorrect number of output sharding");

  for (const auto& output_sharding_and_index :
       llvm::enumerate(output_sharding_attrs)) {
    const auto& output_sharding = output_sharding_and_index.value();
    const int sharding_index = output_sharding_and_index.index();
    if (!mlir::isa<mlir::StringAttr>(output_sharding))
      return cluster_func.emitError(llvm::formatv(
          "non-string output sharding at index {0}", sharding_index));

    xla::OpSharding sharding;
    if (DecodeShardingAttribute(
            mlir::cast<mlir::StringAttr>(output_sharding).getValue().str(),
            sharding)
            .failed()) {
      return cluster_func.emitError("incorrect sharding format for outputs");
    }

    if (sharding.type() == xla::OpSharding::OTHER &&
        sharding.tile_assignment_devices_size() != num_cores_per_replica)
      return cluster_func.emitError(llvm::formatv(
          "incorrect sharding format for outputs. Number of "
          "tiled outputs({0}) must match the number of logical "
          "devices({1})",
          sharding.tile_assignment_devices_size(), num_cores_per_replica));

    if (sharding.type() == xla::OpSharding::MAXIMAL &&
        ((sharding.tile_assignment_devices(0) >= num_cores_per_replica) ||
         (sharding.tile_assignment_devices(0) < 0)))
      return cluster_func.emitError(llvm::formatv(
          "incorrect sharding format for outputs. Maximal "
          "sharding should be assigned to device id in range "
          "[0, {0}). Currently assigned to {1}",
          num_cores_per_replica, sharding.tile_assignment_devices(0)));

    output_sharding_list->emplace_back(std::move(sharding));
  }
  return mlir::success();
}

namespace {

bool IsAssignedToLogicalDevice(const int core_id,
                               const xla::OpSharding& sharding) {
  return sharding.type() == xla::OpSharding::MAXIMAL &&
         sharding.tile_assignment_devices(0) == core_id;
}

// Returns the index of the return value of region in
// `tf_device.parallel_execute` that represents cluster func output at
// index |cluster_func_output_index|. Regions of parallel_execute may
// have different return values depending on output sharding configuration.
mlir::LogicalResult LookupClusterToCoreIndex(
    const mlir::Location& location,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    const int core_id, const int cluster_func_output_index,
    int* core_output_index) {
  *core_output_index =
      cluster_to_core_index[core_id][cluster_func_output_index];
  if (*core_output_index == -1) {
    mlir::emitError(
        location,
        llvm::formatv("Attempted to map cluster_func output index {0} to "
                      "program assigned to core {1}. The tensor at this output "
                      "index was not assigned or sharded to this core.",
                      cluster_func_output_index, core_id));
    return mlir::failure();
  }
  return mlir::success();
}

// Collects tile sharded outputs from a tf_device.parallel_execute to remap from
// the TPU computation result.
mlir::LogicalResult GetTileShardedOutputsToMerge(
    const mlir::Location& location, const int cluster_func_output_index,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    int cluster_idx, mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    llvm::SmallVector<mlir::Value, 4>* outputs_to_merge) {
  // Reorders outputs from TPUExecute op as defined by the output sharding
  // configuration.
  const xla::OpSharding& sharding =
      output_sharding_config[cluster_func_output_index];
  outputs_to_merge->reserve(sharding.tile_assignment_devices_size());
  for (const auto& core_id_and_index :
       llvm::enumerate(sharding.tile_assignment_devices())) {
    auto core_id = core_id_and_index.value();
    auto tile_index = core_id_and_index.index();

    int last_tile_dim_size = *sharding.tile_assignment_dimensions().rbegin();
    if (sharding.replicate_on_last_tile_dim() &&
        tile_index % last_tile_dim_size != 0) {
      continue;
    }

    int region_output_index;
    auto status = LookupClusterToCoreIndex(location, cluster_to_core_index,
                                           core_id, cluster_func_output_index,
                                           &region_output_index);
    if (failed(status)) return mlir::failure();
    const auto output_from_logical_device =
        new_parallel_execute.GetRegionOutputs(cluster_idx +
                                              core_id)[region_output_index];
    outputs_to_merge->emplace_back(output_from_logical_device);
  }

  return mlir::success();
}

mlir::LogicalResult HandleTileShardedOutputsUsingXlaConcatOps(
    const int cluster_func_output_index,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    const mlir::Location& location, mlir::Value cluster_func_output,
    int cluster_idx, mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    mlir::OpBuilder& builder) {
  // Inject concat ops after parallel_execute to merge outputs from
  // concurrently executed computations.
  builder.setInsertionPointAfter(new_parallel_execute);
  const xla::OpSharding& sharding =
      output_sharding_config[cluster_func_output_index];

  const std::vector<int64_t> num_concats(
      sharding.tile_assignment_dimensions().begin(),
      sharding.replicate_on_last_tile_dim()
          ? std::prev(sharding.tile_assignment_dimensions().end())
          : sharding.tile_assignment_dimensions().end());

  const int rank = sharding.replicate_on_last_tile_dim()
                       ? sharding.tile_assignment_dimensions_size() - 1
                       : sharding.tile_assignment_dimensions_size();
  std::vector<int64_t> paddings;
  paddings.reserve(rank);
  auto output_type =
      mlir::cast<mlir::TensorType>(cluster_func_output.getType());
  if (IsShapeKnown(output_type)) {
    auto shape = llvm::to_vector<4>(output_type.getShape());
    for (int dim = 0; dim < rank; ++dim) {
      paddings.push_back(GetPadding(dim,
                                    sharding.tile_assignment_dimensions(dim),
                                    PartialTensorShape(shape)));
    }
  } else {
    paddings.assign(rank, 0);
  }
  // Reorders outputs from TPUExecute op as defined by the output sharding
  // configuration.
  llvm::SmallVector<mlir::Value, 4> outputs_to_merge;
  auto status = GetTileShardedOutputsToMerge(
      location, cluster_func_output_index, output_sharding_config,
      cluster_to_core_index, cluster_idx, new_parallel_execute,
      &outputs_to_merge);
  if (failed(status)) return mlir::failure();

  mlir::TF::XlaConcatNDOp concat_op = CreateXlaConcatNDOp(
      location, outputs_to_merge, num_concats, paddings, builder);
  cluster_func_output.replaceAllUsesWith(concat_op.getResult());
  return mlir::success();
}

mlir::LogicalResult ValidateAndGetTiledExecuteOutputShape(
    const mlir::Location& location,
    const mlir::TensorType cluster_func_output_type,
    const xla::OpSharding& output_sharding,
    mlir::Type* tiled_logical_computation_type) {
  const auto output_shape = cluster_func_output_type.getShape();
  auto new_output_shape = llvm::to_vector<4>(output_shape);
  auto dimension_to_splits_map =
      GetDimensionIndicesAndNumSplitsFromSharding(output_sharding);
  if (!dimension_to_splits_map.ok()) {
    LOG(ERROR) << dimension_to_splits_map.status();
    return mlir::failure();
  }

  for (const auto& dimension_and_output_splits : *dimension_to_splits_map) {
    const auto dimension = dimension_and_output_splits.first;
    const auto output_splits = dimension_and_output_splits.second;

    if (output_shape[dimension] == mlir::ShapedType::kDynamic) {
      *tiled_logical_computation_type = cluster_func_output_type;
      break;
    }
    if (output_shape[dimension] % output_splits == 0) {
      new_output_shape[dimension] = output_shape[dimension] / output_splits;
    } else {
      // Input will be padded to be divisible by output_splits, thus add 1 to
      // the output shape.
      new_output_shape[dimension] =
          (output_shape[dimension] / output_splits) + 1;
    }
  }

  *tiled_logical_computation_type = mlir::RankedTensorType::get(
      new_output_shape, cluster_func_output_type.getElementType());

  return mlir::success();
}

}  // namespace

bool AreInputOutputShapesStaticallyKnownForSplitSharding(
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func) {
  bool sharded_input_output_shape_statically_known = true;
  // Check input shapes
  llvm::SmallVector<mlir::Value, 4> cluster_func_inputs(
      cluster_func.getOperands());
  auto sharding_attrs =
      cluster_func.getOperation()->getAttrOfType<mlir::ArrayAttr>(
          kInputShardingAttr);
  // If sharding attribute does not exist, then move on to check outputs.
  // We only need to know the shapes in case of split sharding to use correct
  // padding.
  if (sharding_attrs) {
    // Enumerate sharding configuration for each inputs and check shapes.
    for (const auto& sharding_attr_and_index :
         llvm::enumerate(sharding_attrs)) {
      const auto& sharding_attr = sharding_attr_and_index.value();
      const auto input_index = sharding_attr_and_index.index();
      const auto& input_value = cluster_func_inputs[input_index];
      const auto input_type =
          mlir::cast<mlir::TensorType>(input_value.getType());
      xla::OpSharding input_sharding;
      if (DecodeShardingAttribute(
              mlir::cast<mlir::StringAttr>(sharding_attr).getValue().str(),
              input_sharding)
              .failed()) {
        sharded_input_output_shape_statically_known = false;
      }
      // We only want to know about the shape for split sharding.
      if (IsSplitSharding(input_sharding)) {
        sharded_input_output_shape_statically_known &= IsShapeKnown(input_type);
      }
    }
  }
  // Check output shapes
  for (const auto& result_and_index :
       llvm::enumerate(cluster_func.getResults())) {
    const auto output_index = result_and_index.index();
    const auto& output_sharding = output_sharding_config[output_index];
    const auto cluster_func_output_type =
        mlir::cast<mlir::TensorType>(result_and_index.value().getType());
    // We only want to know about the shape for split sharding.
    if (IsSplitSharding(output_sharding)) {
      sharded_input_output_shape_statically_known &=
          IsShapeKnown(cluster_func_output_type);
    }
  }
  return sharded_input_output_shape_statically_known;
}

mlir::LogicalResult GetOutputTypesForLogicalDeviceComputation(
    const int core_id, llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func,
    llvm::SmallVectorImpl<mlir::Type>* output_types,
    llvm::SmallVectorImpl<int>* cluster_to_core_index) {
  output_types->reserve(cluster_func.getNumResults());

  int core_index = 0;
  for (const auto& result_and_index :
       llvm::enumerate(cluster_func.getResults())) {
    const auto output_index = result_and_index.index();
    const auto& output_sharding = output_sharding_config[output_index];
    const auto cluster_func_output_type =
        mlir::cast<mlir::TensorType>(result_and_index.value().getType());

    // If output shape of cluster func is statically known and output is tiled
    // sharded, then the corresponding output shape of cluster func must be
    // evenly divisible number of shardings, unless use of XLA split/concat ops
    // is enabled.
    if (IsSplitSharding(output_sharding)) {
      mlir::Type tiled_logical_computation_type;
      if (cluster_func_output_type.hasRank()) {
        auto result = ValidateAndGetTiledExecuteOutputShape(
            cluster_func.getLoc(), cluster_func_output_type, output_sharding,
            &tiled_logical_computation_type);
        if (mlir::failed(result)) return mlir::failure();
      } else {
        tiled_logical_computation_type = cluster_func_output_type;
      }
      cluster_to_core_index->emplace_back(core_index++);
      output_types->emplace_back(tiled_logical_computation_type);
    } else if (IsReplicatedSharding(output_sharding) ||
               IsAssignedToLogicalDevice(core_id, output_sharding)) {
      cluster_to_core_index->emplace_back(core_index++);
      output_types->emplace_back(cluster_func_output_type);
    } else {
      cluster_to_core_index->emplace_back(-1);
    }
  }

  return mlir::success();
}

mlir::LogicalResult RemapOutputsFromLogicalDevices(
    const mlir::Location& location,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    llvm::SmallVector<llvm::SmallVector<int, 4>, 4> cluster_to_core_index,
    int num_results_pre_cluster,
    mlir::tf_device::ParallelExecuteOp old_parallel_execute, int cluster_idx,
    mlir::tf_device::ParallelExecuteOp new_parallel_execute,
    mlir::OpBuilder* builder) {
  for (auto [output_index, old_parallel_execute_output] :
       llvm::enumerate(old_parallel_execute.getResults())) {
    if (output_index < num_results_pre_cluster) {
      // Replace the use of those results of old parallel_execute op from host
      // with corresponding results of new parallel_execute op
      for (auto& use : llvm::make_early_inc_range(
               old_parallel_execute->getResult(output_index).getUses())) {
        use.set(new_parallel_execute->getResult(output_index));
      }
      continue;
    }

    int tpu_cluster_output_index = output_index - num_results_pre_cluster;
    const auto& output_sharding =
        output_sharding_config[tpu_cluster_output_index];
    const auto output_sharding_type = output_sharding.type();

    // If output is demultiplexed using the `tf.TPUPartitionedOutputV2` op, only
    // replicated sharding is supported where i-th output of
    // `tf.TPUPartitionedOutputV2` op maps to the output of i-th logical device.
    // Also `tf.TPUPartitionedOutputV2` op must be a unique user of
    // TPU Cluster (`tf_device.old_parallel_execute`) output.
    mlir::TF::TPUPartitionedOutputV2Op partitioned_output;
    for (auto user : old_parallel_execute_output.getUsers()) {
      if (auto partitioned_output_user =
              llvm::dyn_cast_or_null<mlir::TF::TPUPartitionedOutputV2Op>(
                  user)) {
        partitioned_output = partitioned_output_user;
        break;
      }
    }
    if (partitioned_output) {
      if (!old_parallel_execute_output.hasOneUse())
        return partitioned_output.emitOpError()
               << "must be a unique user of TPU Cluster "
                  "(tf_device.old_parallel_execute) output "
               << *old_parallel_execute_output.getOwner();
      if (UnsupportedPartitionedShardingType(output_sharding_type))
        return old_parallel_execute.emitOpError()
               << "unsupported output sharding type "
               << OpSharding_Type_Name(output_sharding_type) << " for "
               << output_index << "-th output";

      if (output_sharding_type == xla::OpSharding::REPLICATED) {
        for (const auto& index_and_output :
             llvm::enumerate(partitioned_output.getOutput())) {
          auto idx = (cluster_idx + index_and_output.index()) %
                     new_parallel_execute->getNumRegions();
          const auto output_from_logical_device =
              new_parallel_execute.GetRegionOutputs(
                  idx)[tpu_cluster_output_index];
          index_and_output.value().replaceAllUsesWith(
              output_from_logical_device);
        }
      } else {
        assert(output_sharding_type == xla::OpSharding::OTHER);
        llvm::SmallVector<mlir::Value, 4> tile_sharded_outputs;
        if (failed(GetTileShardedOutputsToMerge(
                location, tpu_cluster_output_index, output_sharding_config,
                cluster_to_core_index, cluster_idx, new_parallel_execute,
                &tile_sharded_outputs)))
          return mlir::failure();
        for (auto result :
             llvm::zip(partitioned_output.getOutput(), tile_sharded_outputs))
          std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
      }
      continue;
    }
    if (IsSplitSharding(output_sharding)) {
      auto result = HandleTileShardedOutputsUsingXlaConcatOps(
          tpu_cluster_output_index, output_sharding_config,
          cluster_to_core_index, location, old_parallel_execute_output,
          cluster_idx, new_parallel_execute, *builder);
      if (mlir::failed(result)) return mlir::failure();
      continue;
    }

    int logical_device_id = 0;
    if (output_sharding_type == xla::OpSharding::MAXIMAL)
      logical_device_id = output_sharding.tile_assignment_devices(0);

    // For maximal sharding configuration, correctly remap outputs from
    // parallel_execute region to users of the cluster func.
    int region_output_index;
    if (failed(LookupClusterToCoreIndex(
            location, cluster_to_core_index, logical_device_id,
            tpu_cluster_output_index, &region_output_index)))
      return mlir::failure();

    const auto output_from_logical_device =
        new_parallel_execute.GetRegionOutputs(
            cluster_idx + logical_device_id)[region_output_index];
    old_parallel_execute_output.replaceAllUsesWith(output_from_logical_device);
  }
  return mlir::success();
}

llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> GetMetadataArgumentMapping(
    const tpu::TPUCompileMetadataProto& metadata) {
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> input_mappings(
      metadata.num_cores_per_replica(), llvm::SmallVector<int64_t, 4>());

  if (metadata.num_cores_per_replica() == 1) {
    input_mappings.front().resize(metadata.args_size());
    std::iota(input_mappings.front().begin(), input_mappings.front().end(), 0);
    return input_mappings;
  }

  for (const auto& arg_and_idx : llvm::enumerate(metadata.args())) {
    const auto& sharding = arg_and_idx.value().sharding();
    const int64_t idx = arg_and_idx.index();

    const auto sharding_type = sharding.type();
    if (sharding_type == xla::OpSharding::OTHER) {
      for (const auto& device : sharding.tile_assignment_devices()) {
        CHECK(device >= 0 && device < input_mappings.size());
        input_mappings[device].push_back(idx);
      }
    } else if (sharding_type == xla::OpSharding::REPLICATED) {
      for (auto& input : input_mappings) input.push_back(idx);
    } else {
      assert(sharding_type == xla::OpSharding::MAXIMAL);
      CHECK(sharding.tile_assignment_devices(0) >= 0 &&
            sharding.tile_assignment_devices(0) < input_mappings.size());
      input_mappings[sharding.tile_assignment_devices(0)].push_back(idx);
    }
  }

  return input_mappings;
}

}  // namespace tensorflow

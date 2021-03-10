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

#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tensorflow {

const char* const kInputShardingAttr = "input_sharding_configuration";
const char* const kOutputShardingAttr = "output_sharding_configuration";

namespace {

constexpr char kNumSplitAttr[] = "num_split";

// Creates a tf::SplitOp that splits 'src_input' into 'num_splits' ways
// in 'split_dimension' dimension and returns the split values.
mlir::LogicalResult CreateSplitOp(const int num_split,
                                  const int split_dimension,
                                  const mlir::Location& location,
                                  mlir::Value src_input,
                                  mlir::OpBuilder* builder,
                                  mlir::TF::SplitOp* split_op) {
  // Creates a const op to hold split dimension value.
  auto split_dim_type =
      mlir::RankedTensorType::get({}, builder->getIntegerType(32));
  auto split_dimension_attr =
      mlir::DenseElementsAttr::get(split_dim_type, split_dimension);
  auto split_dimension_op = builder->create<mlir::TF::ConstOp>(
      location, split_dim_type, split_dimension_attr);

  // Correctly set output shapes of split op output if input shape is statically
  // known.
  mlir::Type output_type;
  auto input_type = src_input.getType().cast<mlir::TensorType>();

  if (input_type.hasRank()) {
    if (input_type.getShape()[split_dimension] ==
        mlir::ShapedType::kDynamicSize) {
      output_type = input_type;
    } else {
      auto shape = llvm::to_vector<4>(input_type.getShape());
      if (shape[split_dimension] % num_split != 0) {
        return mlir::emitError(
            location,
            llvm::formatv(
                "incorrect input sharding configuration received. "
                "{0}-th dimension of the input must be evenly divisible by {1}",
                split_dimension, num_split));
      }

      shape[split_dimension] = shape[split_dimension] / num_split;
      output_type =
          mlir::RankedTensorType::get(shape, input_type.getElementType());
    }
  } else {
    output_type = input_type;
  }

  // Creates a split op that splits |src_input| along |split_dimension|.
  llvm::SmallVector<mlir::Type, 4> output_types(num_split, output_type);
  *split_op = builder->create<mlir::TF::SplitOp>(
      location, output_types, split_dimension_op.output(), src_input);
  (*split_op)->setAttr(
      kNumSplitAttr,
      builder->getIntegerAttr(builder->getIntegerType(32), num_split));
  return mlir::success();
}

// Creates a tf::ConcatOp that merges `input` values in `concat_dimension`.
mlir::TF::ConcatOp CreateConcatOp(const int concat_dimension,
                                  const mlir::Location& location,
                                  mlir::ArrayRef<mlir::Value> inputs,
                                  mlir::OpBuilder* builder) {
  // Creates a const op to hold concat dimension value.
  auto concat_dim_type =
      mlir::RankedTensorType::get({}, builder->getIntegerType(32));
  auto concat_dimension_attr =
      mlir::DenseElementsAttr::get(concat_dim_type, concat_dimension);
  auto concat_dimension_op = builder->create<mlir::TF::ConstOp>(
      location, concat_dim_type, concat_dimension_attr);

  // Correctly set output shapes of concat op output if output shape is
  // statically known. Since the shape of TPUExecute op must be the same
  // across logical devices, we refer to the shape of 0th logical device
  // computation output.
  mlir::Type output_type;
  auto input_type = inputs[0].getType().cast<mlir::TensorType>();

  if (input_type.hasRank()) {
    if (input_type.getShape()[concat_dimension] ==
        mlir::ShapedType::kDynamicSize) {
      output_type = input_type;
    } else {
      auto shape = llvm::to_vector<4>(input_type.getShape());
      shape[concat_dimension] = shape[concat_dimension] * inputs.size();
      output_type =
          mlir::RankedTensorType::get(shape, input_type.getElementType());
    }
  } else {
    output_type = input_type;
  }

  return builder->create<mlir::TF::ConcatOp>(
      location, output_type, concat_dimension_op.output(), inputs);
}

// For tile sharded inputs to TPU computation, inject split op between the
// input values and TPU computation so that tiled input values are passed in
// as inputs to TPU computations. If more than one dimension is sharded, then
// a tree of connected split ops are added before tf_device.parallel_execute op.
mlir::LogicalResult HandleTileShardedInputs(
    const mlir::Location& location, const xla::OpSharding& input_sharding,
    const mlir::Value& original_source, mlir::OpBuilder* builder,
    llvm::SmallVectorImpl<mlir::Value>* tiled_inputs) {
  llvm::SmallVector<mlir::TF::SplitOp, 4> split_ops_for_tiled_input;
  split_ops_for_tiled_input.reserve(
      input_sharding.tile_assignment_devices_size());

  // Creates a tree of split nodes for sharding tiled inputs. Splits nodes
  // are created such that input data is sharded in row major order.
  // Split nodes at ith depth from the original input node represent nodes
  // that split the input data at i-th dimension.
  const auto& dimension_splits = input_sharding.tile_assignment_dimensions();
  for (auto num_splits_and_index : llvm::enumerate(dimension_splits)) {
    const int num_splits = num_splits_and_index.value();
    const int dimension_index = num_splits_and_index.index();
    if (num_splits == 1) continue;

    // Creates root split op.
    if (split_ops_for_tiled_input.empty()) {
      mlir::TF::SplitOp root_split_op;
      auto result = CreateSplitOp(num_splits, dimension_index, location,
                                  original_source, builder, &root_split_op);
      if (mlir::failed(result)) return mlir::failure();

      split_ops_for_tiled_input.emplace_back(root_split_op);
      continue;
    }

    llvm::SmallVector<mlir::TF::SplitOp, 4> new_split_ops;
    new_split_ops.reserve(split_ops_for_tiled_input.size() * num_splits);

    for (auto split_op : split_ops_for_tiled_input) {
      for (auto parent_split_output_value : split_op.getResults()) {
        mlir::TF::SplitOp child_split_op;
        auto result =
            CreateSplitOp(num_splits, dimension_index, location,
                          parent_split_output_value, builder, &child_split_op);
        if (mlir::failed(result)) return mlir::failure();

        new_split_ops.emplace_back(child_split_op);
      }
    }

    std::swap(new_split_ops, split_ops_for_tiled_input);
  }

  // `split_ops_for_tiled_input` now includes final split nodes
  // from which sharded data will be fed into TPUExcute ops -- sorted by
  // row major order.
  tiled_inputs->reserve(input_sharding.tile_assignment_devices_size());
  for (auto split_op : split_ops_for_tiled_input)
    tiled_inputs->append(split_op.getResults().begin(),
                         split_op.getResults().end());

  return mlir::success();
}

bool UnsupportedPartitionedShardingType(xla::OpSharding::Type sharding) {
  return sharding != xla::OpSharding::REPLICATED &&
         sharding != xla::OpSharding::OTHER;
}

}  // namespace

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
    sharding.ParseFromString(
        sharding_attr.cast<mlir::StringAttr>().getValue().str());

    const auto input_sharding_type = sharding.type();

    auto tiled_sharding_mismatched = [&](int tiled_input_size) {
      return cluster_func.emitError(
          llvm::formatv("incorrect {0}-th tiled input sharding received. "
                        "Product of tile sharding splits({1}) must be equal to "
                        "number of logical devices : {2}",
                        input_index, tiled_input_size, num_cores_per_replica));
    };

    // If input is already partitioned using the `tf.TPUPartitionedInput` op,
    // only replicated sharding is supported where i-th operand to
    // `tf.TPUPartitionedInput` op is input to the i-th logical device.
    if (auto partitioned_input =
            llvm::dyn_cast_or_null<mlir::TF::TPUPartitionedInputOp>(
                input_value.getDefiningOp())) {
      if (UnsupportedPartitionedShardingType(input_sharding_type))
        return cluster_func->emitOpError()
               << "unsupported input sharding type "
               << OpSharding_Type_Name(input_sharding_type) << " for "
               << input_index << "-th input";

      if (input_sharding_type == xla::OpSharding::REPLICATED) {
        for (auto& index_and_inputs : llvm::enumerate(*input_list)) {
          index_and_inputs.value().emplace_back(
              partitioned_input.getOperand(index_and_inputs.index()));
        }
      } else {
        assert(input_sharding_type == xla::OpSharding::OTHER);
        if (partitioned_input.inputs().size() != num_cores_per_replica)
          return tiled_sharding_mismatched(partitioned_input.inputs().size());

        for (int i = 0; i < sharding.tile_assignment_devices_size(); ++i) {
          const int assigned_logical_device =
              sharding.tile_assignment_devices(i);
          (*input_list)[assigned_logical_device].emplace_back(
              partitioned_input.inputs()[i]);
        }
      }
      continue;
    }

    if (input_sharding_type == xla::OpSharding::OTHER) {
      llvm::SmallVector<mlir::Value, 4> tiled_inputs;
      auto result = HandleTileShardedInputs(
          cluster_func.getLoc(), sharding, input_value, builder, &tiled_inputs);
      if (mlir::failed(result)) return mlir::failure();

      const int64 tiled_inputs_size = tiled_inputs.size();
      if (tiled_inputs_size != num_cores_per_replica)
        return tiled_sharding_mismatched(tiled_inputs.size());

      for (int i = 0; i < sharding.tile_assignment_devices_size(); ++i) {
        const int assigned_logical_device = sharding.tile_assignment_devices(i);
        (*input_list)[assigned_logical_device].emplace_back(tiled_inputs[i]);
      }
    } else if (input_sharding_type == xla::OpSharding::REPLICATED) {
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

  for (auto output_sharding_and_index :
       llvm::enumerate(output_sharding_attrs)) {
    const auto& output_sharding = output_sharding_and_index.value();
    const int sharding_index = output_sharding_and_index.index();
    if (!output_sharding.isa<mlir::StringAttr>())
      return cluster_func.emitError(llvm::formatv(
          "non-string output sharding at index {0}", sharding_index));

    xla::OpSharding sharding;
    if (!sharding.ParseFromString(
            output_sharding.cast<mlir::StringAttr>().getValue().str()))
      return cluster_func.emitError("incorrect sharding format for outputs");

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
int MapClusterOutputIndexWithRegionOutputIndex(
    llvm::ArrayRef<xla::OpSharding> output_sharding_config, const int core_id,
    const int cluster_func_output_index) {
  int region_output_index = 0;
  for (int output_index = 0; output_index < cluster_func_output_index;
       ++output_index) {
    const auto& sharding = output_sharding_config[output_index];
    if (sharding.type() != xla::OpSharding::MAXIMAL ||
        IsAssignedToLogicalDevice(core_id, sharding))
      region_output_index++;
  }

  return region_output_index;
}

// Collects tile sharded outputs from a tf_device.parallel_execute to remap from
// the TPU computation result.
llvm::SmallVector<mlir::Value, 4> GetTileShardedOutputsToMerge(
    const int cluster_func_output_index,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ParallelExecuteOp parallel_execute) {
  // Reorders outputs from TPUExecute op as defined by the output sharding
  // configuration.
  const xla::OpSharding& sharding =
      output_sharding_config[cluster_func_output_index];
  llvm::SmallVector<mlir::Value, 4> outputs_to_merge;
  outputs_to_merge.reserve(sharding.tile_assignment_devices_size());
  for (const auto logical_device_id : sharding.tile_assignment_devices()) {
    const int region_output_index = MapClusterOutputIndexWithRegionOutputIndex(
        output_sharding_config, logical_device_id, cluster_func_output_index);
    const auto output_from_logical_device = parallel_execute.GetRegionOutputs(
        logical_device_id)[region_output_index];
    outputs_to_merge.emplace_back(output_from_logical_device);
  }

  return outputs_to_merge;
}

// Merges outputs from TPU computation for tile-sharded outputs.
void HandleTileShardedOutputs(
    const int cluster_func_output_index,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    const mlir::Location& location, mlir::Value cluster_func_output,
    mlir::tf_device::ParallelExecuteOp parallel_execute,
    mlir::OpBuilder* builder) {
  // Inject concat ops after parallel_execute to merge outputs from
  // concurrently executed computations.
  builder->setInsertionPointAfter(parallel_execute);

  // Reorders outputs from TPUExecute op as defined by the output sharding
  // configuration.
  auto outputs_to_merge = GetTileShardedOutputsToMerge(
      cluster_func_output_index, output_sharding_config, parallel_execute);

  // Creates a tree of Concat ops that merges outputs from multiple logical
  // devices to a single replica output.
  const xla::OpSharding& sharding =
      output_sharding_config[cluster_func_output_index];
  int concat_dimension = sharding.tile_assignment_dimensions_size() - 1;
  for (auto num_splits : llvm::reverse(sharding.tile_assignment_dimensions())) {
    if (num_splits == 1) {
      --concat_dimension;
      continue;
    }

    llvm::SmallVector<mlir::Value, 4> new_outputs;
    new_outputs.reserve(num_splits);
    for (int i = 0, end = outputs_to_merge.size(); i < end;
         i = i + num_splits) {
      mlir::TF::ConcatOp concat_op =
          CreateConcatOp(concat_dimension, location,
                         llvm::ArrayRef<mlir::Value>{
                             outputs_to_merge.begin() + i,
                             outputs_to_merge.begin() + i + num_splits},
                         builder);
      new_outputs.emplace_back(concat_op.getResult());
    }

    std::swap(new_outputs, outputs_to_merge);
    --concat_dimension;
  }

  assert(outputs_to_merge.size() == 1);
  cluster_func_output.replaceAllUsesWith(outputs_to_merge[0]);
}

mlir::LogicalResult ValidateAndGetTiledExecuteOutputShape(
    const mlir::Location& location,
    const mlir::TensorType cluster_func_output_type,
    const xla::OpSharding& output_sharding,
    mlir::Type* tiled_logical_computation_type) {
  auto new_output_shape =
      llvm::to_vector<4>(cluster_func_output_type.getShape());
  for (auto dimension_and_output_splits :
       llvm::enumerate(output_sharding.tile_assignment_dimensions())) {
    const auto dimension_index = dimension_and_output_splits.index();
    const auto output_splits = dimension_and_output_splits.value();
    const auto output_shape = cluster_func_output_type.getShape();

    if (output_shape[dimension_index] == mlir::ShapedType::kDynamicSize) {
      *tiled_logical_computation_type = cluster_func_output_type;
      break;
    }

    auto output_shape_at_dim =
        cluster_func_output_type.getShape()[dimension_index];
    if (output_shape_at_dim % output_splits != 0) {
      mlir::emitError(
          location,
          llvm::formatv("incorrect output sharding received. "
                        "{0}-th dimension of the output must be "
                        "evenly divisible by {1}, got dimension "
                        "shape {2}",
                        dimension_index, output_splits, output_shape_at_dim));
    }

    new_output_shape[dimension_index] =
        output_shape[dimension_index] / output_splits;
  }

  *tiled_logical_computation_type = mlir::RankedTensorType::get(
      new_output_shape, cluster_func_output_type.getElementType());

  return mlir::success();
}

}  // namespace

mlir::LogicalResult GetOutputTypesForLogicalDeviceComputation(
    const int core_id, llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func,
    llvm::SmallVectorImpl<mlir::Type>* output_types) {
  output_types->reserve(cluster_func.getNumResults());

  for (auto result_and_index : llvm::enumerate(cluster_func.getResults())) {
    const auto output_index = result_and_index.index();
    const auto& output_sharding = output_sharding_config[output_index];
    const auto output_sharding_type = output_sharding.type();
    const auto cluster_func_output_type =
        result_and_index.value().getType().cast<mlir::TensorType>();

    // If output shape of cluster func is statically known and output is tiled
    // sharded, then the corresponding output shape of cluster func must be
    // evenly divisible number of shardings.
    if (output_sharding_type == xla::OpSharding::OTHER) {
      mlir::Type tiled_logical_computation_type;
      if (cluster_func_output_type.hasRank()) {
        auto result = ValidateAndGetTiledExecuteOutputShape(
            cluster_func.getLoc(), cluster_func_output_type, output_sharding,
            &tiled_logical_computation_type);
        if (mlir::failed(result)) return mlir::failure();
      } else {
        tiled_logical_computation_type = cluster_func_output_type;
      }
      output_types->emplace_back(tiled_logical_computation_type);
    } else if (output_sharding_type == xla::OpSharding::REPLICATED ||
               IsAssignedToLogicalDevice(core_id, output_sharding)) {
      output_types->emplace_back(cluster_func_output_type);
    }
  }

  return mlir::success();
}

mlir::LogicalResult RemapOutputsFromLogicalDevices(
    const mlir::Location& location,
    llvm::ArrayRef<xla::OpSharding> output_sharding_config,
    mlir::tf_device::ClusterFuncOp cluster_func,
    mlir::tf_device::ParallelExecuteOp parallel_execute,
    mlir::OpBuilder* builder) {
  for (auto result_and_index : llvm::enumerate(cluster_func.getResults())) {
    const auto output_index = result_and_index.index();
    const auto cluster_func_output = result_and_index.value();
    const auto& output_sharding = output_sharding_config[output_index];
    const auto output_sharding_type = output_sharding.type();

    // If output is demultiplexed using the `tf.TPUPartitionedOutput` op, only
    // replicated sharding is supported where i-th output of
    // `tf.TPUPartitionedOutput` op maps to the output of i-th logical device.
    // Also `tf.TPUPartitionedOutput` op must be a unique user of
    // `tf_device.cluster_func` output.
    mlir::TF::TPUPartitionedOutputOp partitioned_output;
    for (auto user : cluster_func_output.getUsers()) {
      if (auto partitioned_output_user =
              llvm::dyn_cast_or_null<mlir::TF::TPUPartitionedOutputOp>(user)) {
        partitioned_output = partitioned_output_user;
        break;
      }
    }
    if (partitioned_output) {
      if (!cluster_func_output.hasOneUse())
        return partitioned_output.emitOpError()
               << "must be a unique user of tf_device.cluster_func output "
               << *cluster_func_output.getOwner();
      if (UnsupportedPartitionedShardingType(output_sharding_type))
        return cluster_func.emitOpError()
               << "unsupported output sharding type "
               << OpSharding_Type_Name(output_sharding_type) << " for "
               << output_index << "-th output";

      if (output_sharding_type == xla::OpSharding::REPLICATED) {
        for (auto index_and_output :
             llvm::enumerate(partitioned_output.output())) {
          const auto output_from_logical_device =
              parallel_execute.GetRegionOutputs(
                  index_and_output.index())[output_index];
          index_and_output.value().replaceAllUsesWith(
              output_from_logical_device);
        }
      } else {
        assert(output_sharding_type == xla::OpSharding::OTHER);
        auto tile_sharded_outputs = GetTileShardedOutputsToMerge(
            output_index, output_sharding_config, parallel_execute);
        for (auto result :
             llvm::zip(partitioned_output.output(), tile_sharded_outputs))
          std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
      }
      continue;
    }

    if (output_sharding_type == xla::OpSharding::OTHER) {
      HandleTileShardedOutputs(output_index, output_sharding_config, location,
                               cluster_func_output, parallel_execute, builder);
      continue;
    }

    int logical_device_id = 0;
    if (output_sharding_type == xla::OpSharding::MAXIMAL)
      logical_device_id = output_sharding.tile_assignment_devices(0);

    // For maximal sharding configuration, correctly remap outputs from
    // parallel_execute region to users of the cluster func.
    const int region_output_index = MapClusterOutputIndexWithRegionOutputIndex(
        output_sharding_config, logical_device_id, output_index);

    const auto output_from_logical_device = parallel_execute.GetRegionOutputs(
        logical_device_id)[region_output_index];
    cluster_func_output.replaceAllUsesWith(output_from_logical_device);
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
      for (const auto& device : sharding.tile_assignment_devices())
        input_mappings[device].push_back(idx);
    } else if (sharding_type == xla::OpSharding::REPLICATED) {
      for (auto& input : input_mappings) input.push_back(idx);
    } else {
      assert(sharding_type == xla::OpSharding::MAXIMAL);
      input_mappings[sharding.tile_assignment_devices(0)].push_back(idx);
    }
  }

  return input_mappings;
}

}  // namespace tensorflow

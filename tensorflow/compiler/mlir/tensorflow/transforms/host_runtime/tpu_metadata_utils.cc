/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/tpu_metadata_utils.h"

#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace mlir {
namespace TFTPU {
namespace {
constexpr char kStepMarkerLocationAttr[] = "step_marker_location";
constexpr char kUseXlaSpmdAttr[] = "use_spmd_for_xla_partitioning";

constexpr char kBadStringArrayElementMsg[] =
    "bad '{0}' attribute at index {1}, not a string";
constexpr char kBadArrayElementMsg[] =
    "bad '{0}' attribute at index {1} with value '{2}': failed to parse to {3}";
constexpr char kBadArrayAttrLengthMsg[] =
    "bad '{0}' attribute, expected array attribute of size {1}, got size {2}";

// Creates a missing attribute error message.
std::string CreateMissingAttributeMsg(llvm::StringRef attribute) {
  return llvm::formatv("requires attribute '{0}'", attribute).str();
}

// Populates a TPUCompileMetadataProto with StepMarkerLocation from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoStepMarkerLocation(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto step_marker_location =
      op->getAttrOfType<StringAttr>(kStepMarkerLocationAttr);
  if (!step_marker_location)
    return op.emitOpError(CreateMissingAttributeMsg(kStepMarkerLocationAttr));

  // Default to `STEP_MARK_AT_ENTRY` for step marker location if attribute is
  // empty.
  xla::DebugOptions::StepMarkerLocation location =
      xla::DebugOptions::STEP_MARK_AT_ENTRY;
  if (!step_marker_location.getValue().empty() &&
      !xla::DebugOptions::StepMarkerLocation_Parse(
          std::string(step_marker_location.getValue()), &location))
    return op.emitOpError(llvm::formatv("bad '{0}' attribute with value '{1}'",
                                        kStepMarkerLocationAttr,
                                        step_marker_location.getValue()));

  metadata->set_step_marker_location(location);

  return success();
}

// Parses a xla::OpSharding from a string attribute.
LogicalResult SetOpSharding(Operation* op, Attribute attr, llvm::StringRef name,
                            int index, xla::OpSharding* sharding_ptr) {
  auto sharding_attr = attr.dyn_cast<StringAttr>();
  if (!sharding_attr)
    return op->emitOpError(
        llvm::formatv(kBadStringArrayElementMsg, name, index));
  if (tensorflow::DecodeShardingAttribute(sharding_attr, *sharding_ptr)
          .failed()) {
    return op->emitOpError(llvm::formatv(kBadArrayElementMsg, name, index,
                                         sharding_attr.getValue(),
                                         "xla::OpSharding"));
  }
  return success();
}

// Populates a TPUCompileMetadataProto with argument types and sharding from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoArgs(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto input_shardings =
      op->getAttrOfType<ArrayAttr>(tensorflow::kInputShardingAttr);
  if (!input_shardings)
    return op.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kInputShardingAttr));

  if (input_shardings.size() != op.getNumOperands())
    return op.emitOpError(
        llvm::formatv(kBadArrayAttrLengthMsg, tensorflow::kInputShardingAttr,
                      op.getNumOperands(), input_shardings.size()));

  // Set args metadata in proto.
  mlir::StringAttr replication_attr_name = mlir::StringAttr::get(
      op.getContext(), "mhlo.is_same_data_across_replicas");

  auto dynamic_arg_idx = op->getAttrOfType<ArrayAttr>(TF::kDynamicArgIndexAttr);
  llvm::SmallSet<int, 4> dynamic_arg_idx_set;
  if (dynamic_arg_idx) {
    for (auto idx : dynamic_arg_idx.getValue()) {
      dynamic_arg_idx_set.insert(idx.dyn_cast<IntegerAttr>().getInt());
    }
  }

  for (auto operand_type_and_idx : llvm::enumerate(op.getOperandTypes())) {
    Type operand_type = operand_type_and_idx.value();
    int index = operand_type_and_idx.index();
    tensorflow::tpu::TPUCompileMetadataProto::Arg* arg = metadata->add_args();
    tensorflow::DataType dtype;
    tensorflow::Status status =
        tensorflow::ConvertToDataType(operand_type, &dtype);
    if (!status.ok())
      return op.emitOpError(
          llvm::formatv("failed to determine operand type at index {0}: {1}",
                        index, status.message()));

    arg->set_dtype(dtype);
    // TODO(lyandy): Support other arg kinds.
    if (dtype == tensorflow::DT_RESOURCE)
      arg->set_kind(tensorflow::tpu::TPUCompileMetadataProto::Arg::VARIABLE);
    else
      arg->set_kind(tensorflow::tpu::TPUCompileMetadataProto::Arg::PARAMETER);

    // Populate argument shapes.
    *arg->mutable_shape() = tensorflow::TensorShapeProto();
    if (auto ranked_tensor_type = operand_type.dyn_cast<RankedTensorType>()) {
      tensorflow::TensorShapeProto shape_proto;
      ConvertToTensorShapeProto(ranked_tensor_type.getShape(), &shape_proto);
      *arg->mutable_shape() = std::move(shape_proto);
    } else {
      arg->mutable_shape()->set_unknown_rank(true);
    }

    if (failed(SetOpSharding(op, input_shardings.getValue()[index],
                             tensorflow::kInputShardingAttr, index,
                             arg->mutable_sharding())))
      return failure();

    // Populate set_is_same_data_across_replicas
    // Note: this information is duplicated and can be removed from the proto
    // and here once MLIR bridge phase 2 doesn't fallback to the old bridge.
    auto attr = op.getFuncOp().getArgAttrOfType<mlir::BoolAttr>(
        index, replication_attr_name);
    arg->set_is_same_data_across_replicas(attr != nullptr && attr.getValue());

    // Currently only support first dimension to be bounded dynamic.
    arg->mutable_is_bounded_dynamic_dim()->Add(
        dynamic_arg_idx_set.contains(index));
  }

  return success();
}

// Populates a TPUCompileMetadataProto with result sharding from a
// `tf_device::ClusterFuncOp`.
LogicalResult SetMetadataProtoRetvals(
    tf_device::ClusterFuncOp op,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  auto output_shardings =
      op->getAttrOfType<ArrayAttr>(tensorflow::kOutputShardingAttr);
  if (!output_shardings)
    return op.emitOpError(
        CreateMissingAttributeMsg(tensorflow::kOutputShardingAttr));

  if (output_shardings.size() != op.getNumResults())
    return op.emitOpError(
        llvm::formatv(kBadArrayAttrLengthMsg, tensorflow::kOutputShardingAttr,
                      op.getNumResults(), output_shardings.size()));

  // Set retvals metadata in proto.
  for (auto output_sharding_and_idx : llvm::enumerate(output_shardings))
    if (failed(SetOpSharding(op, output_sharding_and_idx.value(),
                             tensorflow::kOutputShardingAttr,
                             output_sharding_and_idx.index(),
                             metadata->add_retvals()->mutable_sharding())))
      return failure();

  return success();
}

}  // namespace

// Populates a TPUCompileMetadataProto from attributes of a
// `tf_device::ClusterFuncOp`. If any necessary attributes are missing from the
// op, a failure will be returned.
// TODO(lyandy): Support session handle and guaranteed consts.
LogicalResult SetMetadataProtoFromClusterFuncOp(
    tf_device::ClusterFuncOp op, int num_replicas, int num_cores_per_replica,
    std::optional<xla::DeviceAssignmentProto>&& xla_device_assignment,
    tensorflow::tpu::TPUCompileMetadataProto* metadata) {
  if (auto options_attr =
          op->getAttrOfType<StringAttr>("tpu_compile_options_proto")) {
    if (!metadata->mutable_compile_options()->ParseFromArray(
            options_attr.data(), options_attr.size())) {
      return failure();
    }
  }
  metadata->set_num_replicas(num_replicas);
  metadata->set_num_cores_per_replica(num_cores_per_replica);

  if (failed(SetMetadataProtoStepMarkerLocation(op, metadata)))
    return failure();

  if (xla_device_assignment.has_value())
    *metadata->mutable_device_assignment() =
        std::move(xla_device_assignment.value());
  auto use_spmd_attr = op->getAttrOfType<BoolAttr>(kUseXlaSpmdAttr);
  if (!use_spmd_attr)
    return op.emitOpError(CreateMissingAttributeMsg(kUseXlaSpmdAttr));
  metadata->set_use_spmd_for_xla_partitioning(use_spmd_attr.getValue());

  if (failed(SetMetadataProtoArgs(op, metadata))) return failure();

  return SetMetadataProtoRetvals(op, metadata);
}

}  // namespace TFTPU
}  // namespace mlir

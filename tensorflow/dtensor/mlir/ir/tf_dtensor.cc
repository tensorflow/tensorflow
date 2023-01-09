/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

#include <cstdint>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace TF {
namespace {

RankedTensorType GetRankedTensorType(mlir::Value val) {
  mlir::Type type = val.getType();
  if (auto type_with_subtype =
          mlir::getElementTypeOrSelf(val)
              .dyn_cast<mlir::TF::TensorFlowTypeWithSubtype>()) {
    if (type_with_subtype.GetSubtypes().size() == 1) {
      type = type_with_subtype.GetSubtypes().front();
    }
  }
  return type.dyn_cast_or_null<RankedTensorType>();
}
}  // namespace

mlir::LogicalResult DTensorLayout::verify() {
  DTensorLayout op = *this;
  const auto& layout = op.getLayout();
  if (layout.IsEmpty()) return mlir::success();

  auto input_value = op.getInput();

  RankedTensorType type = GetRankedTensorType(input_value);

  if (!type) return mlir::success();

  const auto& num_shards = layout.num_shards();
  if (num_shards.size() != type.getRank()) {
    return op.emitOpError(llvm::formatv(
        "requires matching rank for layout and input, but got {0} as suggested "
        "rank from layout but {1} from shape.",
        num_shards.size(), type.getRank()));
  }

  for (const auto& dim_and_index :
       llvm::enumerate(llvm::zip(type.getShape(), num_shards))) {
    const int dimension_index = dim_and_index.index();
    const auto& dim_and_shards = dim_and_index.value();
    const int dim = std::get<0>(dim_and_shards);
    const int num_shard_for_dim = std::get<1>(dim_and_shards);
    if (dim <= 0) continue;

    if (dim % num_shard_for_dim != 0)
      return op.emitOpError(llvm::formatv(
          "requires dimension {0} to be divisible by sharding "
          "specified in DTensorLayout, but got dimension size={1} is not "
          "divisible by number of shards in layout for this dimension={2}.",
          dimension_index, dim, num_shard_for_dim));
  }

  return mlir::success();
}

mlir::LogicalResult DTensorAllGatherOp::verify() {
  DTensorAllGatherOp op = *this;
  const tensorflow::dtensor::Layout input_layout = op.getInputLayout();
  const tensorflow::dtensor::Layout output_layout = op.getOutputLayout();

  if (input_layout.rank() != output_layout.rank())
    return op.emitOpError()
           << "received input and output layouts of unequal ranks "
           << input_layout.rank() << " and " << output_layout.rank();

  for (int32_t i = 0; i < input_layout.rank(); ++i) {
    if (input_layout.sharding_spec(i) != output_layout.sharding_spec(i) &&
        tensorflow::dtensor::Layout::IsShardedDimension(
            output_layout.sharding_spec(i))) {
      return op.emitOpError()
             << "dimension " << i << " of output layout has sharding spec "
             << output_layout.sharding_spec(i)
             << " which is more sharded then the input layout spec "
             << input_layout.sharding_spec(i);
    }
  }

  RankedTensorType input_type =
      op.getInput().getType().dyn_cast<RankedTensorType>();
  if (!input_type) return mlir::success();

  if (input_type.getRank() != input_layout.rank())
    return op.emitOpError()
           << "input layout rank " << input_layout.rank()
           << " is not equal to input rank " << input_type.getRank();

  RankedTensorType output_type =
      op.getOutput().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return mlir::success();

  if (output_type.getRank() != output_layout.rank())
    return op.emitOpError()
           << "output layout rank " << output_layout.rank()
           << " is not equal to output rank " << output_type.getRank();

  std::vector<int64_t> computed_output_shape =
      output_layout.LocalShapeFromGlobalShape(
          input_layout.GlobalShapeFromLocalShape(input_type.getShape()));

  for (int32_t i = 0; i < computed_output_shape.size(); ++i) {
    if (computed_output_shape[i] != output_type.getShape()[i]) {
      return op.emitOpError()
             << "computed output shape " << computed_output_shape[i]
             << " at dimension " << i << " is not equal to actual output shape "
             << output_type.getShape()[i];
    }
  }

  return mlir::success();
}

mlir::LogicalResult DTensorAllScatterOp::verify() {
  DTensorAllScatterOp op = *this;
  const tensorflow::dtensor::Layout input_layout = op.getInputLayout();
  const tensorflow::dtensor::Layout output_layout = op.getOutputLayout();

  if (input_layout.rank() != output_layout.rank())
    return op.emitOpError()
           << "received input and output layouts of unequal ranks "
           << input_layout.rank() << " and " << output_layout.rank();

  for (int32_t i = 0; i < input_layout.rank(); ++i) {
    if (input_layout.sharding_spec(i) != output_layout.sharding_spec(i) &&
        tensorflow::dtensor::Layout::IsShardedDimension(
            input_layout.sharding_spec(i))) {
      return op.emitOpError()
             << "dimension " << i << " of input layout has sharding spec "
             << input_layout.sharding_spec(i)
             << " which is more sharded then the output layout spec "
             << output_layout.sharding_spec(i);
    }
  }

  RankedTensorType input_type =
      op.getInput().getType().dyn_cast<RankedTensorType>();
  if (!input_type) return mlir::success();

  if (input_type.getRank() != input_layout.rank())
    return op.emitOpError()
           << "input layout rank " << input_layout.rank()
           << " is not equal to input rank " << input_type.getRank();

  RankedTensorType output_type =
      op.getOutput().getType().dyn_cast<RankedTensorType>();
  if (!output_type) return mlir::success();

  if (output_type.getRank() != output_layout.rank())
    return op.emitOpError()
           << "output layout rank " << output_layout.rank()
           << " is not equal to output rank " << output_type.getRank();

  std::vector<int64_t> computed_output_shape =
      output_layout.LocalShapeFromGlobalShape(
          input_layout.GlobalShapeFromLocalShape(input_type.getShape()));

  for (int32_t i = 0; i < computed_output_shape.size(); ++i) {
    if (computed_output_shape[i] != output_type.getShape()[i]) {
      return op.emitOpError()
             << "computed output shape " << computed_output_shape[i]
             << " at dimension " << i << " is not equal to actual output shape "
             << output_type.getShape()[i];
    }
  }

  return mlir::success();
}

LogicalResult DTensorLayout::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  assert(operands.size() == 1);
  inferredReturnTypes.assign({operands[0].getType()});
  return success();
}

void DTensorOpAdderHook(TensorFlowDialect& dialect) {
  dialect.addOperations<
#define GET_OP_LIST
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.cc.inc"
      >();
}

int RegisterOnce() {
  TF_DIALECT_REGISTER_ADDITIONAL_OPERATIONS(DTensorOpAdderHook)
  return 0;
}

int RegisterDTensorTFOps() {
  static int r = RegisterOnce();
  return r;
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.cc.inc"

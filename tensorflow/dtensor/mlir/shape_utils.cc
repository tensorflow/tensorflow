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

#include "tensorflow/dtensor/mlir/shape_utils.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<llvm::ArrayRef<int64_t>> ExtractGlobalInputShape(
    mlir::OpOperand& input_value) {
  const int operand_index = input_value.getOperandNumber();
  auto input_defining_op = input_value.get().getDefiningOp();

  if (input_defining_op) {
    if (auto layout_op =
            llvm::dyn_cast<mlir::TF::DTensorLayout>(input_defining_op)) {
      auto global_shape = layout_op.getGlobalShape();
      if (!global_shape)
        return errors::Internal("global_shape does not have static rank");
      return *global_shape;
    }
    return ExtractGlobalOutputShape(cast<mlir::OpResult>(input_value.get()));
  }

  // If we reach this point, we're working with a function argument.
  auto op = input_value.getOwner();
  auto enclosing_function = op->getParentOfType<mlir::func::FuncOp>();
  if (!enclosing_function)
    return errors::InvalidArgument(
        llvm::formatv("Could not find global shape of {0}-th input to op: {1}",
                      operand_index, op->getName())
            .str());

  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(input_value.get());
  auto global_shape_attr =
      enclosing_function.getArgAttrOfType<mlir::TF::ShapeAttr>(
          block_arg.getArgNumber(), kGlobalShapeDialectAttr);
  if (!global_shape_attr)
    return errors::InvalidArgument(
        "`tf._global_shape` attribute of operation not found.");

  return global_shape_attr.getShape();
}

StatusOr<llvm::ArrayRef<int64_t>> ExtractGlobalOutputShape(
    mlir::OpResult result_value) {
  auto op = result_value.getOwner();
  const int output_index = result_value.getResultNumber();

  if (op->getOpResult(output_index).hasOneUse()) {
    auto user = op->getOpResult(output_index).getUses().begin().getUser();
    if (auto layout_op = mlir::dyn_cast<mlir::TF::DTensorLayout>(user)) {
      auto global_shape = layout_op.getGlobalShape();
      if (!global_shape)
        return errors::Internal("global_shape does not have static rank");
      return *global_shape;
    }
  }

  auto global_shape_attr = op->getAttrOfType<mlir::ArrayAttr>(kGlobalShape);
  if (!global_shape_attr)
    return errors::InvalidArgument(
        "`_global_shape` attribute of operation not found.");

  const int num_results = op->getNumResults();
  assert(global_shape_attr.size() == num_results);

  if (output_index >= op->getNumResults())
    return errors::InvalidArgument(
        llvm::formatv("Requested global shape of {0} output but op has only "
                      "{1} return values.",
                      output_index, num_results)
            .str());

  auto shape_attr = global_shape_attr[output_index];
  return llvm::cast<mlir::tf_type::ShapeAttr>(shape_attr).getShape();
}

namespace {

// Extracts attributes from a MLIR operation, including derived attributes, into
// one NamedAttrList.
mlir::NamedAttrList GetAllAttributesFromOperation(mlir::Operation* op) {
  mlir::NamedAttrList attr_list;
  attr_list.append(op->getAttrDictionary().getValue());

  if (auto derived = llvm::dyn_cast<mlir::DerivedAttributeOpInterface>(op)) {
    auto materialized = derived.materializeDerivedAttributes();
    attr_list.append(materialized.getValue());
  }

  return attr_list;
}

// Infers output shape of `op` given its local operand shape. For shape
// inference function that requires input operation to be a constant, if input
// operation is `DTensorLayout` op, then we use input of DTensorLayout op
// instead for correct constant matching.
mlir::LogicalResult InferShapeOfTFOpWithCustomOperandConstantFn(
    std::optional<mlir::Location> location, mlir::Operation* op,
    int64_t graph_version,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>& inferred_return_shapes) {
  if (auto type_op = llvm::dyn_cast<mlir::InferTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    llvm::SmallVector<mlir::Type, 4> inferred_return_types;
    auto result = type_op.inferReturnTypes(
        op->getContext(), location, op->getOperands(),
        mlir::DictionaryAttr::get(op->getContext(), attributes),
        op->getPropertiesStorage(), op->getRegions(), inferred_return_types);
    if (failed(result)) return mlir::failure();

    inferred_return_shapes.resize(inferred_return_types.size());
    for (const auto& inferred_return_type :
         llvm::enumerate(inferred_return_types)) {
      if (auto shaped_type =
              llvm::dyn_cast<mlir::ShapedType>(inferred_return_type.value())) {
        if (shaped_type.hasRank()) {
          inferred_return_shapes[inferred_return_type.index()] =
              mlir::ShapedTypeComponents(shaped_type.getShape(),
                                         shaped_type.getElementType());
        } else {
          inferred_return_shapes[inferred_return_type.index()] =
              mlir::ShapedTypeComponents(shaped_type.getElementType());
        }
      }
    }

    return mlir::success();
  }

  if (auto shape_type_op =
          llvm::dyn_cast<mlir::InferShapedTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    return shape_type_op.inferReturnTypeComponents(
        op->getContext(), location, op->getOperands(),
        mlir::DictionaryAttr::get(op->getContext(), attributes),
        op->getPropertiesStorage(), op->getRegions(), inferred_return_shapes);
  }

  // If `operand` is from DTensorLayout op, use input value of DTensorLayout op
  // instead.
  auto operand_as_constant_fn = [](mlir::Value operand) -> mlir::Attribute {
    while (auto input_op = llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
               operand.getDefiningOp())) {
      operand = input_op.getInput();
    }

    mlir::Attribute attr;
    if (matchPattern(operand, m_Constant(&attr))) return attr;
    return nullptr;
  };

  auto op_result_as_shape_fn =
      [](shape_inference::InferenceContext& ic,
         mlir::OpResult op_result) -> shape_inference::ShapeHandle {
    auto rt = llvm::dyn_cast<mlir::RankedTensorType>(op_result.getType());
    if (!rt || rt.getRank() != 1 || !rt.hasStaticShape()) return {};

    std::vector<shape_inference::DimensionHandle> dims(rt.getDimSize(0),
                                                       ic.UnknownDim());
    mlir::Attribute attr;
    if (matchPattern(op_result, m_Constant(&attr))) {
      auto elements = llvm::dyn_cast<mlir::DenseIntElementsAttr>(attr);
      if (elements)
        for (const auto& element :
             llvm::enumerate(elements.getValues<llvm::APInt>()))
          dims[element.index()] = ic.MakeDim(element.value().getSExtValue());
    }
    return ic.MakeShape(dims);
  };

  auto result_element_type_fn = [](int) -> mlir::Type { return nullptr; };

  return mlir::TF::InferReturnTypeComponentsForTFOp(
      location, op, graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn, inferred_return_shapes);
}

}  // namespace

absl::Status InferSPMDExpandedLocalShapeForResourceOutput(
    mlir::OpResult* op_result, const Layout& output_layout,
    mlir::MLIRContext* context) {
  if (llvm::isa<mlir::TF::ResourceType>(
          mlir::getElementTypeOrSelf(*op_result))) {
    TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> global_shape,
                        GetGlobalShapeOfValueFromDTensorLayout(*op_result));
    const std::vector<int64_t>& local_shape =
        output_layout.LocalShapeFromGlobalShape(global_shape);
    auto resource_type = llvm::dyn_cast<mlir::tf_type::ResourceType>(
        op_result->getType().cast<mlir::TensorType>().getElementType());

    auto sub_types = resource_type.getSubtypes();
    auto resource_arg_sub_type = sub_types.front();

    // The local shape that is to be assigned to this resource output.
    llvm::SmallVector<int64_t, 4> local_arg_shape(local_shape.begin(),
                                                  local_shape.end());

    auto local_variable_subtype = mlir::RankedTensorType::get(
        local_arg_shape, resource_arg_sub_type.getElementType());
    auto new_var_type = mlir::RankedTensorType::get(
        {},
        mlir::TF::ResourceType::get(
            mlir::ArrayRef<mlir::TensorType>{local_variable_subtype}, context));
    op_result->setType(new_var_type);
  }
  return absl::OkStatus();
}

mlir::Operation* InferSPMDExpandedLocalShape(mlir::Operation* op) {
  llvm::SmallVector<mlir::ShapedTypeComponents, 4> inferred_return_types;
  (void)InferShapeOfTFOpWithCustomOperandConstantFn(
      op->getLoc(), op, TF_GRAPH_DEF_VERSION, inferred_return_types);
  assert(inferred_return_types.size() == op->getNumResults());

  for (auto it : llvm::zip(inferred_return_types, op->getOpResults())) {
    const auto& return_type = std::get<0>(it);
    auto& op_result = std::get<1>(it);
    const auto element_type =
        llvm::cast<mlir::TensorType>(op_result.getType()).getElementType();

    if (return_type.hasRank()) {
      op_result.setType(
          mlir::RankedTensorType::get(return_type.getDims(), element_type));
    } else {
      op_result.setType(mlir::UnrankedTensorType::get(element_type));
    }
  }

  return op;
}

StatusOr<llvm::ArrayRef<int64_t>> GetShapeOfValue(const mlir::Value& value,
                                                  bool fail_on_dynamic) {
  // Getting the subtype or self allows supporting extracting the underlying
  // shape that variant or resource tensors point to.
  mlir::Type type = GetSubtypeOrSelf(value);
  if (auto ranked_type = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
    if (ranked_type.hasStaticShape() || !fail_on_dynamic)
      return ranked_type.getShape();
    else
      return errors::InvalidArgument("value shape is not static");
  }
  return errors::InvalidArgument("value type is not a RankedTensorType");
}

StatusOr<llvm::ArrayRef<int64_t>> GetGlobalShapeOfValueFromDTensorLayout(
    const mlir::Value& value) {
  if (mlir::isa<mlir::OpResult>(value) &&
      mlir::isa<mlir::TF::DTensorLayout>(value.getDefiningOp())) {
    auto layout_op = mlir::cast<mlir::TF::DTensorLayout>(value.getDefiningOp());
    if (layout_op.getGlobalShape()) return layout_op.getGlobalShape().value();
  } else if (value.hasOneUse() &&
             mlir::isa<mlir::TF::DTensorLayout>(*value.getUsers().begin())) {
    auto layout_op =
        mlir::cast<mlir::TF::DTensorLayout>(*value.getUsers().begin());
    if (layout_op.getGlobalShape()) return layout_op.getGlobalShape().value();
  }
  return errors::InvalidArgument(
      "consumer or producer of value is not a DTensorLayout");
}

}  // namespace dtensor
}  // namespace tensorflow

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

#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/export_utils.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

#define DEBUG_TYPE "tf-shape-inference-utils"

using ::tensorflow::int64;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace TF {

namespace {

// Extracts attributes from a MLIR operation, including derived attributes.
NamedAttrList GetAllAttributesFromOperation(Operation* op) {
  NamedAttrList attr_list;
  attr_list.append(op->getAttrDictionary().getValue());

  if (auto derived = dyn_cast<DerivedAttributeOpInterface>(op)) {
    auto materialized = derived.materializeDerivedAttributes();
    attr_list.append(materialized.getValue());
  }

  return attr_list;
}

// Extracts a PartialTensorShape from the MLIR type.
Optional<tensorflow::PartialTensorShape> GetShapeFromMlirType(Type t) {
  if (auto ranked_type = t.dyn_cast<RankedTensorType>()) {
    // Convert the MLIR shape indices (int64_t) to TensorFlow indices
    // (int64).
    ArrayRef<int64_t> shape = ranked_type.getShape();
    SmallVector<int64, 8> tf_shape(shape.begin(), shape.end());
    return tensorflow::PartialTensorShape(
        MutableArrayRefToSpan<int64>(tf_shape));
  }
  return None;
}

// Gets the subtype's shape and data type for `type`. Templated to support both
// ResourceType and VariantType.
template <typename T>
std::unique_ptr<std::vector<
    std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>
GetSubtypesHelper(Type type) {
  auto type_with_subtypes =
      type.cast<TensorType>().getElementType().dyn_cast<T>();
  if (!type_with_subtypes || type_with_subtypes.getSubtypes().empty()) {
    return nullptr;
  }
  auto shapes_and_types = std::make_unique<std::vector<
      std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>();
  for (auto subtype : type_with_subtypes.getSubtypes()) {
    auto shape = GetShapeFromMlirType(subtype);
    // handle_shapes_and_types requires all shapes to be known. So if any
    // subtype is unknown, clear the vector.
    if (!shape) {
      shapes_and_types = nullptr;
      break;
    }
    tensorflow::DataType dtype;
    auto status =
        tensorflow::ConvertToDataType(subtype.getElementType(), &dtype);
    assert(status.ok() && "Unknown element type");
    shapes_and_types->emplace_back(*shape, dtype);
  }
  return shapes_and_types;
}

// Gets the subtype's shape and data type for `type`.
std::unique_ptr<std::vector<
    std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>
GetSubtypes(Type type) {
  auto subclasses = GetSubtypesHelper<TF::ResourceType>(type);
  if (subclasses) return subclasses;
  return GetSubtypesHelper<TF::VariantType>(type);
}

// Returns a shape inference function call failure at `location`.
LogicalResult EmitErrorFromShapeFunction(Optional<Location> location,
                                         StringRef op_name,
                                         StringRef error_message) {
  LLVM_DEBUG(llvm::dbgs() << "Shape inference error for '" << op_name
                          << "': " << error_message << "\n");
  return emitOptionalError(
      location,
      llvm::formatv(
          "TensorFlow shape inference function errored for op '{0}': {1}",
          op_name, error_message)
          .str());
}

// Extracts shape from a shape handle and inference context.
Optional<SmallVector<int64_t, 8>> GetShapeFromHandle(InferenceContext& context,
                                                     const ShapeHandle& sh) {
  if (!context.RankKnown(sh)) return None;
  SmallVector<int64_t, 8> shape;
  for (int dim : llvm::seq<int>(0, context.Rank(sh)))
    shape.push_back(context.Value(context.Dim(sh, dim)));
  return shape;
}

// Creates a tensor type from a shape handle and element type.
TensorType CreateTensorType(InferenceContext& context, const ShapeHandle& sh,
                            Type element_type) {
  auto shape = GetShapeFromHandle(context, sh);
  if (shape.hasValue())
    return RankedTensorType::get(shape.getValue(), element_type);
  return UnrankedTensorType::get(element_type);
}

// Creates a ShapedTypeComponent from a shape handle and element type.
ShapedTypeComponents CreateShapedTypeComponents(InferenceContext& context,
                                                const ShapeHandle& sh,
                                                Type element_type) {
  auto shape = GetShapeFromHandle(context, sh);
  if (shape.hasValue())
    return ShapedTypeComponents(shape.getValue(), element_type);
  return ShapedTypeComponents(element_type);
}

// Runs TensorFlow shape inference associated to the op type registered in the
// TensorFlow op registry based on Graph version, operands, and attributes.
// Invoking this shape function will invoke conversions of parameters to the
// TensorFlow Graph equivalent data structures and back to MLIR equivalent data
// structures. This does not use a natively implemented shape inference in MLIR,
// and instead is temporary until shape functions are reimplemented/migrated to
// being in MLIR instead of the TensorFlow op registry.
LogicalResult InferReturnTypeComponentsFallback(
    MLIRContext* context, StringRef op_name, int64_t graph_version,
    Optional<Location> location, ValueRange operands,
    const NamedAttrList& attributes, OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  assert(op_name.startswith(TensorFlowDialect::getDialectNamespace()));
  // Drop the `tf.` prefix to query TF registry.
  std::string op_type =
      op_name.drop_front(TensorFlowDialect::getDialectNamespace().size() + 1)
          .str();

  // Get information from the registry and check if we have a shape function for
  // this op.
  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(op_type);
  if (!op_reg_data) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for unregistered op '"
                            << op_name << "'.\n");
    return emitOptionalError(location, "op is unregistered");
  }
  if (!op_reg_data->shape_inference_fn) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping inference for op without shape function '"
               << op_name << "'.\n");
    return emitOptionalError(location, "missing shape function");
  }

  // Convert the operation attributes to be able to use the InferenceContext
  // and the TensorFlow shape function.
  tensorflow::AttrValueMap converted_attributes;
  NamedAttrList attributes_to_convert;
  // Filter out unregistered attributes.
  for (const auto& attr_def : op_reg_data->op_def.attr())
    if (auto registered_attr = attributes.get(attr_def.name()))
      attributes_to_convert.set(attr_def.name(), registered_attr);

  auto attrs_status = tensorflow::ConvertAttributes(
      attributes_to_convert, /*attrs_to_ignore=*/{}, &converted_attributes);
  if (!attrs_status.ok()) {
    LLVM_DEBUG(llvm::dbgs() << "Error creating attribute map for '" << op_name
                            << "': " << attrs_status.error_message() << "\n");
    return emitOptionalError(
        location,
        "failed to convert attributes to proto map<string, AttrValue>");
  }

  // Collect an array with input values for constant operands and input shapes
  // for all the operands.
  std::vector<const tensorflow::Tensor*> input_tensors(operands.size());
  std::vector<tensorflow::PartialTensorShape> input_shapes(operands.size());
  std::vector<tensorflow::Tensor> tensors(operands.size());
  std::vector<std::unique_ptr<std::vector<
      std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>>
      handle_shapes_and_types(operands.size());
  for (auto it : llvm::enumerate(operands)) {
    Value operand = it.value();
    size_t index = it.index();

    // If the operand is constant, then convert it to Tensor.
    if (auto attr = operand_as_constant_fn(operand)) {
      tensorflow::Tensor* input_tensor = &tensors[index];
      auto status =
          tensorflow::ConvertToTensor(attr.cast<ElementsAttr>(), input_tensor);
      if (status.ok()) {
        input_tensors[index] = input_tensor;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Error converting input " << index
                                << " of op '" << op_name << "' to Tensor: "
                                << status.error_message() << "\n");
      }
    }

    Type operand_type = operand.getType();
    if (auto shape = GetShapeFromMlirType(operand_type)) {
      input_shapes[index] = *shape;
    }
    // Collect the handle shapes and types for a resource/variant.
    handle_shapes_and_types[index] = GetSubtypes(operand_type);
  }

  // Perform the shape inference using an InferenceContext with the input
  // shapes. This object is abstracting the information that the ShapeInference
  // function operates on.
  InferenceContext c(graph_version,
                     tensorflow::AttrSlice(&converted_attributes),
                     op_reg_data->op_def, input_shapes, input_tensors,
                     /*input_tensors_as_shapes=*/{}, handle_shapes_and_types);
  auto status = c.Run(op_reg_data->shape_inference_fn);
  if (!status.ok())
    return EmitErrorFromShapeFunction(location, op_name,
                                      status.error_message());

  // Determine if, during shape computation, the shape functions attempted to
  // query an input operand as shape where the input was not known/constant.
  bool requires_inputs =
      any_of(llvm::seq<int>(0, c.num_inputs()), [&](int input) {
        return c.requested_input_tensor_as_partial_shape(input) &&
               !input_tensors[input];
      });
  if (requires_inputs) {
    LLVM_DEBUG(llvm::dbgs() << "\trequired input\n");
    std::vector<ShapeHandle> input_tensors_as_shapes;
    for (int input : llvm::seq<int>(0, c.num_inputs())) {
      if (c.requested_input_tensor_as_partial_shape(input) &&
          !input_tensors[input]) {
        LLVM_DEBUG(llvm::dbgs() << "Requesting " << input << " as shape\n");
        auto op_result = operands[input].dyn_cast<OpResult>();
        if (!op_result) continue;
        // Resize on first valid shape computed.
        input_tensors_as_shapes.resize(c.num_inputs());
        auto handle = op_result_as_shape_fn(c, op_result);
        LLVM_DEBUG(llvm::dbgs() << "Requested " << input << " as shape "
                                << (handle.Handle() ? "found" : "not found"));
        if (handle.Handle()) input_tensors_as_shapes[input] = handle;
      }
    }

    // Attempt to compute the unknown operands as shapes.
    // Note: in the case where no partial outputs could be computed, this
    // would be empty.
    if (!input_tensors_as_shapes.empty()) {
      c.set_input_tensors_as_shapes(input_tensors_as_shapes);
      auto status = c.Run(op_reg_data->shape_inference_fn);
      if (!status.ok())
        return EmitErrorFromShapeFunction(location, op_name,
                                          status.error_message());
    }
  }

  // Update the shape for each of the operation result if the InferenceContext
  // has more precise shapes recorded.
  for (int output : llvm::seq<int>(0, c.num_outputs())) {
    ShapeHandle shape_handle = c.output(output);
    LLVM_DEBUG(llvm::dbgs() << "Inferred output " << output << " : "
                            << c.DebugString(shape_handle) << "\n");

    Type new_element_type = result_element_type_fn(output);
    // Populate the handle shapes for a resource/variant.
    if (new_element_type &&
        new_element_type.isa<TF::ResourceType, TF::VariantType>()) {
      auto handle_shapes_types = c.output_handle_shapes_and_types(output);
      if (handle_shapes_types) {
        SmallVector<TensorType, 1> subtypes;
        Builder b(context);
        for (const auto& shape_n_type : *handle_shapes_types) {
          Type element_type;
          auto status =
              tensorflow::ConvertDataType(shape_n_type.dtype, b, &element_type);
          assert(status.ok() && "Unknown element type");
          subtypes.push_back(
              CreateTensorType(c, shape_n_type.shape, element_type));
        }
        if (new_element_type.isa<TF::ResourceType>()) {
          new_element_type = TF::ResourceType::get(subtypes, context);
        } else {
          new_element_type = TF::VariantType::get(subtypes, context);
        }
      }
    }
    inferred_return_shapes.push_back(
        CreateShapedTypeComponents(c, shape_handle, new_element_type));
  }

  return success();
}

}  // namespace

LogicalResult InferReturnTypeComponentsForTFOp(
    Optional<Location> location, Operation* op, int64_t graph_version,
    OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  auto attributes = GetAllAttributesFromOperation(op);
  return InferReturnTypeComponentsFallback(
      op->getContext(), op->getName().getStringRef(), graph_version, location,
      op->getOperands(), attributes, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn, inferred_return_shapes);
}

LogicalResult InferReturnTypeComponentsForTFOp(
    Optional<Location> location, Operation* op, int64_t graph_version,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  if (auto type_op = dyn_cast<InferTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    SmallVector<Type, 4> inferred_return_types;
    auto result = type_op.inferReturnTypes(
        op->getContext(), location, op->getOperands(),
        DictionaryAttr::get(attributes, op->getContext()), op->getRegions(),
        inferred_return_types);
    if (failed(result)) return failure();

    inferred_return_shapes.resize(inferred_return_types.size());
    for (auto inferred_return_type : llvm::enumerate(inferred_return_types)) {
      if (auto shaped_type =
              inferred_return_type.value().dyn_cast<ShapedType>()) {
        if (shaped_type.hasRank()) {
          inferred_return_shapes[inferred_return_type.index()] =
              ShapedTypeComponents(shaped_type.getShape(),
                                   shaped_type.getElementType());
        } else {
          inferred_return_shapes[inferred_return_type.index()] =
              ShapedTypeComponents(shaped_type.getElementType());
        }
      }
    }

    return success();
  }

  if (auto shape_type_op = dyn_cast<InferShapedTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    return shape_type_op.inferReturnTypeComponents(
        op->getContext(), location, op->getOperands(),
        DictionaryAttr::get(attributes, op->getContext()), op->getRegions(),
        inferred_return_shapes);
  }

  auto operand_as_constant_fn = [](Value operand) -> Attribute {
    Attribute attr;
    if (matchPattern(operand, m_Constant(&attr))) return attr;
    return nullptr;
  };

  auto op_result_as_shape_fn = [](InferenceContext& ic,
                                  OpResult op_result) -> ShapeHandle {
    auto rt = op_result.getType().dyn_cast<RankedTensorType>();
    if (!rt || rt.getRank() != 1 || !rt.hasStaticShape()) return {};

    std::vector<DimensionHandle> dims(rt.getDimSize(0), ic.UnknownDim());
    Attribute attr;
    if (matchPattern(op_result, m_Constant(&attr))) {
      auto elements = attr.dyn_cast<DenseIntElementsAttr>();
      if (elements)
        for (auto element : llvm::enumerate(elements.getIntValues()))
          dims[element.index()] = ic.MakeDim(element.value().getSExtValue());
    }
    return ic.MakeShape(dims);
  };

  auto result_element_type_fn = [](int) -> Type { return nullptr; };

  return InferReturnTypeComponentsForTFOp(
      location, op, graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn, inferred_return_shapes);
}

}  // namespace TF
}  // namespace mlir

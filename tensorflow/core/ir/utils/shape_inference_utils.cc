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

#include "tensorflow/core/ir/utils/shape_inference_utils.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_tensor.h"
#include "tensorflow/core/ir/importexport/convert_types.h"
#include "tensorflow/core/ir/importexport/graphdef_export.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

#define DEBUG_TYPE "tfg-shape-inference-utils"

using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace mlir {
namespace tfg {

namespace {

// Get the tensorflow op name.
llvm::StringRef GetTensorFlowOpName(Operation* inst) {
  llvm::StringRef op_name = inst->getName().stripDialect();
  // Control dialect NextIteration sink ends with ".sink" and Executor dialect
  // NextIteration sink ends with ".Sink".
  if (!op_name.consume_back(".sink")) op_name.consume_back(".Sink");
  return op_name;
}

// Extracts attributes from a MLIR operation, including derived attributes, into
// one NamedAttrList.
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
// Some MLIR shapes may fail to be represented as PartialTensorShape, e.g.
// those where num_elements overflows.
// TODO(tlongeri): Should num_elements overflow be handled by the MLIR
// verifier? Are there other cases?
std::optional<tensorflow::PartialTensorShape> GetShapeFromMlirType(Type t) {
  if (auto ranked_type = llvm::dyn_cast<RankedTensorType>(t)) {
    tensorflow::PartialTensorShape shape;
    const absl::Status status =
        tensorflow::PartialTensorShape::BuildPartialTensorShape(
            ConvertMlirShapeToTF(ranked_type.getShape()), &shape);
    if (status.ok()) return shape;
  }
  return std::nullopt;
}

// Extracts a PartialTensorShape from the MLIR attr.
std::optional<tensorflow::PartialTensorShape> GetShapeFromMlirAttr(Value v) {
  // Function arguments may have shape attr to describe its output shape.
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation* parent_op = arg.getOwner()->getParentOp();
    if (auto func_op = llvm::dyn_cast<FunctionOpInterface>(parent_op)) {
      int arg_idx = arg.getArgNumber();
      auto attrs =
          func_op.getArgAttrOfType<ArrayAttr>(arg_idx, "tf._output_shapes");
      if (!attrs || attrs.size() != 1) return std::nullopt;

      // "tf._output_shapes" in certain models may not store the shape as
      // ShapeAttr, ignore them because we don't know how to interpret it.
      auto shape_attr = llvm::dyn_cast<tf_type::ShapeAttr>(attrs[0]);
      if (shape_attr && shape_attr.hasRank())
        return tensorflow::PartialTensorShape(shape_attr.getShape());
    }
  }
  return std::nullopt;
}

// Gets the subtype's shape and data type for `type`. Templated to support both
// ResourceType and VariantType.
template <typename T>
std::unique_ptr<std::vector<
    std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>
GetSubtypesHelper(Type type) {
  auto type_with_subtypes =
      llvm::dyn_cast<T>(llvm::cast<TensorType>(type).getElementType());
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
    auto status = ConvertToDataType(subtype.getElementType(), &dtype);
    assert(status.ok() && "Unknown element type");
    shapes_and_types->emplace_back(*shape, dtype);
  }
  return shapes_and_types;
}

// Gets the subtype's shape and data type for `type`.
std::unique_ptr<std::vector<
    std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>
GetSubtypes(Type type) {
  auto subclasses = GetSubtypesHelper<tf_type::ResourceType>(type);
  if (subclasses) return subclasses;
  return GetSubtypesHelper<tf_type::VariantType>(type);
}

// Log a shape inference function call failure.
LogicalResult ReportErrorFromShapeFunction(std::optional<Location> location,
                                           llvm::StringRef op_name,
                                           llvm::StringRef error_message) {
  VLOG(3) << "TensorFlow shape inference function errored for op '"
          << op_name.data() << "': " << error_message.data();
  return failure();
}

// Extracts shape from a shape handle and inference context.
std::optional<SmallVector<int64_t, 8>> GetShapeFromHandle(
    InferenceContext& context, const ShapeHandle& sh) {
  if (!context.RankKnown(sh)) return std::nullopt;
  SmallVector<int64_t, 8> shape;
  for (int dim : llvm::seq<int>(0, context.Rank(sh)))
    shape.push_back(context.Value(context.Dim(sh, dim)));
  return shape;
}

// Creates a tensor type from a shape handle and element type.
TensorType CreateTensorType(InferenceContext& context, const ShapeHandle& sh,
                            Type element_type) {
  auto shape = GetShapeFromHandle(context, sh);
  if (shape.has_value())
    return GetTypeFromTFTensorShape(shape.value(), element_type, {});
  return UnrankedTensorType::get(element_type);
}

// Creates a ShapedTypeComponent from a shape handle and element type.
ShapedTypeComponents CreateShapedTypeComponents(InferenceContext& context,
                                                const ShapeHandle& sh,
                                                Type element_type) {
  auto shape = GetShapeFromHandle(context, sh);
  if (shape.has_value())
    return ShapedTypeComponents(ConvertTFShapeToMlir(shape.value()),
                                element_type);
  return ShapedTypeComponents(element_type);
}

}  // namespace

LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, ValueRange operands,
    int64_t graph_version, OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    GetAttrValuesFn get_attr_values_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  llvm::StringRef op_name = GetTensorFlowOpName(op);

  // Get information from the registry and check if we have a shape function for
  // this op.
  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(op_name.str());
  if (!op_reg_data) {
    VLOG(3) << "Skipping inference for unregistered op '" << op_name.data()
            << "'.\n";
    return failure();
  }
  if (!op_reg_data->shape_inference_fn) {
    VLOG(3) << "Skipping inference for op without shape function '"
            << op_name.data() << "'.\n";
    return failure();
  }

  // Convert the operation to NodeDef to get the AttrValue to be able to use the
  // InferenceContext and the TensorFlow shape function.
  tensorflow::AttrValueMap attributes;

  if (get_attr_values_fn) {
    absl::Status status =
        get_attr_values_fn(op, op_name, op_reg_data,
                           /*ignore_unregistered_attrs=*/true, &attributes);
    if (!status.ok()) {
      VLOG(3) << op_name.data()
              << " failed to get AttrValue: " << status.message();
      return failure();
    }
  } else {
    auto* dialect = cast<TFGraphDialect>(op->getDialect());
    tensorflow::NodeDef node_def;
    absl::Status status = ConvertToNodeDef(
        op, &node_def, dialect,
        [&](Value value) { return GetValueName(value, dialect); });
    if (!status.ok()) {
      VLOG(3) << op_name.data()
              << " failed to be converted to NodeDef: " << status.message();
      return failure();
    }
    attributes = node_def.attr();
  }

  // Collect an array with input values for constant operands and input shapes
  // for all the operands.
  const int num_operands = operands.size();
  std::vector<tensorflow::PartialTensorShape> input_shapes(num_operands);
  std::vector<std::unique_ptr<std::vector<
      std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>>
      handle_shapes_and_types(num_operands);
  for (const auto& it : llvm::enumerate(operands)) {
    Value operand = it.value();
    size_t index = it.index();

    Type operand_type = operand.getType();
    if (auto shape = GetShapeFromMlirType(operand_type)) {
      input_shapes[index] = *shape;
    } else if (auto shape = GetShapeFromMlirAttr(operand)) {
      input_shapes[index] = *shape;
    }
    // Collect the handle shapes and types for a resource/variant.
    handle_shapes_and_types[index] = GetSubtypes(operand_type);
  }

  // Perform the shape inference using an InferenceContext with the input
  // shapes. This object is abstracting the information that the ShapeInference
  // function operates on.
  InferenceContext c(graph_version, tensorflow::AttrSlice(&attributes),
                     op_reg_data->op_def, input_shapes, /*input_tensors*/ {},
                     /*input_tensors_as_shapes=*/{}, handle_shapes_and_types);
  if (!c.construction_status().ok()) {
    VLOG(3) << "InferenceContext construction failed on " << op_name.data()
            << ": " << c.construction_status().message();
    return failure();
  }
  auto status = c.Run(op_reg_data->shape_inference_fn);
  if (!status.ok()) {
    return ReportErrorFromShapeFunction(location, op_name,
                                        std::string(status.message()));
  }

  std::vector<const tensorflow::Tensor*> input_tensors(num_operands);
  std::vector<tensorflow::Tensor> tensors(num_operands);
  std::vector<ShapeHandle> input_tensors_as_shapes(num_operands);

  // Determine if, during shape computation, the shape functions attempted to
  // query the input or input as shape where the input wasn't available.
  auto requires_inputs = [&]() {
    return any_of(llvm::seq<int>(0, c.num_inputs()), [&](int input) {
      return !input_tensors[input] &&
             (c.requested_input_tensor(input) ||
              c.requested_input_tensor_as_partial_shape(input));
    });
  };

  // Iterate until no new inputs are requested. Some shape functions may not
  // request all inputs upfront and can return early so this requires multiple
  // iterations.
  while (requires_inputs()) {
    VLOG(4) << "\tfeeding new inputs or input as partial shapes\n";

    bool has_new_inputs = false;
    for (int input : llvm::seq<int>(0, c.num_inputs())) {
      if (input_tensors[input]) continue;

      if (c.requested_input_tensor(input)) {
        if (auto attr = llvm::dyn_cast_if_present<ElementsAttr>(
                operand_as_constant_fn(op->getOperand(input)))) {
          VLOG(4) << "Requesting " << input << " as constant\n";
          tensorflow::Tensor* input_tensor = &tensors.at(input);
          auto status = ConvertToTensor(attr, input_tensor);
          if (status.ok()) {
            input_tensors.at(input) = input_tensor;
            has_new_inputs = true;
          } else {
            VLOG(4) << "Error converting input " << input << " of op '"
                    << op_name.data() << "' to Tensor: " << status.message()
                    << "\n";
          }
        }
      }

      if (c.requested_input_tensor_as_partial_shape(input) &&
          !input_tensors[input] && !input_tensors_as_shapes[input].Handle()) {
        VLOG(4) << "Requesting " << input << " as shape\n";
        auto op_result = dyn_cast<OpResult>(op->getOperand(input));
        if (!op_result) continue;
        // Resize on first valid shape computed.
        auto handle = op_result_as_shape_fn(c, op_result);
        VLOG(4) << "Requested " << input << " as shape "
                << (handle.Handle() ? "found" : "not found");
        if (handle.Handle()) {
          input_tensors_as_shapes[input] = handle;
          has_new_inputs = true;
        }
      }
    }

    if (!has_new_inputs) break;

    c.set_input_tensors(input_tensors);
    c.set_input_tensors_as_shapes(input_tensors_as_shapes);
    auto status = c.Run(op_reg_data->shape_inference_fn);
    if (!status.ok()) {
      return ReportErrorFromShapeFunction(location, op_name,
                                          std::string(status.message()));
    }
  }

  // Update the shape for each of the operation result if the InferenceContext
  // has more precise shapes recorded.
  for (int output : llvm::seq<int>(0, c.num_outputs())) {
    ShapeHandle shape_handle = c.output(output);
    VLOG(4) << "Inferred output " << output << " : "
            << c.DebugString(shape_handle) << "\n";

    Type new_element_type = result_element_type_fn(output);
    // Populate the handle shapes for a resource/variant.
    if (new_element_type &&
        isa<tf_type::ResourceType, tf_type::VariantType>(new_element_type)) {
      auto handle_shapes_types = c.output_handle_shapes_and_types(output);
      if (handle_shapes_types) {
        SmallVector<TensorType, 1> subtypes;
        Builder b(op->getContext());
        for (const auto& shape_n_type : *handle_shapes_types) {
          Type element_type;
          auto status = ConvertDataType(shape_n_type.dtype, b, &element_type);
          assert(status.ok() && "Unknown element type");
          subtypes.push_back(
              CreateTensorType(c, shape_n_type.shape, element_type));
        }
        if (isa<tf_type::ResourceType>(new_element_type)) {
          new_element_type =
              tf_type::ResourceType::get(subtypes, op->getContext());
        } else {
          new_element_type =
              tf_type::VariantType::get(subtypes, op->getContext());
        }
      }
    }
    inferred_return_shapes.push_back(
        CreateShapedTypeComponents(c, shape_handle, new_element_type));
  }

  return success();
}

LogicalResult InferReturnTypeComponentsForTFOp(
    std::optional<Location> location, Operation* op, ValueRange operands,
    int64_t graph_version, OperandAsConstantFn operand_as_constant_fn,
    OpResultAsShapeFn op_result_as_shape_fn,
    ResultElementTypeFn result_element_type_fn,
    SmallVectorImpl<ShapedTypeComponents>& inferred_return_shapes) {
  return InferReturnTypeComponentsForTFOp(
      location, op, operands, graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn,
      /*get_attr_values_fn=*/nullptr, inferred_return_shapes);
}

}  // namespace tfg
}  // namespace mlir

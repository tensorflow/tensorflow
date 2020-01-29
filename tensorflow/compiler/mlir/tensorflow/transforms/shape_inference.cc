/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"

#include <cstdint>
#include <initializer_list>
#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Block.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "mlir/Transforms/FoldUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.pb.h"

#define DEBUG_TYPE "tf-shape-inference"

using ::tensorflow::int64;

namespace mlir {
namespace TF {
namespace {
Optional<llvm::SmallVector<mlir::Type, 4>> InferShapeForFunctionReturnType(
    FuncOp func) {
  // Only infer shape when there is one return op for now.
  if (!has_single_element(func.getBody()) || func.front().empty()) {
    return None;
  }

  // Find the return type.
  auto return_op = dyn_cast<mlir::ReturnOp>(func.front().back());
  if (!return_op) {
    return None;
  }

  // Manually fold tf.Cast that precedes the return instruction and only differs
  // in shape refinement level.
  for (OpOperand& arg_op : return_op.getOperation()->getOpOperands()) {
    Operation* arg_defining_op = arg_op.get().getDefiningOp();
    if (auto cast_op = dyn_cast_or_null<CastOp>(arg_defining_op)) {
      // Shape inference should not change the element type.
      if (cast_op.SrcT() != cast_op.DstT()) continue;
      // We only refine the result shape if the result a dynamic shape, the
      // input has static shape, and the two shapes are compatible.
      auto has_static_shape = [](const Value value) {
        auto shaped_type = value.getType().dyn_cast<ShapedType>();
        return shaped_type && shaped_type.hasStaticShape();
      };
      Value input = cast_op.x();
      Value result = cast_op.y();
      if (!has_static_shape(input) || has_static_shape(result) ||
          failed(verifyCompatibleShape(input.getType(), result.getType())))
        continue;

      arg_op.set(cast_op.x());
      if (cast_op.y().use_empty()) cast_op.erase();
    }
  }

  return llvm::to_vector<4>(return_op.getOperandTypes());
}

// Returns if the shape inference pass supports an op outside the TF dialect.
bool IsSupportedNonTFOp(Operation* op) {
  return isa<tf_executor::YieldOp>(op) || isa<tf_executor::IslandOp>(op) ||
         isa<tf_executor::FetchOp>(op) || isa<tf_executor::GraphOp>(op) ||
         isa<tf_executor::NextIterationSinkOp>(op) || isa<ReturnOp>(op) ||
         isa<tf_device::ReturnOp>(op);
}

// Inserts tf.Cast operation when changing the type of a result if the user is
// not a TF operation, as we can't guarantee that the new type will be OK.
void AddCastBackForUnsupportedNonTFUses(Operation* op, Value result,
                                        Dialect* tf_dialect, Type old_type) {
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  // A tf.Cast operation is lazily created on the first uses that isn't a TF
  // operation.
  TF::CastOp cast_op;
  auto get_cast_op = [&]() {
    if (!cast_op)
      cast_op =
          builder.create<TF::CastOp>(op->getLoc(), old_type, result,
                                     /*truncate=*/builder.getBoolAttr(false));
    return mlir::Value(cast_op);
  };
  for (OpOperand& use : llvm::make_early_inc_range(result.getUses())) {
    if (use.getOwner()->getDialect() != tf_dialect &&
        !IsSupportedNonTFOp(use.getOwner()))
      use.set(get_cast_op());
  }
}

// Extracts a PartialTensorShape from the MLIR type.
Optional<tensorflow::PartialTensorShape> GetShapeFromMlirType(Type t) {
  if (auto ranked_type = t.dyn_cast<RankedTensorType>()) {
    // Convert the MLIR shape indices (int64_t) to TensorFlow indices
    // (int64).
    ArrayRef<int64_t> shape = ranked_type.getShape();
    SmallVector<int64, 8> tf_shape(shape.begin(), shape.end());
    return tensorflow::PartialTensorShape({tf_shape.data(), tf_shape.size()});
  }
  return None;
}

// Passes the operand shapes/types to the op's results.
bool InferShapeForPassThroughOps(OperandRange pass_through_operands,
                                 Operation* op, Dialect* tf_dialect) {
  bool changed = false;
  for (auto entry : llvm::zip(pass_through_operands, op->getResults())) {
    Type operand_type = std::get<0>(entry).getType();
    Value result = std::get<1>(entry);
    if (result.getType() == operand_type) continue;
    AddCastBackForUnsupportedNonTFUses(op, result, tf_dialect,
                                       result.getType());
    result.setType(operand_type);
    changed = true;
  }
  return changed;
}

// Infers shape for necessary ops that are not in the TF dialect.
bool InferShapeForNonTFDialectOperation(Operation* op, Dialect* tf_dialect) {
  if (auto graph_op = dyn_cast<tf_executor::GraphOp>(op)) {
    return InferShapeForPassThroughOps(graph_op.GetFetch().fetches(), op,
                                       tf_dialect);
  }
  if (auto island_op = dyn_cast<tf_executor::IslandOp>(op)) {
    return InferShapeForPassThroughOps(island_op.GetYield().fetches(), op,
                                       tf_dialect);
  }
  if (auto iter_sink = dyn_cast<tf_executor::NextIterationSinkOp>(op)) {
    auto iter_source = cast<tf_executor::NextIterationSourceOp>(
        iter_sink.token().getDefiningOp());
    return InferShapeForPassThroughOps(
        iter_sink.getOperands().drop_front().take_front(), iter_source,
        tf_dialect);
  }
  return false;
}

}  // namespace

bool InferShapeForSingleOperation(Operation* op, Dialect* tf_dialect,
                                  int64_t graph_version) {
  assert(tf_dialect == op->getDialect());

  // If no result for this op needs shape inference, we have a fast-path return.
  // But if the type is a resource, we do not skip it because we might not have
  // the handle shapes.
  if (llvm::all_of(op->getResultTypes(), [](Type type) {
        auto shape_type = type.dyn_cast<ShapedType>();
        return !shape_type ||
               (shape_type.hasStaticShape() &&
                !shape_type.getElementType().isa<TF::ResourceType>());
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for statically shaped op '"
                            << op->getName() << "'.\n";);
    return false;
  }

  // tf.Cast are only inferred if they have at least one user in the tf dialect.
  // This is necessary to avoid reprocessing the tf.Cast that are inserted at
  // the end of this function.
  if (isa<CastOp>(op) &&
      llvm::all_of(op->getResult(0).getUsers(), [&](Operation* user) {
        return user->getDialect() != tf_dialect;
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for tf.Cast with no TF "
                               "dialect operation users '"
                            << *op << "'.\n";);
    return false;
  }

  StringRef op_name = op->getName().getStringRef();
  // Drop the `tf.` prefix to query TF registry.
  auto node_name =
      op_name.drop_front(TensorFlowDialect::getDialectNamespace().size() + 1);

  // Get information from the registry and check if we have a shape function for
  // this op.
  const tensorflow::OpRegistrationData* op_reg_data =
      tensorflow::OpRegistry::Global()->LookUp(node_name.data());
  if (!op_reg_data) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for unregistered op '"
                            << op->getName() << "'.\n";);
    return false;
  }
  if (op_reg_data->shape_inference_fn == nullptr) {
    LLVM_DEBUG(llvm::dbgs()
                   << "Skipping inference for op without shape function '"
                   << op->getName() << "'.\n";);
    return false;
  }

  // Convert the operation to a NodeDef to be able to use the InferenceContext
  // and the TensorFlow shape function.
  auto node_def_or = tensorflow::ConvertTFDialectOpToNodeDef(
      op, node_name, /*ignore_unregistered_attrs=*/true);
  if (!node_def_or.ok()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Error converting op '" << *op << "' to NodeDef: "
               << node_def_or.status().error_message() << "\n");
    return false;
  }
  std::unique_ptr<tensorflow::NodeDef> node_def =
      std::move(node_def_or).ValueOrDie();

  // Collect an array with input values for constant operands and input shapes
  // for all the operands.
  std::vector<const tensorflow::Tensor*> input_tensors(op->getNumOperands());
  std::vector<tensorflow::PartialTensorShape> input_shapes(
      op->getNumOperands());
  std::vector<tensorflow::Tensor> tensors(op->getNumOperands());
  std::vector<std::unique_ptr<std::vector<
      std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>>
      handle_shapes_and_types(op->getNumOperands());
  for (auto it : llvm::enumerate(op->getOperands())) {
    Value operand = it.value();
    size_t index = it.index();

    // If the operand is constant, then convert it to Tensor.
    ElementsAttr attr;
    if (matchPattern(operand, m_Constant(&attr))) {
      tensorflow::Tensor* input_tensor = &tensors[index];
      auto status = tensorflow::ConvertToTensor(attr, input_tensor);
      if (status.ok()) {
        input_tensors[index] = input_tensor;
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "Error converting input " << index << " of op '" << *op
                   << "' to Tensor: " << status.error_message() << "\n");
      }
    }

    Type operand_type = operand.getType();
    if (auto shape = GetShapeFromMlirType(operand_type)) {
      input_shapes[index] = *shape;
    }
    // Collect the handle shapes and types for a resource.
    if (auto resource_type = operand_type.cast<TensorType>()
                                 .getElementType()
                                 .dyn_cast<TF::ResourceType>()) {
      if (resource_type.getSubtypes().empty()) continue;
      auto shapes_and_types = absl::make_unique<std::vector<
          std::pair<tensorflow::PartialTensorShape, tensorflow::DataType>>>();
      for (auto subtype : resource_type.getSubtypes()) {
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
      handle_shapes_and_types[index] = std::move(shapes_and_types);
    }
  }

  // Perform the shape inference using an InferenceContext with the input
  // shapes. This object is abstracting the information that the ShapeInference
  // function operates on.
  tensorflow::shape_inference::InferenceContext c(
      graph_version, *node_def, op_reg_data->op_def, input_shapes,
      input_tensors, /*input_tensors_as_shapes=*/{}, handle_shapes_and_types);
  auto status = c.Run(op_reg_data->shape_inference_fn);
  if (!status.ok()) {
    LLVM_DEBUG(llvm::dbgs() << "Shape inference error for '" << *op
                            << "': " << status.error_message() << "\n");
    return false;
  }

  assert(c.num_outputs() == op->getNumResults() &&
         "inference context matches the MLIR number of results.");

  // Update the shape for each of the operation result if the InferenceContext
  // has more precise shapes recorded.
  bool changed = false;
  for (int output : llvm::seq<int>(0, c.num_outputs())) {
    // Skip already statically shaped results.
    Value result = op->getResult(output);
    auto shaped_type = result.getType().dyn_cast<ShapedType>();
    if (!shaped_type || shaped_type.hasStaticShape()) continue;

    tensorflow::shape_inference::ShapeHandle shape_handle = c.output(output);
    LLVM_DEBUG(llvm::dbgs() << "Inferred output " << output << " : "
                            << c.DebugString(shape_handle) << "\n");
    auto get_tensor_type =
        [&c](const tensorflow::shape_inference::ShapeHandle& sh,
             Type element_type) -> TensorType {
      if (!c.RankKnown(sh)) return UnrankedTensorType::get(element_type);
      // Convert the shape from TensorFlow (int64) to MLIR (int64_t).
      SmallVector<int64_t, 8> shape;
      for (int dim : llvm::seq<int>(0, c.Rank(sh)))
        shape.push_back(c.Value(c.Dim(sh, dim)));
      return RankedTensorType::get(shape, element_type);
    };
    auto new_element_type = shaped_type.getElementType();
    // Populate the handle shapes for a resource.
    if (auto resource_type = new_element_type.dyn_cast<TF::ResourceType>()) {
      auto handle_shapes_types = c.output_handle_shapes_and_types(output);
      if (handle_shapes_types) {
        llvm::SmallVector<mlir::TensorType, 1> subtypes;
        OpBuilder b(op);
        for (const auto& shape_n_type : *handle_shapes_types) {
          Type element_type;
          auto status =
              tensorflow::ConvertDataType(shape_n_type.dtype, b, &element_type);
          assert(status.ok() && "Unknown element type");
          subtypes.push_back(get_tensor_type(shape_n_type.shape, element_type));
        }
        new_element_type = TF::ResourceType::get(subtypes, op->getContext());
      }
    }
    auto new_type = get_tensor_type(shape_handle, new_element_type);
    if (result.getType() == new_type) continue;
    // Inserts a cast back to the original type if any user is not in the TF
    // dialect.
    AddCastBackForUnsupportedNonTFUses(op, result, tf_dialect,
                                       result.getType());
    // Finally we inferred the shape and replace the type for this result.
    result.setType(new_type);
    changed = true;
  }
  if (changed)
    LLVM_DEBUG(llvm::dbgs()
               << "Modified after shape inference: '" << *op << "'\n");
  return changed;
}

// Updates input types and refine shapes inside body of functions that are
// attached to ControlFlow ops (If/While). These functions include Then/Else
// branches of IfOp and Cond/Body functions of WhileOp. These functions share
// following common properties:
//   1) They are never reused, ie. having a single use in module.
//   2) Their input types match those of their parent ops (excluding inputs like
//      predicate).
// Returns a boolean indicating whether any change has been applied.
LogicalResult RefineShapeForControlFlowFunc(FuncOp func,
                                            llvm::ArrayRef<Type> input_types,
                                            int64_t graph_version,
                                            int64_t max_iteration) {
  ModuleOp module = func.getParentOfType<ModuleOp>();
  auto func_uses = SymbolTable::getSymbolUses(func, &module.getBodyRegion());
  int num_uses = std::distance(func_uses->begin(), func_uses->end());
  if (num_uses != 1) {
    func.emitError(llvm::formatv(
        "expected control flow function {0} to have exactly 1 use, found {1}.",
        func.getName(), num_uses));
    return failure();
  }

  FunctionType func_type = func.getType();
  if (input_types == func_type.getInputs()) return success();

  func.setType(FunctionType::get(input_types, func_type.getResults(),
                                 func.getContext()));

  for (auto arg_and_idx : llvm::enumerate(func.getArguments())) {
    arg_and_idx.value().setType(input_types[arg_and_idx.index()]);
  }

  auto res =
      InferShapeUntilFixPoint(&func.getBody(), graph_version, max_iteration);
  if (failed(res)) return res;

  auto new_return_types = InferShapeForFunctionReturnType(func);
  if (new_return_types.hasValue()) {
    func.setType(FunctionType::get(input_types, new_return_types.getValue(),
                                   func.getContext()));
  }

  return success();
}

LogicalResult PropagateShapeToFunctions(
    ModuleOp module, Operation::operand_type_range input_types,
    llvm::ArrayRef<StringRef> func_names, int64_t graph_version,
    int64_t max_iteration) {
  bool success = true;
  auto types = llvm::to_vector<4>(input_types);
  for (auto func_name : func_names) {
    FuncOp func = module.lookupSymbol<FuncOp>(func_name);
    if (failed(RefineShapeForControlFlowFunc(func, types, graph_version,
                                             max_iteration))) {
      success = false;
    }
  }
  return mlir::success(success);
}

LogicalResult PropagateShapeIntoAttachedFunctions(Operation* op,
                                                  int64_t graph_version,
                                                  int64_t max_iteration) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  if (auto if_op = dyn_cast<TF::IfOp>(op)) {
    return PropagateShapeToFunctions(
        module, llvm::drop_begin(if_op.getOperandTypes(), 1),
        {if_op.then_branch(), if_op.else_branch()}, graph_version,
        max_iteration);
  } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
    return PropagateShapeToFunctions(module, while_op.getOperandTypes(),
                                     {while_op.cond(), while_op.body()},
                                     graph_version, max_iteration);
  } else if (auto call_op = dyn_cast<TF::PartitionedCallOp>(op)) {
    return PropagateShapeToFunctions(module, call_op.getOperandTypes(),
                                     {call_op.f()}, graph_version,
                                     max_iteration);
  }

  // TODO(ycao): Implement support for Call op, including function reuse.

  return success();
}

LogicalResult InferShapeUntilFixPoint(Region* region, int64_t graph_version,
                                      int64_t max_iteration) {
  MLIRContext* ctx = region->getContext();
  Dialect* tf_dialect = ctx->getRegisteredDialect<TensorFlowDialect>();

  // An operation folder that is used to attempt folding before inference.
  OperationFolder folder(ctx);
  bool changed = true;

  // TODO(aminim): we could have a more efficient traversal by guiding the
  // traversal with a worklist and reconsider only the nodes for which an
  // operand type was inferred. This would need to be careful if working on a
  // region that would not be isolated.
  for (int iteration = 0; iteration < max_iteration && changed; ++iteration) {
    changed = false;
    LLVM_DEBUG(llvm::dbgs()
               << "Shape inference, iteration " << iteration << "\n");
    region->walk([&](Operation* op) {
      if (op->getDialect() != tf_dialect) {
        changed |= InferShapeForNonTFDialectOperation(op, tf_dialect);
        return;
      }

      // Before attempting inference, just try to fold the operation.
      if (succeeded(folder.tryToFold(op))) return;

      // Best-effort shape inference in attached functions. Do not return
      // failure even if it doesn't get to fixed point.
      if (failed(PropagateShapeIntoAttachedFunctions(op, graph_version,
                                                     max_iteration))) {
        op->emitWarning() << "unable to refine shape of attached function "
                             "arguments and bodies";
      }

      changed |= InferShapeForSingleOperation(op, tf_dialect, graph_version);
    });
  }

  if (changed) {
    return region->getParentOp()->emitWarning()
           << "Shape inference did not reach stable state after "
           << max_iteration << " iterations";
  }
  return success();
}

LogicalResult InferShapeForFunction(FuncOp func,
                                    ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                    int64_t graph_version) {
  mlir::FunctionType func_type = func.getType();
  bool needs_refinement = false;
  llvm::SmallVector<mlir::Type, 4> new_arg_types;
  new_arg_types.reserve(func_type.getNumInputs());

  // Update argument types in-place using the provided arg_shapes.
  for (size_t i = 0; i < func_type.getNumInputs(); ++i) {
    ArrayRef<int64_t> shape = arg_shapes[i];
    mlir::Type element_type;
    if (auto input_ty =
            func_type.getInput(i).dyn_cast<mlir::RankedTensorType>()) {
      if (!input_ty || input_ty.getShape().size() != shape.size()) {
        return failure();
      }
      element_type = input_ty.getElementType();
    } else {
      auto unranked_input_ty =
          func_type.getInput(i).dyn_cast<mlir::TensorType>();
      if (!unranked_input_ty) {
        return failure();
      }
      element_type = unranked_input_ty.getElementType();
    }

    auto new_arg_type = mlir::RankedTensorType::get(shape, element_type);
    if (new_arg_type != func_type.getInput(i)) {
      // If the new type is more detailed, trigger shape inference.
      func.getArgument(i).setType(new_arg_type);
      needs_refinement = true;
    }
    new_arg_types.push_back(new_arg_type);
  }

  if (!needs_refinement) {
    return success();
  }

  mlir::LogicalResult result =
      mlir::TF::InferShapeUntilFixPoint(&func.getBody(), graph_version);
  if (failed(result)) {
    return failure();
  }

  auto return_types = InferShapeForFunctionReturnType(func);
  func.setType(mlir::FunctionType::get(new_arg_types,
                                       return_types.hasValue()
                                           ? return_types.getValue()
                                           : func.getType().getResults(),
                                       func.getContext()));

  return success();
}

LogicalResult InferShapeForFunctionType(FuncOp func) {
  if (auto return_types = InferShapeForFunctionReturnType(func)) {
    func.setType(mlir::FunctionType::get(func.getType().getInputs(),
                                         return_types.getValue(),
                                         func.getContext()));
  }

  return success();
}

}  // namespace TF
}  // namespace mlir

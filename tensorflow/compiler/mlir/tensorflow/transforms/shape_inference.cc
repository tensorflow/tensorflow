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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/export_tf_dialect_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#define DEBUG_TYPE "tf-shape-inference"

using ::tensorflow::int64;

namespace mlir {
namespace TF {

bool InferShapeForSingleOperation(Operation* op, Dialect* tf_dialect,
                                  int64_t graph_version) {
  assert(tf_dialect == op->getDialect());

  // If no result for this op needs shape inference, we have a fast-path return.
  if (llvm::all_of(op->getResultTypes(), [](Type type) {
        auto shape_type = type.dyn_cast<ShapedType>();
        return !shape_type || shape_type.hasStaticShape();
      })) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping inference for statically shaped op '"
                            << op->getName() << "'.\n";);
    return false;
  }

  // tf.Cast are only inferred if they have at least one user in the tf dialect.
  // This is necessary to avoid reprocessing the tf.Cast that are inserted at
  // the end of this function.
  if (isa<CastOp>(op) &&
      llvm::all_of(op->getResult(0)->getUsers(), [&](Operation* user) {
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
  const tensorflow::OpRegistrationData* op_reg_data;
  if (!tensorflow::OpRegistry::Global()
           ->LookUp(node_name.data(), &op_reg_data)
           .ok()) {
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

  // Collect an array describing the input shape for every operand.
  std::vector<tensorflow::PartialTensorShape> input_shapes;
  input_shapes.reserve(op->getNumOperands());
  for (Type operand_type : op->getOperandTypes()) {
    auto shaped_type = operand_type.dyn_cast<ShapedType>();
    // Non-shaped type and dynamically ranked type are marked by an empty entry.
    if (!shaped_type || !shaped_type.hasRank()) {
      input_shapes.emplace_back();
      continue;
    }
    // Convert the MLIR shape indices (int64_t) to TensorFlow indices (int64).
    ArrayRef<int64_t> shape = shaped_type.getShape();
    SmallVector<int64, 8> tf_shape(shape.begin(), shape.end());
    input_shapes.push_back(
        tensorflow::PartialTensorShape({tf_shape.data(), tf_shape.size()}));
  }

  // Perform the shape inference using an InferenceContext with the input
  // shapes. This object is abstracting the information that the ShapeInference
  // function operates on.
  tensorflow::shape_inference::InferenceContext c(
      graph_version, *node_def, op_reg_data->op_def, input_shapes,
      /*input_tensors=*/{}, /*input_tensors_as_shapes=*/{},
      /*input_handle_shapes_and_types=*/{});
  auto status = c.Run(op_reg_data->shape_inference_fn);
  if (!status.ok()) {
    LLVM_DEBUG(llvm::dbgs() << "Shape inference error for '" << *op
                            << "': " << status.error_message() << "\n");
    return false;
  }

  assert(c.num_outputs() == op->getNumResults() &&
         "inference context matches the MLIR number of results.");

  // Update the shape for each of the operation result if the InferenceContext
  // has more precise shapes recorded. A builder is used to insert tf.Cast
  // operation when changing the type of a result is the user is not a TF
  // operation, as we can't guarantee that the new type will be OK.
  bool changed = false;
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (int output : llvm::seq<int>(0, c.num_outputs())) {
    // Skip already statically shaped results.
    Value* result = op->getResult(output);
    auto shaped_type = result->getType().dyn_cast<ShapedType>();
    if (!shaped_type || shaped_type.hasStaticShape()) continue;

    tensorflow::shape_inference::ShapeHandle shape_handle = c.output(output);
    LLVM_DEBUG(llvm::dbgs() << "Inferred output " << output << " : "
                            << c.DebugString(shape_handle) << "\n");
    if (!c.RankKnown(shape_handle)) continue;

    // Convert the shape from TensorFlow (int64) to MLIR (int64_t).
    SmallVector<int64_t, 8> shape;
    for (int dim : llvm::seq<int>(0, c.Rank(shape_handle)))
      shape.push_back(c.Value(c.Dim(shape_handle, dim)));
    auto new_type = builder.getTensorType(shape, shaped_type.getElementType());

    // A tf.Cast operation is lazily created on the first uses that isn't a TF
    // operation.
    TF::CastOp cast_op;
    auto get_cast_op = [&]() {
      if (!cast_op)
        cast_op =
            builder.create<TF::CastOp>(op->getLoc(), result->getType(), result,
                                       /*truncate=*/builder.getBoolAttr(false));
      return cast_op;
    };
    for (OpOperand& use : llvm::make_early_inc_range(result->getUses())) {
      if (use.getOwner()->getDialect() != tf_dialect) use.set(get_cast_op());
    }

    // Finally we inferred the shape and replace the type for this result.
    result->setType(new_type);
    changed = true;
  }
  if (changed)
    LLVM_DEBUG(llvm::dbgs()
               << "Modified after shape inference: '" << *op << "'\n");
  return changed;
}

LogicalResult InferShapeUntilFixPoint(Region* region, int64_t graph_version,
                                      int64_t max_iteration) {
  Dialect* tf_dialect = region->getContext()->getRegisteredDialect(
      TensorFlowDialect::getDialectNamespace());
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
      if (op->getDialect() == tf_dialect)
        changed |= InferShapeForSingleOperation(op, tf_dialect, graph_version);
    });
  }
  return success(!changed);
}

LogicalResult InferShapeForFunction(FuncOp op,
                                    ArrayRef<ArrayRef<int64_t>> arg_shapes,
                                    int64_t graph_version) {
  auto main_func = op;
  mlir::FunctionType func_type = main_func.getType();
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
      main_func.getArgument(i)->setType(new_arg_type);
      needs_refinement = true;
    }
    new_arg_types.push_back(new_arg_type);
  }

  if (!needs_refinement) {
    return success();
  }

  mlir::LogicalResult result =
      mlir::TF::InferShapeUntilFixPoint(&main_func.getBody(), graph_version);
  if (failed(result)) {
    return failure();
  }

  // Must only have 1 block so that there is only one return op.
  if (main_func.getBody().getBlocks().size() != 1 ||
      main_func.front().empty()) {
    return failure();
  }

  // Find the return type.
  auto return_op = dyn_cast<mlir::ReturnOp>(*main_func.front().rbegin());
  if (!return_op) {
    return failure();
  }

  // Manually fold tf.Cast that precedes the return instruction and only differ
  // in shape refinement level.
  for (OpOperand& arg_op : return_op.getOperation()->getOpOperands()) {
    if (auto cast_op = dyn_cast<CastOp>(arg_op.get()->getDefiningOp())) {
      if (cast_op.SrcT() != cast_op.DstT()) continue;
      arg_op.set(cast_op.x());
      if (cast_op.y()->use_empty()) cast_op.erase();
    }
  }

  llvm::SmallVector<mlir::Type, 4> return_types(return_op.getOperandTypes());

  // Update function signature with the results of inference.
  main_func.setType(
      mlir::FunctionType::get(new_arg_types, return_types, op.getContext()));

  return success();
}

}  // namespace TF
}  // namespace mlir

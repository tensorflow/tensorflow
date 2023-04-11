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

#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORSPMDEXPANSION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kMainFunctionName[] = "main";

// Updates `function` input signature operand at `argument_index` with
// `new_shape`.
void UpdateFunctionInputShape(const int argument_index,
                              mlir::RankedTensorType new_arg_type,
                              mlir::func::FuncOp function) {
  auto func_type = function.getFunctionType();
  auto input_types = llvm::to_vector<8>(func_type.getInputs());
  input_types[argument_index] = new_arg_type;
  auto new_func_type = mlir::FunctionType::get(
      function.getContext(), input_types, func_type.getResults());
  function.setType(new_func_type);
  function.getBody()
      .getArgument(argument_index)
      .setType(function.getFunctionType().getInput(argument_index));
}

// If `op` is a TF operation, return itself. If it is an DTensorLayout op,
// return it's consumer TF operation.
mlir::Operation* NextTFOp(mlir::Operation* op) {
  while (auto layout = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    if (op->getUsers().empty()) return nullptr;
    op = *(op->getUsers().begin());
  }
  return op;
}

// Updates the shape of resource argument if argument has `tf._layout`
// attribute.
// For example:
// main(%arg0: tensor<!tf_type.resource<tensor<4x4xf32>>
//                  {tf._layout = "mesh:TPU,x=2,y=2 layout:x,not_sharded"})
//
// will be converted to:
//
// main(%arg0: tensor<!tf_type.resource<tensor<2x4xf32>>
//                   {tf._layout = "mesh:TPU,x=2,y=2 layout:x,not_sharded"})
//
// Note that resource argument type is still a resource type. But it's subtype
// has been changed to reflect local shape.
// If resource argument does not have subtype or subtype does not have static
// shapes or if resource argument does not have corresponding layout attribute,
// this function is an no-op.
mlir::LogicalResult UpdateResourceArgumentType(
    const int arg_index, mlir::func::FuncOp function,
    absl::optional<mlir::RankedTensorType> new_subtype = absl::nullopt) {
  auto resource_arg = function.getArgument(arg_index);
  if (new_subtype) {
    auto new_var_type = mlir::RankedTensorType::get(
        {}, mlir::TF::ResourceType::get(
                mlir::ArrayRef<mlir::TensorType>{*new_subtype},
                function.getContext()));
    UpdateFunctionInputShape(arg_index, new_var_type, function);
    function.setArgAttr(arg_index, kAssignedResourceLocalShape,
                        ConvertTypeToTensorShapeAttr(*new_subtype));
    return mlir::success();
  }

  auto resource_type = resource_arg.getType()
                           .cast<mlir::TensorType>()
                           .getElementType()
                           .dyn_cast<mlir::TF::ResourceType>();
  if (!resource_type) return mlir::success();

  auto sub_types = resource_type.getSubtypes();
  if (sub_types.size() != 1) return mlir::success();

  auto resource_arg_sub_type = sub_types.front();
  if (!resource_arg_sub_type.hasStaticShape()) return mlir::success();

  // The local shape that is to be assigned to this resource argument type. We
  // will either pull it from the assigned local shape attribute or compute it
  // based on the layout.
  // TODO(srujun): use the attribute value only to check the computed shape.
  // This is currently blocked by an "empty_layout" set on the resource
  // arguments, meaning it is not possible to compute local layout.
  llvm::SmallVector<int64_t, 4> local_arg_shape;
  auto assigned_resource_local_shape_attr =
      function.getArgAttrOfType<mlir::TF::ShapeAttr>(
          arg_index, kAssignedResourceLocalShape);
  if (assigned_resource_local_shape_attr) {
    local_arg_shape.append(
        assigned_resource_local_shape_attr.getShape().begin(),
        assigned_resource_local_shape_attr.getShape().end());
  } else {
    auto layout_or_status = ExtractLayoutFromOperand(resource_arg);
    if (!layout_or_status.ok())
      return function.emitOpError(layout_or_status.status().error_message());

    const auto& layout = layout_or_status.value();
    if (!layout) return mlir::success();

    std::vector<int64_t> local_arg_shape_vec =
        layout->LocalShapeFromGlobalShape(resource_arg_sub_type.getShape());
    local_arg_shape.append(local_arg_shape_vec.begin(),
                           local_arg_shape_vec.end());
  }

  auto local_variable_subtype = mlir::RankedTensorType::get(
      local_arg_shape, resource_arg_sub_type.getElementType());
  auto new_var_type = mlir::RankedTensorType::get(
      {}, mlir::TF::ResourceType::get(
              mlir::ArrayRef<mlir::TensorType>{local_variable_subtype},
              function.getContext()));

  UpdateFunctionInputShape(arg_index, new_var_type, function);
  function.setArgAttr(
      arg_index, kAssignedResourceLocalShape,
      mlir::TF::ShapeAttr::get(local_variable_subtype.getContext(),
                               mlir::ArrayRef<int64_t>(local_arg_shape)));

  return mlir::success();
}

// Returns whether `value` is used by AssignVariable op, skipping DTensorLayout
// op.
bool GetResourceArgIndexIfUsedInAssignmentOp(
    mlir::Value value, int* resource_argument_index_for_assign_variable) {
  for (auto user : value.getUsers()) {
    if (auto assign_variable_op =
            llvm::dyn_cast_or_null<mlir::TF::AssignVariableOp>(
                NextTFOp(user))) {
      auto resource =
          GetForwardedDTensorLayoutInput(assign_variable_op.getResource());
      if (llvm::isa<mlir::BlockArgument>(resource)) {
        *resource_argument_index_for_assign_variable =
            resource.cast<mlir::BlockArgument>().getArgNumber();
        return true;
      }
    }
  }
  return false;
}

// Updates argument shapes of `function` based on `tf._layout` attribute.
mlir::LogicalResult UpdateFunctionArgsUsingLayout(mlir::func::FuncOp function) {
  for (int argument_index = 0; argument_index < function.getNumArguments();
       ++argument_index) {
    auto arg_layout_attr = function.getArgAttrOfType<mlir::StringAttr>(
        argument_index, kCustomDeviceAttr);
    if (!arg_layout_attr) continue;

    auto arg_layout = Layout::FromString(arg_layout_attr.getValue().str());
    if (!arg_layout.ok())
      return function.emitOpError(llvm::formatv(
          "Invalid layout attribute found during SPMD expansion: {0}",
          arg_layout.status().error_message()));

    // XLA SPMD will handle argument shape updating for us.
    if (arg_layout->mesh().IsSingleDevice() ||
        arg_layout->mesh().use_xla_spmd()) {
      continue;
    }

    mlir::Type arg_type = mlir::getElementTypeOrSelf(
        function.getFunctionType().getInput(argument_index));

    // If argument is a resource type update the subtype shape information
    // to reflect local shape of resources.
    if (arg_type.isa<mlir::TF::ResourceType>()) {
      if (mlir::failed(UpdateResourceArgumentType(argument_index, function)))
        return mlir::failure();
      continue;
    }

    mlir::RankedTensorType ranked_type =
        function.getFunctionType()
            .getInput(argument_index)
            .dyn_cast<mlir::RankedTensorType>();
    if (!ranked_type) continue;

    // If input value is non-resource type, then update the value to reflect
    // local shape.
    llvm::ArrayRef<int64_t> arg_shape = ranked_type.getShape();
    const std::vector<int64_t> arg_local_shape =
        arg_layout->LocalShapeFromGlobalShape(arg_shape);
    mlir::RankedTensorType new_arg_type = mlir::RankedTensorType::get(
        arg_local_shape, ranked_type.getElementType());
    UpdateFunctionInputShape(argument_index, new_arg_type, function);

    // If Resource is an input to the function and a non-resource value was used
    // for AssignVariable op, then ensure that
    // resource shape of updated/assigned resource is consistent with the
    // local shape of assigned value.
    int assigned_resource_argument_index = -1;
    if (GetResourceArgIndexIfUsedInAssignmentOp(
            function.getArgument(argument_index),
            &assigned_resource_argument_index)) {
      (void)UpdateResourceArgumentType(assigned_resource_argument_index,
                                       function, new_arg_type);
    }
  }
  return mlir::success();
}

// Given SPMD expanded `function_operands` to `function`, update the function
// signature to reflect the local shape of `function_operands`.
mlir::LogicalResult UpdateFunctionWithLocalInputShapes(
    mlir::MutableArrayRef<mlir::OpOperand> function_operands,
    mlir::func::FuncOp function) {
  for (auto& operand : function_operands) {
    const int index = operand.getOperandNumber();
    auto arg_type = operand.get().getType().dyn_cast<mlir::RankedTensorType>();
    if (!arg_type) continue;

    auto arg_local_shape = arg_type.getShape();
    auto new_arg_type =
        mlir::RankedTensorType::get(arg_local_shape, arg_type.getElementType());
    UpdateFunctionInputShape(index, new_arg_type, function);
  }
  return mlir::success();
}

// Updates output shapes of enclosing op or function containing `terminator_op`
// to local shapes.
mlir::LogicalResult UpdateReturnValueShapes(mlir::ModuleOp module,
                                            mlir::Operation* terminator_op) {
  auto parent_op = terminator_op->getBlock()->getParentOp();
  if (!parent_op) return mlir::success();

  auto output_types = llvm::to_vector<8>(terminator_op->getOperandTypes());
  if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(parent_op)) {
    // Update function output type to have local shape.
    auto new_func_type = mlir::FunctionType::get(
        function.getContext(), function.getFunctionType().getInputs(),
        output_types);
    function.setType(new_func_type);

    // Update function callsite operations to reflect local output shapes.
    auto function_uses =
        mlir::SymbolTable::getSymbolUses(function, &module.getBodyRegion());
    if (!function_uses) return mlir::success();

    // Update function callsite operations to reflect local output shapes.
    for (auto function_use : *function_uses) {
      auto callsite_op = function_use.getUser();
      if (!callsite_op) continue;

      for (const auto& output_type_and_index : llvm::enumerate(output_types)) {
        int index = output_type_and_index.index();
        const auto& type = output_type_and_index.value();
        callsite_op->getResult(index).setType(type);
      }
    }
  } else {
    for (const auto& output_type_and_index : llvm::enumerate(output_types)) {
      int index = output_type_and_index.index();
      const auto& type = output_type_and_index.value();
      parent_op->getResult(index).setType(type);
    }
  }

  return mlir::success();
}

// Conducts SPMD expansion for all ops in `module`. If function call operation
// exists, walk the function in topological order to update inputs/outputs of
// functions before SPMD expansion of callsite operations is done.
// Note that the iteration won't work with recursive function calls.
mlir::LogicalResult ConductSPMDExpansion(mlir::ModuleOp module) {
  auto main_func = module.lookupSymbol<mlir::func::FuncOp>(kMainFunctionName);
  if (!main_func)
    return module.emitOpError(
        "could not find `main` function in module for SPMD expansion.");

  if (mlir::failed(UpdateFunctionArgsUsingLayout(main_func)))
    return mlir::failure();

  TopologicalIterator iterator(main_func);
  while (iterator.hasNext()) {
    mlir::Operation* op = iterator.next();
    absl::optional<mlir::func::FuncOp> func = MaybeFindFunction(op);
    if (func.has_value()) {
      if (mlir::failed(
              UpdateFunctionWithLocalInputShapes(op->getOpOperands(), *func)))
        return mlir::failure();
    }

    const bool is_terminator_op =
        llvm::isa<mlir::func::ReturnOp, mlir::tf_device::ReturnOp>(op);
    if (auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(op))
      layout_op.getOutput().setType(layout_op.getInput().getType());

    mlir::Operation* expanded_op = nullptr;
    auto status = RunSPMDExpansion(op, &expanded_op);
    if (!status.ok() || expanded_op == nullptr) {
      // Sometimes op may been erased and expanded_op set.
      // In this case we should emit the error on the expanded op.
      mlir::Operation* emit_op = op;
      if (expanded_op != nullptr) emit_op = expanded_op;
      return emit_op->emitError(WithContext(status, __FILE__, __LINE__,
                                            "While computing SPMD expansion")
                                    .error_message());
    }

    // If expanded op is terminator of tf_device.Cluster or a function, then
    // make sure to update the function return value as well as the shape of
    // it's callsite operation.
    if (is_terminator_op)
      if (mlir::failed(UpdateReturnValueShapes(module, expanded_op)))
        return mlir::failure();
  }
  return mlir::success();
}

// Removes temporary attrs created during SPMD expansion.
void RemoveTemporarySPMDAttrs(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation* op) {
    if (op->hasAttr(kDeviceSeedForMeshDims)) {
      op->removeAttr(kDeviceSeedForMeshDims);
    }
  });
}

// MLIR pass that converts graph in global view into a local view which can be
// invoked in parallel on distributed set of devices. This pass removes
// all DTensorLayout ops after the expansion is done. Temporary nodes and
// attributes are also removed after the pass is done.
struct DTensorSPMDExpansion
    : public impl::DTensorSPMDExpansionBase<DTensorSPMDExpansion> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    if (failed(ConductSPMDExpansion(module))) return signalPassFailure();

    RemoveDTensorLayoutOps(module, /*remove_xla_spmd_layouts=*/false);

    RemoveTemporarySPMDAttrs(module);
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSPMDExpansion() {
  return std::make_unique<DTensorSPMDExpansion>();
}

}  // namespace dtensor
}  // namespace tensorflow

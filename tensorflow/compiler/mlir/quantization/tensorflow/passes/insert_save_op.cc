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
#include <cstdint>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace quant {
namespace {

using ::mlir::tf_saved_model::GetInitializerFunction;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;

constexpr StringRef kTfQuantSaveV2OpName = "tf_quant__save_save_v2";
constexpr StringRef kTfQuantSaveReturnOpName = "tf_quant__save_return";

// A pass that creates a new function that wraps the newly created SaveV2 op.
// The new function's name is "tf_quant__save". The function accepts a single
// string tensor as argument, which specifies the path to the checkpoint to
// which the variable's tensor values are saved. It finds
// `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` pattern in the initializer
// function of type "restore_op" to identify the VarHandleOps that should be
// saved using the SaveV2 op.
class InsertSaveOpPass
    : public PassWrapper<InsertSaveOpPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertSaveOpPass)

  explicit InsertSaveOpPass() = default;

  // The argument used to refer to the pass in the textual format (e.g. on the
  // commandline).
  StringRef getArgument() const final { return "quant-insert-save-op"; }

  StringRef getDescription() const final {
    return "Inserts a new function that wraps a SaveV2 op. The SaveV2 op saves "
           "the values of the VarHandleOps that are found in the initializer "
           "function of 'restore_op' type.";
  }

  void runOnOperation() override;
};

// Finds `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` patterns and removes
// `tf.AssignVariableOp`s and `tf.Const`s. Collects and returns the
// `tf.VarHandleOp`s that are initialized by these `tf.AssignVariableOp`s.
SmallVector<TF::VarHandleOp> CollectVariableOps(
    func::FuncOp session_init_func) {
  SmallVector<TF::VarHandleOp> var_handle_ops{};

  for (auto assign_variable_op : llvm::make_early_inc_range(
           session_init_func.getOps<TF::AssignVariableOp>())) {
    Value resource_operand = assign_variable_op.getOperand(0);
    auto var_handle_op =
        dyn_cast<TF::VarHandleOp>(resource_operand.getDefiningOp());
    if (!var_handle_op) continue;

    Value assigned_value_operand = assign_variable_op.getOperand(1);
    auto const_op =
        dyn_cast<TF::ConstOp>(assigned_value_operand.getDefiningOp());
    if (!const_op) continue;

    var_handle_ops.emplace_back(var_handle_op);
  }

  return var_handle_ops;
}

// Creates a `ConstOp` of 1-dimensional TF::StringType out of `str_values`.
TF::ConstOp Create1DStringConst(const ArrayRef<std::string> str_values,
                                const Location loc, OpBuilder& builder) {
  const auto tensor_type =
      RankedTensorType::get(/*shape=*/{static_cast<int64_t>(str_values.size())},
                            /*elementType=*/builder.getType<TF::StringType>());

  return builder.create<TF::ConstOp>(
      loc, DenseStringElementsAttr::get(
               tensor_type,
               SmallVector<StringRef>(str_values.begin(), str_values.end())));
}

// Creates a 1D string array constant for "tensor_names" input of `RestoreV2`
// op. The `ConstOp` will be created at `builder`'s current insertion point.
TF::ConstOp CreateTensorNamesConst(const ArrayRef<std::string> tensor_names,
                                   OpBuilder& builder) {
  const auto loc = NameLoc::get(builder.getStringAttr("tensor_names"));
  return Create1DStringConst(tensor_names, loc, builder);
}

// Creates a 1D string array constant for "shape_and_slices" input of
// `RestoreV2` op. The `ConstOp` will be created at `builder`'s current
// insertion point. It will be filled with `size` empty strings.
TF::ConstOp CreateShapeAndSlicesConst(const int size, OpBuilder& builder) {
  const SmallVector<std::string> shape_and_slices_values(size, /*Value=*/"");

  const auto loc = NameLoc::get(builder.getStringAttr("shape_and_slices"));
  return Create1DStringConst(shape_and_slices_values, loc, builder);
}

// Returns cloned `VarHandleOp`s. Assumes `save_func`'s body is empty.
SmallVector<TF::VarHandleOp> CloneVarHandleOpsIntoSaveFunc(
    func::FuncOp save_func, const ArrayRef<TF::VarHandleOp> var_handle_ops) {
  Block& save_op_block = save_func.getBody().front();

  IRMapping mapper{};
  SmallVector<TF::VarHandleOp> cloned_var_handle_ops = {};
  for (auto var_handle_op : var_handle_ops) {
    Operation* cloned_var_handle_op = var_handle_op->clone(mapper);
    save_op_block.push_back(cloned_var_handle_op);

    cloned_var_handle_ops.push_back(
        cast<TF::VarHandleOp>(cloned_var_handle_op));
  }

  return cloned_var_handle_ops;
}

// Creates and returns a `TF::SaveV2Op` for the `var_handle_ops`. For each
// VarHandleOp in `var_handle_ops` the tensor value is read via
// `TF::ReadVariableOp` and provided as arguments to the newly created SaveV2
// op.
TF::SaveV2Op CreateSaveV2Op(func::FuncOp save_func,
                            const ArrayRef<TF::VarHandleOp> var_handle_ops) {
  auto builder = OpBuilder::atBlockEnd(&save_func.getBody().front());

  SmallVector<std::string> tensor_names = {};
  SmallVector<Value> tensor_values = {};
  for (auto var_handle_op : var_handle_ops) {
    tensor_names.emplace_back(var_handle_op.getSharedName().str());

    auto read_var_op = builder.create<TF::ReadVariableOp>(
        var_handle_op.getLoc(), var_handle_op.resource_subtype(),
        var_handle_op);
    tensor_values.emplace_back(read_var_op.getResult());
  }

  TF::ConstOp tensor_names_const =
      CreateTensorNamesConst(tensor_names, builder);
  TF::ConstOp shape_and_slices_const =
      CreateShapeAndSlicesConst(tensor_names.size(), builder);

  BlockArgument filename_arg = save_func.getArgument(0);
  return builder.create<TF::SaveV2Op>(
      NameLoc::get(builder.getStringAttr(kTfQuantSaveV2OpName)),
      /*prefix=*/filename_arg, tensor_names_const, shape_and_slices_const,
      /*tensors=*/tensor_values);
}

// Creates and returns a new `FuncOp` named "tf_quant__save". The resulting
// `FuncOp`'s body has no ops.
func::FuncOp CreateEmptySaveFunc(ModuleOp module_op) {
  OpBuilder builder(module_op);
  builder.setInsertionPointToEnd(&module_op.getBodyRegion().front());

  auto filename_input_type = RankedTensorType::get(
      /*shape=*/{}, /*elementType=*/builder.getType<TF::StringType>());

  FunctionType func_type = builder.getFunctionType(
      /*inputs=*/{filename_input_type}, /*results=*/{});
  auto save_func = builder.create<func::FuncOp>(
      NameLoc::get(builder.getStringAttr(kTfQuantSaveFuncName)),
      /*sym_name=*/kTfQuantSaveFuncName, func_type);
  save_func.addEntryBlock();
  save_func.setPrivate();

  return save_func;
}

// Creates a save function that contains the `TF::SaveV2Op` for the variables in
// `var_handle_ops`. The `var_handle_ops` are cloned into the new function and
// provides the tensor values to be saved. The new function is a private
// function and has one argument for the file prefix (the directory to the
// checkpoint).
void CreateSaveFunc(ModuleOp module_op,
                    const ArrayRef<TF::VarHandleOp> var_handle_ops) {
  func::FuncOp save_func = CreateEmptySaveFunc(module_op);

  const SmallVector<TF::VarHandleOp> cloned_var_handle_ops =
      CloneVarHandleOpsIntoSaveFunc(save_func, var_handle_ops);

  CreateSaveV2Op(save_func, cloned_var_handle_ops);

  // Create a "func.return".
  auto builder = OpBuilder::atBlockEnd(&save_func.getBody().front());
  builder.create<func::ReturnOp>(
      NameLoc::get(builder.getStringAttr(kTfQuantSaveReturnOpName)));
}

void InsertSaveOpPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  func::FuncOp session_init_func = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerRestoreType);
  if (!session_init_func) {
    LOG(INFO) << "No session initializer function with type 'restore_op'. "
                 "SaveV2 op will not be created.";
    return;
  }

  SmallVector<TF::VarHandleOp> target_var_handle_ops =
      CollectVariableOps(session_init_func);
  if (target_var_handle_ops.empty()) {
    LOG(INFO) << "There are no VarHandleOps to save. SaveV2 op will not "
                 "be created.";
    return;
  }

  CreateSaveFunc(module_op, target_var_handle_ops);
}

static PassRegistration<InsertSaveOpPass> pass{};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateInsertSaveOpPass() {
  return std::make_unique<InsertSaveOpPass>();
}

}  // namespace quant
}  // namespace mlir

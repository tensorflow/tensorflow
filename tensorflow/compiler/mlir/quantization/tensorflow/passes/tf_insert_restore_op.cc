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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/constants.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace tf_quant {
namespace {

using ::mlir::tf_saved_model::GetInitializerFunction;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;
using ::mlir::tf_saved_model::kTfSavedModelInitializerRestoreType;

// This pass creates a RestoreV2 op in the initializer function with
// type "restore_op" that initializes variables from checkpoint. It finds
// tf.AssignVariableOp(tf.VarHandleOp, tf.Const) patterns in the initializer
// function and replaces tf.Consts with the results of RestoreV2.
class InsertRestoreOpPass
    : public PassWrapper<InsertRestoreOpPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertRestoreOpPass)

  explicit InsertRestoreOpPass() = default;

  // The argument used to refer to the pass in the textual format (e.g. on the
  // commandline).
  StringRef getArgument() const final { return "tf-quant-insert-restore-op"; }

  StringRef getDescription() const final {
    return "Creates RestoreV2 op to initialize the variables in the "
           "initializer function (`tf_saved_model.initializer_type == "
           "'restore_op'`). Replaces each occurrence of "
           "`tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` patterns with "
           "`tf.AssignVariableOp(tf.VarHandleOp, restore_op_output#N)`, where "
           "`restore_op_output#N` is the Nth output of the newly created "
           "RestoreV2Op.";
  }

  void runOnOperation() override;
};

// Finds `tf.AssignVariableOp(tf.VarHandleOp, tf.Const)` patterns and returns
// the `tf.VarHandleOp`s that are initialized by these `tf.AssignVariableOp`s.
std::vector<TF::VarHandleOp> CollectVariableOps(
    func::FuncOp session_init_func) {
  std::vector<TF::VarHandleOp> var_handle_ops{};

  for (auto assign_variable_op : llvm::make_early_inc_range(
           session_init_func.getOps<TF::AssignVariableOp>())) {
    Value resource_operand = assign_variable_op.getOperand(0);
    Value assigned_value_operand = assign_variable_op.getOperand(1);

    if (auto var_handle_op =
            dyn_cast<TF::VarHandleOp>(resource_operand.getDefiningOp());
        var_handle_op &&
        isa<TF::ConstOp>(assigned_value_operand.getDefiningOp())) {
      var_handle_ops.emplace_back(var_handle_op);
    }
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

// Creates a new argument for `func_op` that accepts a string tensor containing
// the checkpoint file's prefix.
BlockArgument InsertFilePrefixArgument(func::FuncOp func_op,
                                       OpBuilder& builder) {
  const auto filename_op_type = RankedTensorType::get(
      /*shape=*/{}, /*elementType=*/builder.getType<TF::StringType>());
  const auto file_prefix_attr = builder.getStringAttr(quant::kTfFilePrefix);
  const auto arg_attrs = builder.getDictionaryAttr({builder.getNamedAttr(
      kTfSavedModelIndexPathAttr, builder.getArrayAttr({file_prefix_attr}))});

  const int insert_idx = func_op.getNumArguments();

  (void)func_op.insertArgument(insert_idx, /*argType=*/filename_op_type,
                               arg_attrs, NameLoc::get(file_prefix_attr));

  return func_op.getArgument(insert_idx);
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

// Creates a `tf.RestoreV2Op` that loads the variable values from the checkpoint
// file. The loaded tensors will be used to initialize `tf.VarHandleOp`s via
// `tf.AssignVariableOp`s.
void CreateRestoreV2Op(std::vector<TF::VarHandleOp>& target_var_handle_ops,
                       func::FuncOp session_init_func) {
  SmallVector<Type> tensor_types{};
  SmallVector<std::string> tensor_names{};
  for (auto var_handle_op : target_var_handle_ops) {
    tensor_names.emplace_back(var_handle_op.getSharedName().str());
    // Location must be set to the same name as the shared name. The Location is
    // later tranlated to the op's name when exported to `GraphDef`. This is
    // required to find the correct variable name to restore when it is
    // imported back to MLIR. When importing the graph to MLIR, the name of the
    // op is used to retrieve the tensor values of each variable. See
    // `InitializeVariablesInSessionInitializer` for further details.
    const auto loc = NameLoc::get(StringAttr::get(
        var_handle_op.getContext(), var_handle_op.getSharedName()));
    var_handle_op->setLoc(loc);

    // Ex) If VarHandleOp's type is tensor<!tf_type.resource<tensor<1xf32>>>,
    // then tensor<1xf32> is the subtype.
    tensor_types.emplace_back(var_handle_op.resource_subtype());
  }

  auto builder =
      OpBuilder::atBlockTerminator(&session_init_func.getBody().front());

  const BlockArgument filename_arg =
      InsertFilePrefixArgument(session_init_func, builder);

  TF::ConstOp tensor_names_const =
      CreateTensorNamesConst(tensor_names, builder);
  TF::ConstOp shape_and_slices_const =
      CreateShapeAndSlicesConst(tensor_names.size(), builder);

  auto restore_op = builder.create<TF::RestoreV2Op>(
      session_init_func.getLoc(),
      /*tensors=*/tensor_types,
      /*prefix=*/filename_arg, tensor_names_const, shape_and_slices_const);

  for (auto [idx, restore_result] : llvm::enumerate(restore_op.getResults())) {
    builder.create<TF::AssignVariableOp>(
        restore_op.getLoc(), target_var_handle_ops[idx], restore_result);
  }
}

// TODO(b/261813194): Do not create a new RestoreV2 op when a RestoreV2 op
// already exists.
void InsertRestoreOpPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  func::FuncOp session_init_func = GetInitializerFunction(
      module_op, /*initializer_type=*/kTfSavedModelInitializerRestoreType);
  if (!session_init_func) {
    LOG(INFO) << "No session initializer function with type 'restore_op'. "
                 "RestoreV2 op will not be created.";
    return;
  }

  std::vector<TF::VarHandleOp> target_var_handle_ops =
      CollectVariableOps(session_init_func);
  if (target_var_handle_ops.empty()) {
    LOG(INFO) << "There are no VarHandleOps to restore. RestoreV2 op will not "
                 "be created.";
    return;
  }

  CreateRestoreV2Op(target_var_handle_ops, session_init_func);
}

static PassRegistration<InsertRestoreOpPass> pass{};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateInsertRestoreOpPass() {
  return std::make_unique<InsertRestoreOpPass>();
}

}  // namespace tf_quant
}  // namespace mlir

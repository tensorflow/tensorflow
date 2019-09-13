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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

// Rewrites `tf_device.launch_func` operations assigned to TPU into actual TPU
// jit-compile runtime ops.
//
// For example:
//   %1 = "tf_device.launch_func"(%0) {_tpu_replicate = "cluster", func =
//         @tpu_func}
//   %2 = "tf.SomeOp"(%1)
//
// Would become following ops (unimportant attributes, types are omitted):
//    %1 = "tf.Shape"(%0)
//    %2:2 = "tf.MLIRCompileToTPU"(%1) {module = "<Serialized @tpu_func>"}
//    "tf.TPUCompileSucceededAssert"(%2#0)
//    %3 = "tf.TPUExecute"(%0, %2#1)
//    %4 = "tf.SomeOp"(%3)

namespace {
struct TPURewritePass : public ModulePass<TPURewritePass> {
  void runOnModule() override;
};

// Recursively visits all attributes of `op` to find any Attribute of type
// `SymbolRefAttr`.
llvm::SmallVector<SymbolRefAttr, 8> GetAllSymbolRefAttrs(Operation* op) {
  llvm::SmallVector<SymbolRefAttr, 8> symbol_ref_attrs;

  llvm::SmallVector<Attribute, 8> worklist;
  for (auto named_attr : op->getAttrs()) {
    worklist.push_back(named_attr.second);
  }

  while (!worklist.empty()) {
    Attribute attr = worklist.pop_back_val();

    if (SymbolRefAttr symbol_ref_attr = attr.dyn_cast<SymbolRefAttr>()) {
      // Found a SymbolRefAttr, add it to result list.
      symbol_ref_attrs.push_back(symbol_ref_attr);
    } else if (ArrayAttr array_attr = attr.dyn_cast<ArrayAttr>()) {
      // Found an ArrayAttr, add its nested Attributes to worklist for further
      // inspection.
      worklist.append(array_attr.begin(), array_attr.end());
    } else if (DictionaryAttr dict_attr = attr.dyn_cast<DictionaryAttr>()) {
      // Found a DictionaryAttr, add its nested value Attributes to worklist for
      // further inspection.
      for (NamedAttribute named_attr : dict_attr.getValue()) {
        worklist.push_back(named_attr.second);
      }
    }
  }

  return symbol_ref_attrs;
}

// Creates a new self-contained module that contains `entry_func` and all
// referenced functions in `entry_func`. entry_func is renamed to "main".
// Return value is serialized text formate of newly-created module.
std::string EncapsulateFuncAndSerialize(FuncOp entry_func) {
  ModuleOp module = entry_func.getParentOfType<ModuleOp>();
  llvm::SmallVector<FuncOp, 4> referenced({entry_func});

  // Create a new module to hold func and all referenced functions.
  OwningModuleRef module_for_func =
      ModuleOp::create(mlir::UnknownLoc::get(entry_func.getContext()));
  ModuleManager module_manager(module_for_func.get());

  while (!referenced.empty()) {
    auto func = referenced.pop_back_val();

    // Skip functions that have already been cloned into new module.
    if (module_manager.lookupSymbol<FuncOp>(func.getName())) continue;

    // Find any SymbolRefAttr in func that maps to a FuncOp. We need to clone
    // all found FuncOps to new_module to make sure new_module is
    // self-contained.
    func.walk([&](Operation* op) {
      for (auto symbol_ref_attr : GetAllSymbolRefAttrs(op)) {
        FuncOp referenced_func =
            module.lookupSymbol<FuncOp>(symbol_ref_attr.getValue());

        // Skip Symbols that do not map to a function.
        if (!referenced_func) continue;

        referenced.emplace_back(referenced_func);
      }
    });

    auto clone = func.clone();
    if (clone.getName() == entry_func.getName()) {
      // We can simply change name of TPU program's main function because there
      // should be no other reference to it.
      clone.setName("main");
    }
    module_manager.insert(clone);
  }

  // Serialize module and return.
  std::string txt_module;
  {
    llvm::raw_string_ostream os(txt_module);
    module_for_func.get().print(os);
  }
  return txt_module;
}

// Create a `tf.MLIRCompileToTPU` that contains a MLIR module that is
// functionally equivalent to the function referenced by launch_func.
Operation* BuildCompileOp(tf_device::LaunchFuncOp launch_func,
                          OpBuilder* builder) {
  // TODO(b/139377366): Use tf_tpu.compile build method when it is defined.
  OperationState compile_op_state(launch_func.getLoc(), "tf._TPUCompileMlir");

  // Build a shape op for each input to launch_func.
  // TODO(b/139377366): When shape inference is ready, we can use compile time
  // shape inference to get inputs that have static shapes and only use shape
  // ops for the rest.
  llvm::SmallVector<Value*, 4> compile_op_operands;
  compile_op_operands.reserve(launch_func.getNumOperands());

  for (Value* v : launch_func.getOperands()) {
    auto shape_op = builder->create<TF::ShapeOp>(
        launch_func.getLoc(),
        builder->getTensorType({-1}, builder->getIntegerType(64)), v);
    compile_op_operands.emplace_back(shape_op.getResult());
  }
  compile_op_state.addOperands(compile_op_operands);
  compile_op_state.addAttribute(
      "NumDynamicShapes",
      builder->getI64IntegerAttr(compile_op_operands.size()));

  SymbolRefAttr func_attr = launch_func.getAttrOfType<SymbolRefAttr>("func");
  if (!func_attr) {
    launch_func.emitOpError("does not have `func` attribute");
    return nullptr;
  }
  FuncOp func = launch_func.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      func_attr.getValue());

  std::string txt_module = EncapsulateFuncAndSerialize(func);
  compile_op_state.addAttribute("mlir_module",
                                builder->getStringAttr(txt_module));

  // Result #0 is a string indicating whether compilation is successful or not.
  compile_op_state.addTypes(
      builder->getTensorType({}, builder->getType<TF::StringType>()));

  // Result #1 is key to look up executable binary in compilation cache.
  compile_op_state.addTypes(
      builder->getTensorType({}, builder->getType<TF::StringType>()));

  return builder->createOperation(compile_op_state);
}

// Creates a `tf.TPUExecute` op that executes TPU program generated by
// `compile_op`.
Operation* BuildExecuteOp(Operation* compile_op,
                          tf_device::LaunchFuncOp launch_func,
                          OpBuilder* builder) {
  // TODO(b/139377366): Use tf.TPUExecute build method when it is defined.
  OperationState execute_op_state(launch_func.getLoc(), "tf.TPUExecute");

  // TPUExecute inherits all launch_func inputs.
  llvm::SmallVector<Value*, 4> tensor_inputs(launch_func.getOperands());
  execute_op_state.addOperands(tensor_inputs);

  // TODO(b/139377366): Need to snapshot all resource variable inputs in
  // follow-up CLs.

  // Set Targs of TPUExecute according to launch_func input types.
  llvm::SmallVector<Attribute, 4> tensor_input_types_attrs;
  tensor_input_types_attrs.reserve(tensor_inputs.size());
  for (Value* v : tensor_inputs) {
    tensor_input_types_attrs.emplace_back(builder->getTypeAttr(v->getType()));
  }
  execute_op_state.addAttribute(
      "Targs", builder->getArrayAttr(tensor_input_types_attrs));

  // TPUExecute takes an additional input for compilation cache key.
  execute_op_state.addOperands(compile_op->getResult(1));

  // Set Tresults of TPUExecute according to launch_func results types.
  llvm::SmallVector<Attribute, 4> output_types_attrs;
  output_types_attrs.reserve(launch_func.getNumResults());
  for (Value* v : launch_func.getResults()) {
    output_types_attrs.emplace_back(builder->getTypeAttr(v->getType()));
  }
  execute_op_state.addAttribute("Tresults",
                                builder->getArrayAttr(output_types_attrs));

  // TPUExecute has same output types as launch_func.
  llvm::SmallVector<Type, 4> output_types(launch_func.getResultTypes());
  execute_op_state.addTypes(output_types);

  return builder->createOperation(execute_op_state);
}

// Creates a `tf.TPUCompileSucceededAssert` operation that parses compilation
// status of `compile_op` to check whether compilation is successful.
void BuildTPUCompileSucceededAssertOp(Operation* compile_op,
                                      OpBuilder* builder) {
  OperationState assert_op_state(compile_op->getLoc(),
                                 "tf.TPUCompileSucceededAssert");
  assert_op_state.addOperands(compile_op->getResult(0));
  builder->createOperation(assert_op_state);
}

// Rewrites a `tf_device.launch_func` operation into a set of TPU Runtime
// Operations that jit-compiles and executes function in `tf_device.launch_func`
// on TPU.
void Rewrite(tf_device::LaunchFuncOp launch_func, OpBuilder* builder) {
  // Skip non-tpu device launch_func.
  auto replicate_attr = launch_func.getAttrOfType<StringAttr>("_tpu_replicate");
  if (!replicate_attr) return;

  builder->setInsertionPoint(launch_func);
  Operation* compile_op = BuildCompileOp(launch_func, builder);

  // After rewrite, find if there is a TPUCompilationResultOp in the block with
  // the same _tpu_replicate attribute and replace it with the result of the
  // compile op. This op is used as a placeholder to hook during graph creation
  // the other ops that are intended to consume the compile result.
  Block* block = launch_func.getOperation()->getBlock();
  for (auto compile_result_op : block->getOps<TF::TPUCompilationResultOp>())
    compile_result_op.output()->replaceAllUsesWith(compile_op->getResult(0));

  BuildTPUCompileSucceededAssertOp(compile_op, builder);
  // TODO(ycao): Right now we only support single-core case. The right thing to
  // do is to read from launch_func attributes to determine how many execute
  // ops to build.
  Operation* execute_op = BuildExecuteOp(compile_op, launch_func, builder);
  launch_func.replaceAllUsesWith(execute_op);
  launch_func.erase();
}

void TPURewritePass::runOnModule() {
  OpBuilder builder(&getContext());
  getModule().walk([&](tf_device::LaunchFuncOp op) {
    Rewrite(op, &builder);
  });

  // Eliminate TPUReplicatedInput and TPUReplicatedOutput now that the rewrite
  // is complete.
  getModule().walk([&](Operation* op) {
    auto op_name = op->getName().getStringRef();
    if (op_name != "tf.TPUReplicatedInput" &&
        op_name != "tf.TPUReplicatedOutput")
      return;
    op->getResult(0)->replaceAllUsesWith(op->getOperand(0));
    op->erase();
  });

  // TODO(b/139377366): Remove functions that are no longer needed.
}

}  // namespace

std::unique_ptr<ModulePassBase> CreateTPURewritePass() {
  return std::make_unique<TPURewritePass>();
}

static PassRegistration<TPURewritePass> pass(
    "tf-tpu-rewrite",
    "Rewriting `tf_device.launch_func` on TPUs into TPU runtime ops");

}  // namespace TFTPU
}  // namespace mlir

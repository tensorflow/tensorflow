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
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/stablehlo_type_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/core/ir/types/dialect.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_REPLACESTABLEHLOOPSINMAINFUNCTIONWITHXLACALLMODULEOPSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

constexpr StringRef kQuantizeTargetOpAttr = "tf_quant.composite_function";

class ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass
    : public impl::
          ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPassBase<
              ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass)

  ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass() = default;

  ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass(
      const ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass& other) =
      default;

 private:
  void runOnOperation() override;
};

// Finds the main function from module_op. Returns nullptr if not found.
// The model's signature keys will contain "@serving_default" as default TF
// Model signature, or "@main" if it is in being exported from MLIR module to
// GraphDef.
func::FuncOp GetMainFunc(ModuleOp module_op) {
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    if (func_op.getSymName().equals("main") ||
        func_op.getSymName().equals("serving_default"))
      return func_op;
  }
  return nullptr;
}

// Creates a unique stablehlo function name based on op order.
std::string CreateStablehloFunctionName(const int id) {
  return Twine("_stablehlo_main_").concat(std::to_string(id)).str();
}

// Follows the structure of Live-variable analysis.
class LiveOuts {
 public:
  LiveOuts() = default;

  explicit LiveOuts(OperandRange range)
      : liveouts_(range.begin(), range.end()), prev_liveouts_(liveouts_) {}

  // Delete the current op from liveouts and moves on to the parent ops.
  void update(Operation& op) {
    for (Value result_value : op.getResults()) {
      liveouts_.erase(result_value);
    }
    for (Value operand : op.getOperands()) {
      liveouts_.insert(operand);
    }
  }

  // Snapshot the current live values to previous live values.
  void snapshot_previous_state() { prev_liveouts_ = liveouts_; }

  // Return the current live values.
  DenseSet<Value>& get() { return liveouts_; }

  // Return the previous live values.
  DenseSet<Value>& get_previous() { return prev_liveouts_; }

 private:
  DenseSet<Value> liveouts_;
  DenseSet<Value> prev_liveouts_;
};

// Creates the tf.XlaCallModuleOp from attributes.
void CreateXlaCallModuleOp(ArrayRef<Value> inputs, ArrayRef<Value> outputs,
                           ArrayRef<Type> result_types,
                           ArrayRef<Operation*> reverse_subgraph,
                           func::FuncOp stablehlo_func_op, ModuleOp module_op) {
  MLIRContext* ctx = module_op.getContext();
  OpBuilder builder(ctx);
  Operation* last_subgraph_op = reverse_subgraph.front();
  builder.setInsertionPointAfter(last_subgraph_op);

  // Create attributes used for creating an XlaCallModuleOp.
  SmallVector<Attribute> shape_attrs;
  for (const Type result_type : result_types) {
    shape_attrs.push_back(
        tf_type::ShapeAttr::get(ctx, result_type.cast<ShapedType>()));
  }
  auto empty_array_attr = ArrayAttr::get(ctx, {});
  // TODO - b/303363466: Allow XlaCallModuleOp with versions >5.
  auto xla_call_module_op = builder.create<TF::XlaCallModuleOp>(
      module_op.getLoc(), /*output=*/result_types,
      /*args=*/inputs,
      /*version=*/5, /*module=*/"",
      /*Sout=*/ArrayAttr::get(ctx, shape_attrs),
      /*dim_args_spec=*/empty_array_attr,
      /*platforms=*/empty_array_attr,
      /*function_list=*/empty_array_attr,
      /*has_token_input_output=*/false,
      /*disabled_checks=*/empty_array_attr);
  xla_call_module_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                              SymbolRefAttr::get(stablehlo_func_op));

  for (auto [original_output_value, xla_call_module_op_result_value] :
       llvm::zip_equal(outputs, xla_call_module_op->getResults())) {
    original_output_value.replaceAllUsesExcept(xla_call_module_op_result_value,
                                               /*exceptedUser=*/nullptr);
  }
}

// Replaces the StableHLO ops with a separate XlaCallModuleOp, then wires it
// back into the main graph.
void ReplaceStablehloOpsWithXlaCallModuleOp(
    ArrayRef<Value> inputs, ArrayRef<Value> outputs,
    ArrayRef<Operation*> reverse_subgraph, const int stablehlo_func_id,
    ModuleOp module_op) {
  MLIRContext* ctx = module_op.getContext();
  OpBuilder builder(ctx);

  // Identify arg types & arg locs.
  SmallVector<Type> arg_types;
  SmallVector<Location> arg_locs;
  for (const Value input_value : inputs) {
    arg_types.push_back(input_value.getType());
    arg_locs.push_back(input_value.getLoc());
  }

  // Identify result types.
  SmallVector<Type> result_types;
  for (const Value output_value : outputs) {
    result_types.push_back(output_value.getType());
  }

  // 1) Create FuncOp for the StableHLO ops. They will be separate subgraphs.
  builder.setInsertionPoint(&*module_op.begin());
  auto stablehlo_func_op = builder.create<func::FuncOp>(
      module_op.getLoc(), CreateStablehloFunctionName(stablehlo_func_id),
      FunctionType::get(ctx, arg_types, result_types));
  stablehlo_func_op.setVisibility(SymbolTable::Visibility::Private);
  stablehlo_func_op->setAttr(TF::kFromXlaCallModuleAttrName,
                             builder.getUnitAttr());

  builder.createBlock(&stablehlo_func_op.getBody(), stablehlo_func_op.begin(),
                      arg_types, arg_locs);

  IRMapping mapper;
  for (auto [input, stablehlo_func_arg] :
       llvm::zip_equal(inputs, stablehlo_func_op.getArguments())) {
    mapper.map(input, stablehlo_func_arg);
  }

  for (Operation* subgraph_op : llvm::reverse(reverse_subgraph)) {
    // Create a deep copy of the subgraph ops' operands to the func op.
    stablehlo_func_op.getBody().begin()->push_back(subgraph_op->clone(mapper));
  }

  SmallVector<Value> result_values;
  for (const Value original_output_value : outputs) {
    // Use the mapped values in the newly created function that correspond to
    // outputs in the original function.
    result_values.push_back(mapper.lookup(original_output_value));
  }
  builder.create<func::ReturnOp>(module_op.getLoc(), result_values);

  // 2) Create XlaCallModuleOp (with ops mapped).
  CreateXlaCallModuleOp(inputs, outputs, result_types, reverse_subgraph,
                        stablehlo_func_op, module_op);

  // 3) Erase the replaced ops.
  for (Operation* subgraph_op : reverse_subgraph) {
    subgraph_op->erase();
  }
}

// Replaces the StableHLO ops in the main function block with
// tf.XlaCallModuleOps as separate subgraphs. Wires them back to the main
// function block to be compatible with SavedModel structure.
void ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOps(
    ModuleOp module_op, func::FuncOp main_func) {
  Block& main_func_block = main_func.getBody().front();

  // LiveOuts keeps track of live values at the output of some op. The updates
  // must be made in a reverse, bottom-up manner.
  auto result_values = main_func_block.getTerminator()->getOperands();
  LiveOuts liveouts(result_values);

  // Copy ops to iterate because we will be modifying the block during
  // iteration. The ordering should be reversed because liveness analysis is a
  // bottom-up analysis. The terminator is not included because the return
  // statement is not included in any subgraph (e.g. XlaCallModuleOp) and is
  // untouched.
  SmallVector<Operation*> reverse_main_func_block_ops;
  for (Operation& main_func_block_op :
       llvm::reverse(main_func_block.without_terminator())) {
    reverse_main_func_block_ops.push_back(&main_func_block_op);
  }

  // Create a separate subgraph invoked with XlaCallModuleOp per each
  // set of StableHLO ops in the main func block.
  SmallVector<Operation*> reverse_subgraph;
  DenseSet<Value> operands;
  DenseSet<Value> defined_values;

  int stablehlo_func_id = 0;
  for (Operation* op : reverse_main_func_block_ops) {
    if (!IsStablehloOp(op)) {
      // Create an XlaCallModuleOp if reverse_subgraph isn't empty.
      if (!reverse_subgraph.empty()) {
        DenseSet<Value> outputs = liveouts.get_previous();
        for (Value live_value : liveouts.get()) {
          outputs.erase(live_value);
        }

        ReplaceStablehloOpsWithXlaCallModuleOp(
            SmallVector<Value>(operands.begin(), operands.end()),
            SmallVector<Value>(outputs.begin(), outputs.end()),
            reverse_subgraph, stablehlo_func_id++, module_op);

        // Reset states and start a new subgraph.
        reverse_subgraph.clear();
        operands.clear();
        defined_values.clear();
      }
    }

    // Move on to the parent ops.
    liveouts.update(*op);

    if (!IsStablehloOp(op)) {
      // Always update the liveouts when the subgraph isn't being continued.
      liveouts.snapshot_previous_state();
      continue;
    }

    reverse_subgraph.push_back(op);
  }

  // Create the last subgraph if it isn't empty.
  if (!reverse_subgraph.empty()) {
    DenseSet<Value> outputs = liveouts.get_previous();
    for (Value live_value : liveouts.get()) {
      outputs.erase(live_value);
    }
    // Additionally remove arguments from the outputs, as it provides liveness
    // throughout (functions as an invisible op above the very first op that
    // returns the arguments).
    for (const BlockArgument arg : main_func.getArguments()) {
      outputs.erase(arg);
    }

    ReplaceStablehloOpsWithXlaCallModuleOp(
        SmallVector<Value>(operands.begin(), operands.end()),
        SmallVector<Value>(outputs.begin(), outputs.end()), reverse_subgraph,
        stablehlo_func_id++, module_op);
  }
}

void ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass::
    runOnOperation() {
  ModuleOp module_op = getOperation();

  func::FuncOp main_func = GetMainFunc(module_op);
  if (!main_func) return;

  ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOps(module_op, main_func);

  // TODO - b/298966126: Currently quantizable functions are identified in TF
  // Quantizer via the tf_quant.composite_function UnitAttr attached to func
  // ops. We remove this attribute as this interferes with VHLO conversion.
  // Remove this temporary hack.
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    func_op->removeAttr(kQuantizeTargetOpAttr);
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo

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
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/quantization/common/func.h"
#include "tensorflow/compiler/mlir/quantization/common/lift_as_function_call.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/stablehlo_type_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_call_module_attrs.h"
#include "tensorflow/core/ir/types/dialect.h"

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_REPLACESTABLEHLOOPSINMAINFUNCTIONWITHXLACALLMODULEOPSPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

constexpr StringRef kStablehloModuleAttrsAttrName = "_stablehlo_module_attrs";
constexpr StringRef kUsesShapePolymorphismAttr = "jax.uses_shape_polymorphism";

// Default version number for native serialization.
constexpr int64_t kDefaultVersion = 9;
// Platforms for XlaCallModuleOp.
constexpr StringRef kPlatformCpu = "CPU";
constexpr StringRef kPlatformTpu = "TPU";

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

// Creates a unique stablehlo function name based on op order.
std::string CreateStablehloFunctionName(const int id) {
  return Twine("_stablehlo_main_").concat(std::to_string(id)).str();
}

// Follows the structure of Live-variable analysis. It is a form of
// CFG (Control Flow Graph) analysis, often used in compilers.
//
// A variable is live if it holds a value that may be used in the future.
// It is live-in at node n if it is live on any of the node's in-edges.
// It is live-out at node n if it is live on any of the node's out-edges.
// def[n] refers to values that are defined at node n.
// use[n] refers to values that are used at node n.
//
// Given a node n, variables' liveliness is defined like the following:
// live_in[n] = use[n] U (live_out[n] - def[n])
// live_out[n] = U {live_in[s] | s ε succ[n]}
//
// Consider a sequence of op:
//
// ```
// node 1: %0 = stablehlo.constant
// node 2: %1 = stablehlo.constant
// node 3: %2 = stablehlo.add %0, %1
// node 4: %3 = stablehlo.multiply %2, %1
// node 5: return %3
// ```
//
// In Backward Liveliness analysis, the liveliness for each node above becomes:
// live_in[5] = use[5]   U (live_out[5] - def[5])
//            = {%3}     U ({∅} - {∅})            = {%3}
// live_in[4] = use[4]   U (live_out[4] - def[4])
//            = {%1, %2} U ({%3} - {%3})          = {%1, %2}
// live_in[3] = use[3]   U (live_out[3] - def[3])
//            = {%0, %1} U ({%1, %2} - {%2})      = {%0, %1}
// live_in[2] = use[2]   U (live_out[2] - def[2])
//            = {∅}      U ({%0, %1} - {%1})      = {%0}
// live_in[1] = use[1]   U (live_out[1] - def[1])
//            = {∅}      U ({%0} - {%0})          = {∅}
//
// This analogy is used throughout this pass to ensure only live edges form
// proper subgraphs.
class LiveOuts {
 public:
  LiveOuts() = default;

  explicit LiveOuts(OperandRange range)
      : liveouts_(range.begin(), range.end()), prev_liveouts_(liveouts_) {}

  // Delete the current op from liveouts and moves on to the parent ops.
  void update(Operation& op) {
    for (Value result_value : op.getResults()) {
      liveouts_.remove(result_value);
    }
    for (Value operand : op.getOperands()) {
      liveouts_.insert(operand);
    }
  }

  // Snapshot the current live values to previous live values.
  void snapshot_previous_state() { prev_liveouts_ = liveouts_; }

  // Return the current live values.
  const SetVector<Value>& get() const { return liveouts_; }

  // Return the previous live values.
  const SetVector<Value>& get_previous() const { return prev_liveouts_; }

 private:
  // Use SerVector to ensure deterministic traversal order.
  SetVector<Value> liveouts_;
  SetVector<Value> prev_liveouts_;
};

// Creates the tf.XlaCallModuleOp from attributes.
void CreateXlaCallModuleOp(ValueRange inputs, ValueRange outputs,
                           const TypeRange result_types,
                           const SetVector<Operation*>& reverse_subgraph,
                           const func::FuncOp stablehlo_func_op,
                           ModuleOp module_op) {
  MLIRContext* ctx = module_op.getContext();
  OpBuilder builder(ctx);
  Operation* last_subgraph_op = reverse_subgraph.front();
  builder.setInsertionPointAfter(last_subgraph_op);

  // Create attributes used for creating an XlaCallModuleOp.
  SmallVector<Attribute> shape_attrs;
  for (const Type result_type : result_types) {
    shape_attrs.push_back(
        tf_type::ShapeAttr::get(ctx, mlir::cast<ShapedType>(result_type)));
  }
  const auto empty_array_attr = ArrayAttr::get(ctx, {});
  // TODO: b/310291615 - find a better way for platform support.
  const auto platforms = ArrayAttr::get(
      ctx,
      {StringAttr::get(ctx, kPlatformCpu), StringAttr::get(ctx, kPlatformTpu)});

  auto xla_call_module_op = builder.create<TF::XlaCallModuleOp>(
      module_op.getLoc(), /*output=*/result_types,
      /*args=*/inputs,
      /*version=*/kDefaultVersion, /*module=*/"",
      /*Sout=*/ArrayAttr::get(ctx, shape_attrs),
      /*dim_args_spec=*/empty_array_attr, platforms,
      /*function_list=*/empty_array_attr,
      /*has_token_input_output=*/false,
      /*disabled_checks=*/empty_array_attr);
  xla_call_module_op->setAttr(TF::kStablehloEntryFunctionAttrName,
                              SymbolRefAttr::get(stablehlo_func_op));
  // Set jax.uses_shape_polymorphism=true to enable shape refinement at runtime.
  // This is needed for native serialization version >= 8.
  xla_call_module_op->setAttr(
      kStablehloModuleAttrsAttrName,
      builder.getDictionaryAttr(builder.getNamedAttr(
          kUsesShapePolymorphismAttr, builder.getBoolAttr(true))));

  for (auto [original_output_value, xla_call_module_op_result_value] :
       llvm::zip_equal(outputs, xla_call_module_op->getResults())) {
    original_output_value.replaceAllUsesExcept(xla_call_module_op_result_value,
                                               /*exceptedUser=*/nullptr);
  }
}

// Replaces the StableHLO ops with a separate XlaCallModuleOp, then wires it
// back into the main graph.
void ReplaceStablehloOpsWithXlaCallModuleOp(
    const ArrayRef<Value> inputs, const ArrayRef<Value> outputs,
    const SetVector<Operation*>& reverse_subgraph, const int stablehlo_func_id,
    ModuleOp module_op) {
  MLIRContext* ctx = module_op.getContext();
  OpBuilder builder(ctx);

  // Identify arg types & arg locs.
  SmallVector<Type> arg_types;
  SmallVector<Location> arg_locs;

  // Add an argument for platform_index. This allows for multiple platforms.
  // TODO: b/310291615 - find a better way for platform support.
  arg_types.push_back(RankedTensorType::get({}, builder.getI32Type()));
  arg_locs.push_back(module_op.getLoc());
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
  // stablehlo_func_op has 1 extra arg for platform index.
  for (auto [input, stablehlo_func_arg] : llvm::zip_equal(
           inputs, stablehlo_func_op.getArguments().take_back(inputs.size()))) {
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

// Contains the actual logic for updating states and replacing StableHLO ops
// with tf.XlaCallModuleOps.
void UpdateStatesAndReplaceStablehloOps(
    const SetVector<Value>& operands, const SetVector<Value>& defined_values,
    const LiveOuts& liveouts, ModuleOp module_op,
    const SetVector<Operation*>& reverse_subgraph, const int stablehlo_func_id,
    func::FuncOp main_func, const bool is_last_subgraph = false) {
  SetVector<Value> inputs = operands;
  for (Value defined_value : defined_values) {
    inputs.remove(defined_value);
  }

  SetVector<Value> outputs = liveouts.get_previous();
  for (const Value live_value : liveouts.get()) {
    outputs.remove(live_value);
  }

  if (is_last_subgraph) {
    // Additionally remove arguments from the outputs, as it provides liveness
    // throughout (functions as an invisible op above the very first op that
    // returns the arguments).
    for (const BlockArgument arg : main_func.getArguments()) {
      outputs.remove(arg);
    }
  }

  ReplaceStablehloOpsWithXlaCallModuleOp(
      SmallVector<Value>(inputs.begin(), inputs.end()),
      SmallVector<Value>(outputs.begin(), outputs.end()), reverse_subgraph,
      stablehlo_func_id, module_op);
}

// Check if the op should be added to the subgraph.
// The op should be added to the subgraph if all of its users match one
// of following two conditions:
// 1: The user is already in the current subgraph.
// 2: The user will reach a dead end.
//
// If the op should be added to the subgraph and there are users who
// will reach the dead end, add the ops on the dead end to the subgraph as well.
bool ShouldAddOpToSubgraph(Operation* op,
                           const SetVector<Operation*>& reverse_subgraph,
                           const SetVector<Operation*>& ops_to_add,
                           SmallVector<Operation*>& all_descendants) {
  if (!op) {
    return false;
  }

  SmallVector<Operation*> current_layer_descendants;
  SmallVector<Operation*> next_layer_descendants;
  int current_depth = 0;
  current_layer_descendants.push_back(op);
  // BFS downstream ops for current user.
  // If any one of the descendants meet one of the three conditions, we return
  // false for the current value:
  // 1: The descendant is not in the ops_to_add.
  // 2: The descendant is not a stablehlo op.
  // 3: The depth of the descendant is larger than 5, we don't want to search
  // too deep, max depth is arbitrarily chosen.
  while (!current_layer_descendants.empty()) {
    if (current_depth > 5) {
      all_descendants.clear();
      return false;
    }
    current_depth++;

    for (Operation* descendant : current_layer_descendants) {
      if (!IsStablehloOp(descendant) || !ops_to_add.contains(descendant)) {
        all_descendants.clear();
        return false;
      }
      for (Operation* next_descendant : descendant->getUsers()) {
        if (reverse_subgraph.contains(next_descendant)) {
          continue;
        }
        next_layer_descendants.push_back(next_descendant);
      }
      all_descendants.push_back(descendant);
    }

    current_layer_descendants = next_layer_descendants;
    next_layer_descendants.clear();
  }

  return true;
}

// Replaces the StableHLO ops in the main function block with
// tf.XlaCallModuleOps as separate subgraphs. Wires them back to the main
// function block to be compatible with SavedModel structure.
void ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOps(
    ModuleOp module_op, func::FuncOp main_func, int& stablehlo_func_id) {
  Block& main_func_block = main_func.getBody().front();

  // LiveOuts keeps track of live values at the output of some op. The updates
  // must be made in a reverse, bottom-up manner.
  const auto result_values = main_func_block.getTerminator()->getOperands();
  LiveOuts liveouts(result_values);

  // Copy ops to iterate because we will be modifying the block during
  // iteration. The ordering should be reversed because liveness analysis is a
  // bottom-up analysis. The terminator is not included because the return
  // statement is not included in any subgraph (e.g. XlaCallModuleOp) and is
  // untouched.
  SmallVector<Operation*> reverse_main_func_block_ops;
  SetVector<Operation*> ops_to_add;
  for (Operation& main_func_block_op :
       llvm::reverse(main_func_block.without_terminator())) {
    reverse_main_func_block_ops.push_back(&main_func_block_op);
    ops_to_add.insert(&main_func_block_op);
  }

  // Create a separate subgraph invoked with XlaCallModuleOp per each
  // set of StableHLO ops in the main func block.
  SetVector<Operation*> reverse_subgraph;
  SetVector<Value> operands;
  SetVector<Value> defined_values;

  // Add op to the subgraph.
  const auto add_to_subgraph = [&](Operation* op) {
    // Move on to the parent ops.
    liveouts.update(*op);
    ops_to_add.remove(op);

    if (!IsStablehloOp(op)) {
      // Always update the liveouts when the subgraph isn't being continued.
      liveouts.snapshot_previous_state();
      return;
    }

    reverse_subgraph.insert(op);
    defined_values.insert(op->getResults().begin(), op->getResults().end());
    operands.insert(op->getOperands().begin(), op->getOperands().end());
  };

  for (Operation* op : reverse_main_func_block_ops) {
    if (!ops_to_add.contains(op)) continue;
    // When hitting a non-StableHLO op, i.e. tf.CustomAggregatorOp, start
    // recursively tracing defining ops of the current subgraph's operands. This
    // makes sure that all dependencies needed for shape inference are included
    // in the subgraph. We only trace StableHLO ops that have all users inside
    // the current subgraph.
    // TODO: b/311239049 - Consider rewrite this using BFS.
    if (!IsStablehloOp(op)) {
      bool should_add_op = true;
      while (should_add_op) {
        should_add_op = false;
        SmallVector<Operation*> all_descendants;
        for (Value v : operands) {
          if (defined_values.contains(v)) continue;
          if (ShouldAddOpToSubgraph(v.getDefiningOp(), reverse_subgraph,
                                    ops_to_add, all_descendants)) {
            should_add_op = true;
            break;
          }
        }
        if (should_add_op) {
          for (auto descendant : llvm::reverse(all_descendants)) {
            add_to_subgraph(descendant);
          }
        }
      }
      // Create an XlaCallModuleOp if reverse_subgraph isn't empty.
      if (!reverse_subgraph.empty()) {
        UpdateStatesAndReplaceStablehloOps(operands, defined_values, liveouts,
                                           module_op, reverse_subgraph,
                                           ++stablehlo_func_id, main_func);

        // Reset states and start a new subgraph.
        reverse_subgraph.clear();
        operands.clear();
        defined_values.clear();
      }
    }
    add_to_subgraph(op);
  }

  // Create the last subgraph if it isn't empty.
  if (!reverse_subgraph.empty()) {
    UpdateStatesAndReplaceStablehloOps(
        operands, defined_values, liveouts, module_op, reverse_subgraph,
        ++stablehlo_func_id, main_func, /*is_last_subgraph=*/true);
  }
}

// Duplicates small constants for each use.
//
// In the subsequent graph partitioning, constants for shape inference need to
// be in the same subgraph. But graph partitioning stops at ops with multiple
// uses. So here we duplicate small constants for each use so that if a
// constant is useful for shape inference for multiple subgraphs, they can be
// included in each subgraphs. If duplicate constants are accidentally created
// in the same subgraph, they can be easily removed with a canonicalizer pass.
//
// We set a size limit since constants needed for shape inference are no
// larger than tensor rank. This avoids duplicating large constants.
void DuplicateSmallConstantOps(ModuleOp module_op, func::FuncOp main_func) {
  OpBuilder builder(main_func.getContext());
  for (auto constant_op :
       main_func.getBody().getOps<mlir::stablehlo::ConstantOp>()) {
    builder.setInsertionPointAfter(constant_op);
    if (constant_op.getResult().use_empty() ||
        constant_op.getResult().hasOneUse())
      continue;
    // Do not duplicate constant op if the size is too large.
    // 32 is chosen to be larger than all constants useful for shape references,
    // while not too large to possibly significantly increase model size.
    if (constant_op.getValue().getNumElements() > 32) continue;
    while (!constant_op.getResult().hasOneUse()) {
      auto new_constant_op = builder.clone(*constant_op.getOperation());
      constant_op.getResult().getUses().begin()->assign(
          dyn_cast<mlir::stablehlo::ConstantOp>(new_constant_op));
    }
  }
}

void ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass::
    runOnOperation() {
  ModuleOp module_op = getOperation();

  func::FuncOp main_func = FindMainFuncOp(module_op);
  if (!main_func) return;

  // In case the model has tf.StatefulPartitionedCallOp or tf.PartitionedCallOp,
  // we recursively find called functions and process StableHLO ops in them.
  SmallVector<func::FuncOp> func_ops;
  func_ops.push_back(main_func);
  int stablehlo_func_id = -1;
  while (!func_ops.empty()) {
    auto main_func = func_ops.back();
    func_ops.pop_back();
    if (!main_func) continue;

    SymbolTable symbol_table(module_op);
    for (auto call_op : main_func.getOps<TF::PartitionedCallOp>()) {
      func_ops.push_back(dyn_cast_or_null<func::FuncOp>(symbol_table.lookup(
          mlir::cast<FlatSymbolRefAttr>(call_op.getFAttr()).getValue())));
    }
    for (auto call_op : main_func.getOps<TF::StatefulPartitionedCallOp>()) {
      func_ops.push_back(
          dyn_cast_or_null<func::FuncOp>(symbol_table.lookup(call_op.getF())));
    }

    DuplicateSmallConstantOps(module_op, main_func);
    ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOps(module_op, main_func,
                                                          stablehlo_func_id);
  }

  // TODO - b/298966126: Currently quantizable functions are identified in TF
  // Quantizer via the tf_quant.composite_function UnitAttr attached to
  // func ops. We remove this attribute as this interferes with VHLO conversion.
  // Remove this temporary hack.
  for (auto func_op : module_op.getOps<func::FuncOp>()) {
    func_op->removeAttr(kFusedFunctionAttr);
  }
}

}  // namespace

}  // namespace mlir::quant::stablehlo

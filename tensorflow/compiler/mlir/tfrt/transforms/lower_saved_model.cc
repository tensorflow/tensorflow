/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <deque>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"

namespace tensorflow {
namespace {

constexpr char kCpuDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

bool IsSessionInitializer(mlir::FuncOp op) {
  auto session_initializer_op = mlir::tf_saved_model::GetSessionInitializerOp(
      op->getParentOfType<mlir::ModuleOp>());
  if (!session_initializer_op) return false;

  for (auto sym_ref : session_initializer_op.initializers()) {
    if (op.sym_name() == sym_ref.cast<mlir::FlatSymbolRefAttr>().getValue())
      return true;
  }

  return false;
}

mlir::TF::ResourceHandle GetResourceHandle(mlir::Operation *op) {
  llvm::StringRef device;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("device")) {
    device = attr.getValue();
  }

  llvm::StringRef container;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("container")) {
    container = attr.getValue();
  }

  llvm::StringRef shared_name;
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("shared_name")) {
    shared_name = attr.getValue();
  }

  return {container, shared_name, device, /*op=*/nullptr};
}

struct HoistInfo {
  // All hoisted ops in the topological order.
  llvm::SmallVector<mlir::Operation *, 4> hoists_in_topological_order;

  // Mapping from the old values produced by hoisted ops before hoisting to the
  // new values after hoisting.
  mlir::BlockAndValueMapping value_mapping;

  // `hoisted_values` is to keep all values that are produced by hoisted ops
  // but used by non-hoisted ops. These values will be replaced by results of
  // tf._TfrtGetResource op. The index of each value in this array will be the
  // index used in tf._TfrtGetResource and tf._TfrtSetResource op. This also
  // stores the ResourceHandle which has the shared_name and container
  // attributes used by later resource alias analysis and side effect analysis
  // passes.
  llvm::SmallVector<std::pair<mlir::Value, mlir::TF::ResourceHandle>, 4>
      hoisted_values;
};

void ReplaceHoistedValues(
    llvm::ArrayRef<std::pair<mlir::Value, mlir::TF::ResourceHandle>>
        hoisted_values,
    mlir::OpBuilder &builder) {
  struct HoistedValueInfo {
    llvm::SmallVector<mlir::Value, 4> hoisted_values;
    llvm::SmallVector<int64_t, 4> indices;
    llvm::SmallVector<llvm::StringRef, 4> shared_names;
    llvm::SmallVector<llvm::StringRef, 4> containers;
  };
  // Rearrange the hoisted values by each function and each device.
  llvm::DenseMap<mlir::Block *, llvm::StringMap<HoistedValueInfo>>
      hoisted_values_by_block_device;

  // Find a block where to place tf._TfrtGetResource operation. We do not place
  // get resource operations inside the `tf_device.cluster` operations, because
  // these blocks are intended for later on-device compilation. Insert resource
  // reads to the closest block outside of the `tf_device.cluster` operation.
  auto hoist_into_block = [](mlir::Value value) -> mlir::Block * {
    mlir::Operation *cluster_op =
        value.getDefiningOp()->getParentOfType<mlir::tf_device::ClusterOp>();
    return cluster_op ? cluster_op->getBlock() : value.getParentBlock();
  };

  for (auto iter : llvm::enumerate(hoisted_values)) {
    auto value = iter.value().first;
    auto index = iter.index();
    auto &device_map = hoisted_values_by_block_device[hoist_into_block(value)];

    assert(value.getDefiningOp() && "hoisted values must not be arguments.");
    llvm::StringRef device = kCpuDeviceName;
    if (auto device_attr =
            value.getDefiningOp()->getAttrOfType<mlir::StringAttr>("device")) {
      if (!device_attr.getValue().empty()) device = device_attr.getValue();
    }

    auto &item = device_map[device];

    item.hoisted_values.push_back(value);
    item.indices.push_back(index);
    item.shared_names.push_back(iter.value().second.name);
    item.containers.push_back(iter.value().second.container);
  }

  // Create tf._TfrtGetResource op for each function and device.
  for (const auto &block_iter : hoisted_values_by_block_device) {
    auto *block = block_iter.first;
    const auto &device_map = block_iter.second;

    builder.setInsertionPointToStart(block);
    for (const auto &device_iter : device_map) {
      llvm::StringRef device = device_iter.getKey();
      mlir::ValueRange old_values = device_iter.getValue().hoisted_values;
      const auto &indices = device_iter.getValue().indices;
      const auto &shared_name_arr = device_iter.getValue().shared_names;
      const auto &container_arr = device_iter.getValue().containers;

      auto get_resource_op = builder.create<mlir::TF::_TfrtGetResourceOp>(
          block->getParentOp()->getLoc(), old_values.getTypes(),
          builder.getI64ArrayAttr(indices),
          builder.getStrArrayAttr(shared_name_arr),
          builder.getStrArrayAttr(container_arr));
      get_resource_op->setAttr("device", builder.getStringAttr(device));

      auto new_values = get_resource_op.results();
      for (auto iter : llvm::zip(old_values, new_values)) {
        auto old_value = std::get<0>(iter);
        auto new_value = std::get<1>(iter);
        old_value.replaceAllUsesWith(new_value);
      }
    }
  }
}

bool OnlyHasReadEffect(mlir::Operation *op) {
  auto interface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!interface) return false;
  return interface.onlyHasEffect<mlir::MemoryEffects::Read>();
}

bool CanHoist(const llvm::DenseSet<mlir::TF::ResourceHandle> &read_only_vars,
              mlir::Operation *op) {
  // return ops should not be hoisted.
  if (op->mightHaveTrait<mlir::OpTrait::IsTerminator>()) return false;

  // Non-side-effecting ops can be hoisted.
  if (mlir::MemoryEffectOpInterface::hasNoEffect(op)) return true;

  // ResourceHandle ops can be hoisted.
  if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::HashTableV2Op>(op))
    return true;

  // If it is ReadVariableOp and the variable is readonly, it can be hoisted.
  if (auto read_var_op = llvm::dyn_cast<mlir::TF::ReadVariableOp>(op)) {
    if (auto var_handle_op = llvm::dyn_cast_or_null<mlir::TF::VarHandleOp>(
            read_var_op.resource().getDefiningOp())) {
      if (read_only_vars.count(GetResourceHandle(var_handle_op)) > 0)
        return true;
    }
  }

  // If it is LookupTableSizeOp, it can be hoisted as the size of the hash table
  // cannot be changed after initialization.
  if (auto lookup_table_size_op =
          llvm::dyn_cast<mlir::TF::LookupTableSizeV2Op>(op)) {
    if (auto hash_table_op = llvm::dyn_cast_or_null<mlir::TF::HashTableV2Op>(
            lookup_table_size_op.table_handle().getDefiningOp())) {
      if (read_only_vars.count(GetResourceHandle(hash_table_op)) > 0)
        return true;
    }
  }

  // TODO(chky): Allow more readonly ops.

  return false;
}

void HoistInvariantOpsInFunction(
    mlir::FuncOp func,
    const llvm::DenseSet<mlir::TF::ResourceHandle> &read_only_vars,
    const mlir::TF::SideEffectAnalysis::Info &side_effect_analysis,
    mlir::OpBuilder &builder, HoistInfo &module_hoist_info) {
  // Keep the hoisted ops in this function.
  llvm::DenseSet<mlir::Operation *> hoists;

  auto all_operands_in_hoists = [&module_hoist_info](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (module_hoist_info.value_mapping.lookupOrNull(operand) == nullptr)
        return false;
    }
    return true;
  };

  auto all_control_predeccessors_in_hoists = [&hoists, &side_effect_analysis](
                                                 mlir::Operation *op) {
    auto preds = side_effect_analysis.DirectControlPredecessors(op);
    return std::all_of(
        preds.begin(), preds.end(),
        [&hoists](mlir::Operation *pred) { return hoists.count(pred) > 0; });
  };

  std::deque<mlir::Operation *> work_list;

  // Start with ops with tf.VarHandleOp ops and tf.Const ops.
  //
  // TODO(chky): Consider allowing other ops including custom ops to be hoisted.
  func.walk([&work_list](mlir::Operation *op) {
    if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::HashTableV2Op,
                  mlir::TF::ConstOp>(op))
      work_list.push_back(op);
  });

  while (!work_list.empty()) {
    auto *op = work_list.front();
    work_list.pop_front();

    // Skip if it is already hoisted.
    if (hoists.count(op) > 0) continue;

    // If the op can be hoisted, and all of its data dependencies and control
    // dependencies are hoisted, then we hoist it. Otherwise, skip.
    if (!(CanHoist(read_only_vars, op) && all_operands_in_hoists(op) &&
          all_control_predeccessors_in_hoists(op)))
      continue;

    // Record the hoisted operation.
    hoists.insert(op);
    module_hoist_info.hoists_in_topological_order.push_back(op);

    // Create a copy in the init function.
    builder.clone(*op, module_hoist_info.value_mapping);

    for (mlir::Operation *user : op->getUsers()) {
      work_list.push_back(user);
    }
  }

  // Find out the values that are produced by hoisted ops but used by
  // non-hoisted ops. These values need to be replaced.
  for (auto *op : hoists) {
    for (auto result : op->getResults()) {
      if (std::any_of(result.getUsers().begin(), result.getUsers().end(),
                      [&hoists](mlir::Operation *user) {
                        return hoists.count(user) == 0;
                      })) {
        module_hoist_info.hoisted_values.push_back(
            {result, GetResourceHandle(op)});
      }
    }
  }
}

void FindCalleesRecursive(const mlir::SymbolTable &symbol_table,
                          mlir::FuncOp func, llvm::StringSet<> &callees) {
  assert(func);
  func.walk([&](mlir::Operation *op) {
    for (const auto &named_attr : op->getAttrs()) {
      if (auto symbol_attr =
              named_attr.getValue().dyn_cast<mlir::FlatSymbolRefAttr>()) {
        auto symbol = symbol_attr.getValue();
        if (!callees.contains(symbol)) {
          callees.insert(symbol);

          auto func = symbol_table.lookup<mlir::FuncOp>(symbol);
          if (!func) continue;

          FindCalleesRecursive(symbol_table, func, callees);
        }
      }
    }
  });
}

void HoistInvariantOps(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);

  // Find all resources used in non-init functions.
  llvm::DenseMap<mlir::TF::ResourceHandle,
                 llvm::SmallVector<mlir::Operation *, 4>>
      resources;

  // Find all callees referenced in the initialization functions.
  llvm::StringSet<> init_callees;

  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::HashTableV2Op>(op)) {
      auto func = op->getParentOfType<mlir::FuncOp>();
      if (IsSessionInitializer(func)) return;
      resources[GetResourceHandle(op)].push_back(op);
    } else if (auto func = llvm::dyn_cast<mlir::FuncOp>(op)) {
      if (!IsSessionInitializer(func)) return;
      FindCalleesRecursive(symbol_table, func, init_callees);
    }
  });

  llvm::DenseSet<mlir::TF::ResourceHandle> read_only_vars;
  for (const auto &iter : resources) {
    const auto &key = iter.first;
    const auto &vars = iter.second;
    if (std::all_of(vars.begin(), vars.end(), [](mlir::Operation *op) {
          for (auto *user : op->getUsers()) {
            if (!OnlyHasReadEffect(user)) return false;
          }
          return true;
        })) {
      read_only_vars.insert(key);
    }
  }

  mlir::TF::SideEffectAnalysis side_effect_analysis(module);

  mlir::OpBuilder builder(&module.body());
  // "_tfrt_resource_init" is the special function that executes all invariant
  // ops (eg. read-only variables) used in the model. This function should be
  // executed after user-specified initialization.
  auto init_func_op = builder.create<mlir::FuncOp>(
      module.getLoc(), "_tfrt_resource_init",
      mlir::FunctionType::get(module.getContext(), /*inputs=*/{},
                              /*results=*/{}));
  auto *block = init_func_op.addEntryBlock();
  builder.setInsertionPointToStart(block);

  HoistInfo module_hoist_info;

  for (auto func : module.getOps<mlir::FuncOp>()) {
    // Skips hoisting if this function is an init function or any callees,
    // including recursive ones, of an init functions, because otherwise the
    // hoisted values won't be initialized when this function is called.
    if (IsSessionInitializer(func) || init_callees.contains(func.sym_name()) ||
        func == init_func_op)
      continue;

    HoistInvariantOpsInFunction(func, read_only_vars,
                                side_effect_analysis.GetAnalysisForFunc(func),
                                builder, module_hoist_info);
  }

  // Create tf._TfrtSetResource ops in the init function.
  for (auto iter : llvm::enumerate(module_hoist_info.hoisted_values)) {
    mlir::Value value = iter.value().first;
    int64_t index = iter.index();

    auto new_value = module_hoist_info.value_mapping.lookup(value);
    auto *new_op = new_value.getDefiningOp();
    assert(new_op);
    builder.setInsertionPointAfter(new_op);
    auto set_resource_op = builder.create<mlir::TF::_TfrtSetResourceOp>(
        new_op->getLoc(), new_value, index);

    // Preserve the device attribute.
    llvm::StringRef device = kCpuDeviceName;
    if (auto device_attr = new_op->getAttrOfType<mlir::StringAttr>("device")) {
      if (!device_attr.getValue().empty()) device = device_attr.getValue();
    }
    set_resource_op->setAttr("device", builder.getStringAttr(device));
  }

  builder.setInsertionPointToEnd(block);
  // Finish building the init function by inserting an return op.
  builder.create<mlir::ReturnOp>(init_func_op.getLoc());

  // Now that we have the index for each value that will be replaced, we can
  // create the tf._TfrtGetResource op in each function using these indices.
  ReplaceHoistedValues(module_hoist_info.hoisted_values, builder);

  // Lastly, erase the hoisted ops in reverse topological order.
  for (auto *op :
       llvm::reverse(module_hoist_info.hoists_in_topological_order)) {
    assert(op->use_empty());
    op->erase();
  }
}

// This pass rewrites tf_saved_model dialect's ops according to TFRT's
// requirements:
//
// 1) Remove all tf_saved_model's attributes and ops.
// 2) Create a function for every exported names of the original function.
// 3) Promote all uses of global tensors from resource handles to the underlying
// tensors.
// 4) Hoist invariant ops (ie. guaranteed to return the same value on every
// invocation) for every non-init function.
//
class LowerTFSavedModelPass
    : public mlir::PassWrapper<LowerTFSavedModelPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit LowerTFSavedModelPass(bool hoist_invariant_ops) {
    hoist_invariant_ops_ = hoist_invariant_ops;
  }
  LowerTFSavedModelPass() = default;
  LowerTFSavedModelPass(const LowerTFSavedModelPass &) {}

  llvm::StringRef getArgument() const final {
    return "tfrt-lower-tf-savedmodel";
  }
  llvm::StringRef getDescription() const final {
    return "Lower tf-saved-model ops according to TFRT's requirements.";
  }

  void runOnOperation() override {
    auto module = getOperation();

    // TODO(b/185928201): Create a standalone pass for hoisting invariant ops so
    // that it can be reusable and configurable in other contexts than saved
    // models.
    if (hoist_invariant_ops_) HoistInvariantOps(module);

    // Skip non-savedmodel MLIR module.
    if (!mlir::tf_saved_model::HasTfSavedModelSemantics(module)) return;

    mlir::SymbolTable symbol_table(module);

    // TODO(b/177590991): Remove PromoteGlobalTensors() once non lite MLIR
    // importer is no longer used. PromoteGlobalTensors() is only used for non
    // lite MLIR importer which rewrites resource variables to global_tensors.
    // However, for many models it is not supported.
    for (auto func : module.getOps<mlir::FuncOp>()) {
      if (mlir::tf_saved_model::IsExported(func)) {
        if (mlir::failed(PromoteGlobalTensors(func, symbol_table))) {
          func.emitOpError("failed to promote resource variables.");
          signalPassFailure();
          return;
        }
      }
    }

    module->removeAttr("tf_saved_model.semantics");

    mlir::OpBuilder builder(&getContext());
    auto resource_id = builder.getIdentifier("tf.resource_name");
    auto bound_id = builder.getIdentifier("tf_saved_model.bound_input");
    auto path_id = builder.getIdentifier("tf_saved_model.index_path");

    module.walk([resource_id, bound_id, path_id,
                 &builder](mlir::Operation *op) mutable {
      if (auto func_op = llvm::dyn_cast<mlir::FuncOp>(op)) {
        // Remove tf_saved_model specific function arg attributes.
        for (unsigned i = 0, e = func_op.getNumArguments(); i != e; ++i) {
          if (auto sym = func_op.getArgAttrOfType<mlir::FlatSymbolRefAttr>(
                  i, bound_id)) {
            func_op.removeArgAttr(i, bound_id);
            func_op.setArgAttr(i, resource_id,
                               builder.getStringAttr(sym.getValue()));
          }
          func_op.removeArgAttr(i, path_id);
        }
        for (unsigned i = 0, e = func_op.getNumResults(); i != e; ++i) {
          func_op.removeResultAttr(i, bound_id);
          func_op.removeResultAttr(i, path_id);
        }
        if (auto exported_names = func_op->getAttrOfType<mlir::ArrayAttr>(
                "tf_saved_model.exported_names")) {
          bool is_session_initializer = IsSessionInitializer(func_op);

          // Create a function for each exported name.
          //
          // TODO(b/148477882): TFRT dialect should have similar concepts of
          // exported names so that a function can be referenced by multiple
          // exported names.
          func_op->removeAttr("tf_saved_model.exported_names");
          for (auto exported_name : exported_names) {
            auto exported_func_op = func_op.clone();
            exported_func_op.setName(exported_name.cast<mlir::StringAttr>());

            // If it is a session initializer, we want to maximize parallelism
            // and do not perform any stream merge, to minimize latency.
            //
            // TODO(b/183219530): This is a workaround as the cost model used
            // currently is not very accurate, and leads to performance
            // regression on IO ops that are common in initialization functions.
            if (is_session_initializer) {
              exported_func_op->setAttr("tfrt.cost_threshold",
                                        builder.getI64IntegerAttr(1));
            }

            builder.setInsertionPoint(func_op);
            builder.insert(exported_func_op);
          }
          func_op.erase();
        }
      }
    });

    module.walk([](mlir::Operation *op) {
      if (llvm::isa<mlir::tf_saved_model::TensorFlowSavedModelDialect>(
              op->getDialect())) {
        // Remove all tf_saved_model ops.
        op->erase();
      }
    });
  }

 private:
  // Promote global tensors used by an exported function.
  mlir::LogicalResult PromoteGlobalTensors(
      mlir::FuncOp op, const mlir::SymbolTable &symbol_table);

  // Replace a function argument that is a resource hanndle with an argument of
  // the underlying tensor type. It also replaces all its uses recursively.
  mlir::LogicalResult PromoteFunctionArgument(
      mlir::FuncOp func, unsigned arg_index, mlir::Type promoted_type,
      const mlir::SymbolTable &symbol_table);

  // Replace an operand that is a resource handle with an operand of the
  // underlying type and replace all uses of this operation if the results are
  // also promoted. If it is a control flow op, it will process the callees
  // recursively. The original op will be invalidated in some cases.
  mlir::LogicalResult PromoteOpOperand(mlir::Operation *op,
                                       unsigned operand_number,
                                       mlir::Value promoted,
                                       const mlir::SymbolTable &symbol_table);

  // Replace all uses of a resource handle value with its promoted version
  // recursively.
  mlir::LogicalResult PromoteValueUses(mlir::Value old, mlir::Value promoted,
                                       const mlir::SymbolTable &symbol_table);

  Option<bool> hoist_invariant_ops_{*this, "hoist-invariant-ops",
                                    llvm::cl::desc("hoist-invariant-ops"),
                                    llvm::cl::init(false)};
};

static llvm::SmallVector<unsigned, 4> CompareTypes(mlir::TypeRange x,
                                                   mlir::TypeRange y) {
  llvm::SmallVector<unsigned, 4> results;
  assert(x.size() == y.size());
  for (int i = 0, e = x.size(); i < e; ++i) {
    if (x[i] != y[i]) results.push_back(i);
  }
  return results;
}

mlir::LogicalResult LowerTFSavedModelPass::PromoteGlobalTensors(
    mlir::FuncOp op, const mlir::SymbolTable &symbol_table) {
  for (int i = 0, e = op.getNumArguments(); i < e; ++i) {
    auto global_tensor_op = mlir::tf_saved_model::LookupBoundInputOfType<
        mlir::tf_saved_model::GlobalTensorOp>(op, i, symbol_table);
    if (!global_tensor_op) continue;

    auto result_types = op.getType().getResults();
    if (failed(PromoteFunctionArgument(op, i, global_tensor_op.type(),
                                       symbol_table)))
      return mlir::failure();

    if (!CompareTypes(op.getType().getResults(), result_types).empty())
      op.emitOpError("cannot promote exported functions's results");
  }
  return mlir::success();
}

mlir::LogicalResult LowerTFSavedModelPass::PromoteFunctionArgument(
    mlir::FuncOp func, unsigned arg_index, mlir::Type promoted_type,
    const mlir::SymbolTable &symbol_table) {
  // Replace this argument before replacing its uses.
  auto &block = func.front();
  auto arg = block.getArgument(arg_index);

  auto cleanup_on_failure = llvm::make_scope_exit(
      [&, orig_type = arg.getType()]() { arg.setType(orig_type); });

  arg.setType(promoted_type);

  // Promote all uses of `arg`.
  if (failed(PromoteValueUses(arg, arg, symbol_table))) return mlir::failure();

  cleanup_on_failure.release();

  // Update the function type accordingly.
  auto return_op = llvm::cast<mlir::ReturnOp>(block.getTerminator());
  auto new_results = return_op.operands();

  func.setType(mlir::FunctionType::get(
      func.getContext(), block.getArgumentTypes(), new_results.getTypes()));
  return mlir::success();
}

mlir::LogicalResult LowerTFSavedModelPass::PromoteOpOperand(
    mlir::Operation *op, unsigned operand_number, mlir::Value promoted,
    const mlir::SymbolTable &symbol_table) {
  // TODO(chky): Consider a more scalable way to handling all read-only ops.

  // If it is a ReadVariableOp, we just need to replace all its uses and erase
  // this op.
  if (auto read_var_op = llvm::dyn_cast<mlir::TF::ReadVariableOp>(op)) {
    read_var_op.value().replaceAllUsesWith(promoted);
    op->erase();
    return mlir::success();
  }

  // Next, we handle control flow ops.
  if (!llvm::isa<mlir::TF::IfOp, mlir::TF::CaseOp, mlir::TF::WhileOp,
                 mlir::CallOpInterface, mlir::TF::BatchFunctionOp,
                 mlir::ReturnOp>(op))
    return op->emitOpError("unsupported users of resource variables");

  llvm::SmallVector<unsigned, 2> promoted_result_indices;
  auto update_promoted_result_indices =
      [&promoted_result_indices](
          mlir::Operation *op,
          llvm::ArrayRef<mlir::Type> result_types) -> mlir::LogicalResult {
    if (op->getNumResults() != result_types.size())
      return op->emitOpError(
          "cannot promote call ops whose op resutls do not fully match the "
          "callee results");

    auto result = CompareTypes(op->getResultTypes(), result_types);
    if (promoted_result_indices.empty()) {
      promoted_result_indices.assign(result.begin(), result.end());
    } else {
      // We cannot handle the case where two branches' results are promoted
      // differently.
      if (promoted_result_indices != result)
        return op->emitOpError(
            "cannot promote callees with different result types");
    }
    return mlir::success();
  };

  if (auto if_op = llvm::dyn_cast<mlir::TF::IfOp>(op)) {
    if (operand_number == 0)
      return if_op.emitOpError("cannot promote cond tensor for tf.If");

    auto then_branch = symbol_table.lookup<mlir::FuncOp>(if_op.then_branch());
    auto else_branch = symbol_table.lookup<mlir::FuncOp>(if_op.else_branch());
    assert(then_branch);
    assert(else_branch);

    unsigned arg_index = operand_number - 1;
    for (auto func : {then_branch, else_branch}) {
      if (func.getType().getInput(arg_index) != promoted.getType()) {
        // Rescursively promote the uses in branches.
        if (failed(PromoteFunctionArgument(func, arg_index, promoted.getType(),
                                           symbol_table)))
          return mlir::failure();
      }

      if (failed(update_promoted_result_indices(if_op,
                                                func.getType().getResults())))
        return mlir::failure();
    }
  } else if (auto case_op = llvm::dyn_cast<mlir::TF::CaseOp>(op)) {
    assert(operand_number > 0);
    unsigned arg_index = operand_number - 1;
    for (auto branch_attr : case_op.branches()) {
      auto branch = symbol_table.lookup<mlir::FuncOp>(
          branch_attr.cast<mlir::FlatSymbolRefAttr>().getValue());

      if (branch.getType().getInput(arg_index) != promoted.getType()) {
        // Rescursively promote the uses in branches.
        if (failed(PromoteFunctionArgument(branch, arg_index,
                                           promoted.getType(), symbol_table)))
          return mlir::failure();
      }

      if (failed(update_promoted_result_indices(case_op,
                                                branch.getType().getResults())))
        return mlir::failure();
    }
  } else if (auto while_op = llvm::dyn_cast<mlir::TF::WhileOp>(op)) {
    auto cond = symbol_table.lookup<mlir::FuncOp>(while_op.cond());
    auto body = symbol_table.lookup<mlir::FuncOp>(while_op.body());
    assert(cond);
    assert(body);

    unsigned arg_index = operand_number;
    if (cond.getType().getInput(arg_index) != promoted.getType()) {
      auto cond_result_type = cond.getType().getResult(0);
      if (failed(PromoteFunctionArgument(cond, arg_index, promoted.getType(),
                                         symbol_table)))
        return mlir::failure();

      // We cannot promote the result of cond branch as it may change the
      // behavior of this while op.
      if (cond_result_type != cond.getType().getResult(0))
        return while_op.emitOpError("failed to promote cond for tf.While");
    }

    if (body.getType().getInput(arg_index) != promoted.getType()) {
      if (failed(PromoteFunctionArgument(body, /*arg_index=*/operand_number,
                                         promoted.getType(), symbol_table)))
        return mlir::failure();
    }

    if (failed(update_promoted_result_indices(while_op,
                                              body.getType().getResults())))
      return mlir::failure();

  } else if (auto call_interface = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
    auto callee_name = call_interface.getCallableForCallee()
                           .get<mlir::SymbolRefAttr>()
                           .cast<mlir::FlatSymbolRefAttr>()
                           .getValue();
    auto callee = symbol_table.lookup<mlir::FuncOp>(callee_name);
    assert(callee);

    unsigned arg_index =
        operand_number - call_interface.getArgOperands().getBeginOperandIndex();
    if (callee.getType().getInput(arg_index) != promoted.getType()) {
      if (failed(PromoteFunctionArgument(callee, arg_index, promoted.getType(),
                                         symbol_table)))
        return mlir::failure();
    }

    if (failed(
            update_promoted_result_indices(op, callee.getType().getResults())))
      return mlir::failure();
  } else if (auto batch_function_op =
                 llvm::dyn_cast<mlir::TF::BatchFunctionOp>(op)) {
    auto batch_fn = symbol_table.lookup<mlir::FuncOp>(
        batch_function_op.f().getRootReference());
    assert(batch_fn);

    unsigned arg_index = operand_number;
    if (batch_fn.getType().getInput(arg_index) != promoted.getType()) {
      if (failed(PromoteFunctionArgument(batch_fn, arg_index,
                                         promoted.getType(), symbol_table)))
        return mlir::failure();
    }

    if (failed(update_promoted_result_indices(op,
                                              batch_fn.getType().getResults())))
      return mlir::failure();
  }

  // Replace the operand.
  op->setOperand(operand_number, promoted);

  if (promoted_result_indices.empty()) return mlir::success();

  // If results are also promoted, we need to create a new op with the new
  // results and replaces all uses recursively.

  mlir::OpBuilder builder(op);

  llvm::SmallVector<mlir::Type, 4> new_result_types(op->result_type_begin(),
                                                    op->result_type_end());
  for (unsigned result_number : promoted_result_indices) {
    new_result_types[result_number] = promoted.getType();
  }

  mlir::OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands());
  state.addTypes(new_result_types);
  state.addAttributes(op->getAttrs());

  auto *new_op = builder.createOperation(state);

  // Replace all uses of `op`, and recursively replace those promoted uses.
  for (unsigned i = 0, j = 0, e = op->getNumResults(); i < e; ++i) {
    if (j < promoted_result_indices.size() && promoted_result_indices[j] == i) {
      j++;
      if (failed(PromoteValueUses(op->getResult(i), new_op->getResult(i),
                                  symbol_table))) {
        // On failure, replace all uses of new_op with op and erase the new op.
        new_op->replaceAllUsesWith(op);
        new_op->erase();
        return mlir::failure();
      }
    } else {
      op->getResult(i).replaceAllUsesWith(new_op->getResult(i));
    }
  }

  // On success, erase the original op.
  op->erase();

  return mlir::success();
}

mlir::LogicalResult LowerTFSavedModelPass::PromoteValueUses(
    mlir::Value old, mlir::Value promoted,
    const mlir::SymbolTable &symbol_table) {
  // Retrieve the current uses before replacing the uses as the use list can be
  // invalidated later.
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 4> uses;
  for (auto &use : old.getUses())
    uses.push_back({use.getOwner(), use.getOperandNumber()});

  // Replace uses recursively.
  for (const auto &use : uses) {
    if (failed(PromoteOpOperand(/*op=*/use.first,
                                /*operand_number=*/use.second, promoted,
                                symbol_table)))
      return mlir::failure();
  }

  return mlir::success();
}

// Converts ref variables to resource variables in a few cases.
//
// If the users of one variable in the entire module satisfies the following
// condition, it will be converted to resource variable:
//
// 1) tf.Identity op
// 2) tf.Assign op
// 3) side-effect-free ops: This is also the TF1 behavior that the TF executor
//    will automatically convert ref tensors to non-ref tensors if the user is
//    not expecting a ref tensor. Refer to
//    http://cs?q=tensorflow/core/common_runtime/executor.cc:932%20at_cl:356873227
class ConvertReferenceVariableToResourceVariablePass
    : public mlir::PassWrapper<ConvertReferenceVariableToResourceVariablePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  llvm::StringRef getArgument() const final {
    return "tfrt-convert-ref-variables";
  }
  llvm::StringRef getDescription() const final {
    return "Convert reference variable to resource variables.";
  }
  void runOnOperation() override;
};

mlir::LogicalResult ConvertReferenceVariableToResourceVariable(
    mlir::TF::VariableV2Op var_op) {
  auto tensor_type =
      mlir::TF::DropRefType(var_op.ref().getType()).cast<mlir::TensorType>();

  llvm::SmallVector<mlir::TF::IdentityOp, 4> identity_ops;
  llvm::SmallVector<mlir::TF::AssignOp, 4> assign_ops;
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 4>
      side_effect_free_ops;

  for (mlir::OpOperand &use : var_op.ref().getUses()) {
    mlir::Operation *user = use.getOwner();

    if (auto identity = llvm::dyn_cast<mlir::TF::IdentityOp>(user)) {
      identity_ops.push_back(identity);
      continue;
    } else if (auto assign = llvm::dyn_cast<mlir::TF::AssignOp>(user)) {
      // Conservatively we only allow the case that the output of this tf.Assign
      // is not consumed by any other ops.
      if (assign.output_ref().use_empty()) {
        assign_ops.push_back(assign);
        continue;
      }
    } else if (mlir::MemoryEffectOpInterface::hasNoEffect(user)) {
      side_effect_free_ops.push_back({user, use.getOperandNumber()});
      continue;
    }

    return var_op.emitOpError()
           << "failed to convert reference variables with unexpected users. "
           << *user;
  }

  mlir::OpBuilder builder(var_op);

  auto var_handle_op = builder.create<mlir::TF::VarHandleOp>(
      var_op.getLoc(),
      mlir::RankedTensorType::get(
          {}, mlir::TF::ResourceType::get(
                  llvm::ArrayRef<mlir::TensorType>{tensor_type},
                  builder.getContext())),
      var_op.container(), var_op.shared_name());

  for (auto op : identity_ops) {
    // Set insertion point to this identity_op so that the side-effect
    // visibility is preserved.
    builder.setInsertionPoint(op);
    auto read_var_op = builder.create<mlir::TF::ReadVariableOp>(
        op.getLoc(), op.getType(), var_handle_op);
    op.replaceAllUsesWith(read_var_op.value());
    op.erase();
  }

  for (auto op : assign_ops) {
    // Set the insertion point after the assign op so that all operands are
    // dominating the newly created op.
    builder.setInsertionPoint(op);
    builder.create<mlir::TF::AssignVariableOp>(op.getLoc(), var_handle_op,
                                               op.value());
    op.erase();
  }

  for (auto pair : side_effect_free_ops) {
    mlir::Operation *op = pair.first;
    unsigned idx = pair.second;
    // Set the insertion point after the op so that all operands are dominating
    // the newly created op.
    builder.setInsertionPoint(op);
    // Create a new read variable op, so that the side-effects are preserved.
    auto read_var_op = builder.create<mlir::TF::ReadVariableOp>(
        op->getLoc(), tensor_type, var_handle_op);
    op->setOperand(idx, read_var_op.value());
  }

  return mlir::success();
}

void ConvertReferenceVariableToResourceVariablePass::runOnOperation() {
  auto module = getOperation();

  // The key here is a tuple of device, container and shared_name to uniquely
  // identify a variable.
  llvm::DenseMap<std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>,
                 llvm::SmallVector<mlir::TF::VariableV2Op, 4>>
      ref_vars;

  // First, we collect all variables' corresponding tf.VariableV2 ops.
  module.walk([&ref_vars](mlir::TF::VariableV2Op op) {
    if (op.shared_name().empty()) {
      op.emitOpError()
          << "unable to convert reference variables with empty shared_names.";
      return mlir::WalkResult::interrupt();
    }

    llvm::StringRef device;
    if (auto device_attr = op->getAttrOfType<mlir::StringAttr>("device")) {
      device = device_attr.getValue();
    }

    ref_vars[{device, op.container(), op.shared_name()}].push_back(op);

    return mlir::WalkResult::advance();
  });

  // Then we perform rewrite for each variable if possible.
  for (const auto &iter : ref_vars) {
    const auto &var_ops = iter.second;

    for (auto var_op : var_ops) {
      if (mlir::succeeded(ConvertReferenceVariableToResourceVariable(var_op)))
        var_op.erase();
    }
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateLowerTFSavedModelPass(bool hoist_invariant_ops) {
  return std::make_unique<LowerTFSavedModelPass>(hoist_invariant_ops);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateConvertReferenceVariableToResourceVariablePass() {
  return std::make_unique<ConvertReferenceVariableToResourceVariablePass>();
}

static mlir::PassRegistration<LowerTFSavedModelPass> saved_model_pass;

static mlir::PassRegistration<ConvertReferenceVariableToResourceVariablePass>
    ref_var_pass;

}  // namespace tensorflow

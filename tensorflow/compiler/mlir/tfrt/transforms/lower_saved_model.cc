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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/utils.h"

namespace tensorflow {
namespace {

using ::mlir::tf_saved_model::kTfSavedModelExportedNamesAttr;
using ::mlir::tf_saved_model::kTfSavedModelIndexPathAttr;

constexpr char kCpuDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

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
  mlir::IRMapping value_mapping;

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

bool OnlyHasReadOrNoEffect(mlir::Operation *op) {
  auto interface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op);
  if (!interface) return false;
  return interface.onlyHasEffect<mlir::MemoryEffects::Read>() ||
         interface.hasNoEffect();
}

bool CanHoist(const llvm::DenseSet<mlir::TF::ResourceHandle> &read_only_vars,
              mlir::Operation *op) {
  // return ops should not be hoisted.
  if (op->mightHaveTrait<mlir::OpTrait::IsTerminator>()) return false;

  // Fixes a corner case where hoisting the tf.BatchFunction leads to
  // a compilation error; such a case may occur in unit tests.
  if (llvm::isa<mlir::TF::BatchFunctionOp>(op)) return false;

  // Non-side-effecting ops can be hoisted.
  if (mlir::isMemoryEffectFree(op)) return true;

  // ResourceHandle ops can be hoisted.
  if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::HashTableV2Op>(op))
    return true;

  // If it is ReadVariableOp and the variable is readonly, it can be hoisted.
  if (auto read_var_op = llvm::dyn_cast<mlir::TF::ReadVariableOp>(op)) {
    if (auto var_handle_op = llvm::dyn_cast_or_null<mlir::TF::VarHandleOp>(
            read_var_op.getResource().getDefiningOp())) {
      if (read_only_vars.count(GetResourceHandle(var_handle_op)) > 0)
        return true;
    }
  }

  // If it is LookupTableSizeOp, it can be hoisted as the size of the hash table
  // cannot be changed after initialization.
  if (auto lookup_table_size_op =
          llvm::dyn_cast<mlir::TF::LookupTableSizeV2Op>(op)) {
    if (auto hash_table_op = llvm::dyn_cast_or_null<mlir::TF::HashTableV2Op>(
            lookup_table_size_op.getTableHandle().getDefiningOp())) {
      if (read_only_vars.count(GetResourceHandle(hash_table_op)) > 0)
        return true;
    }
  }

  // TODO(chky): Allow more readonly ops.

  return false;
}

void HoistInvariantOpsInFunction(
    mlir::func::FuncOp func,
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

void FindCalleesRecursiveForOp(const mlir::SymbolTable &symbol_table,
                               mlir::Operation *op,
                               llvm::StringSet<> &callees) {
  for (const auto &named_attr : op->getAttrs()) {
    if (auto symbol_attr =
            named_attr.getValue().dyn_cast<mlir::FlatSymbolRefAttr>()) {
      auto symbol = symbol_attr.getValue();
      if (!callees.contains(symbol)) {
        callees.insert(symbol);

        auto func = symbol_table.lookup<mlir::func::FuncOp>(symbol);
        if (!func) continue;

        func.walk([&](mlir::Operation *op) {
          FindCalleesRecursiveForOp(symbol_table, op, callees);
        });
      }
    }
  }
}

void FindCalleesRecursive(const mlir::SymbolTable &symbol_table,
                          mlir::func::FuncOp func, llvm::StringSet<> &callees) {
  assert(func);
  func.walk([&](mlir::Operation *op) {
    FindCalleesRecursiveForOp(symbol_table, op, callees);
  });
}

// This pass rewrites tf_saved_model dialect's ops according to TFRT's
// requirements:
//
// 1) Remove all tf_saved_model's attributes and ops.
// 2) Create a function for every exported names of the original function.
// 3) Hoist invariant ops (ie. guaranteed to return the same value on every
// invocation) for every non-init function.
//
class LowerTFSavedModelPass
    : public mlir::PassWrapper<LowerTFSavedModelPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTFSavedModelPass)

  explicit LowerTFSavedModelPass(bool hoist_invariant_ops,
                                 bool fuse_get_resource_ops) {
    hoist_invariant_ops_ = hoist_invariant_ops;
    fuse_get_resource_ops_ = fuse_get_resource_ops;
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

    module->removeAttr("tf_saved_model.semantics");

    mlir::OpBuilder builder(&getContext());
    auto resource_id = builder.getStringAttr("tf.resource_name");
    auto bound_id = builder.getStringAttr("tf_saved_model.bound_input");
    auto path_id = builder.getStringAttr(kTfSavedModelIndexPathAttr);

    module.walk([resource_id, bound_id, path_id,
                 &builder](mlir::Operation *op) mutable {
      if (auto func_op = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
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
                kTfSavedModelExportedNamesAttr)) {
          bool is_session_initializer = IsSessionInitializer(func_op);

          // Create a function for each exported name.
          //
          // TODO(b/148477882): TFRT dialect should have similar concepts of
          // exported names so that a function can be referenced by multiple
          // exported names.
          func_op->removeAttr(kTfSavedModelExportedNamesAttr);
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
  void HoistInvariantOps(mlir::ModuleOp module);
  void ReplaceHoistedValues(
      llvm::ArrayRef<std::pair<mlir::Value, mlir::TF::ResourceHandle>>
          hoisted_values,
      mlir::OpBuilder &builder);

  Option<bool> hoist_invariant_ops_{*this, "hoist-invariant-ops",
                                    llvm::cl::desc("hoist-invariant-ops"),
                                    llvm::cl::init(false)};
  Option<bool> fuse_get_resource_ops_{*this, "fuse-get-resource-ops",
                                      llvm::cl::desc("fuse get resource ops"),
                                      llvm::cl::init(true)};
};

void LowerTFSavedModelPass::HoistInvariantOps(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);

  // Find all resources used in non-init functions.
  llvm::DenseMap<mlir::TF::ResourceHandle,
                 llvm::SmallVector<mlir::Operation *, 4>>
      resources;

  // Find all callees referenced in the initialization functions.
  llvm::StringSet<> init_callees;

  // Recursively find all callees referenced in the tf.XlaLaunch op.
  // At and after the point of calling this pass, the MLIR xla function is no
  // longer used. So there is no point to do hoisting for xla functions.
  llvm::StringSet<> xla_launch_callees;

  module.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::TF::VarHandleOp, mlir::TF::HashTableV2Op>(op)) {
      auto func = op->getParentOfType<mlir::func::FuncOp>();
      if (IsSessionInitializer(func)) return;
      resources[GetResourceHandle(op)].push_back(op);
    } else if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
      if (!IsSessionInitializer(func)) return;
      FindCalleesRecursive(symbol_table, func, init_callees);
    } else if (llvm::isa<mlir::TF::XlaLaunchOp>(op)) {
      // TODO(b/275095412): Clean up MLIR XLA functions after they are written
      // back to function library, so that we don't need to do special handling
      // for those functions here.
      FindCalleesRecursiveForOp(symbol_table, op, xla_launch_callees);
    }
  });

  llvm::DenseSet<mlir::TF::ResourceHandle> read_only_vars;
  for (const auto &iter : resources) {
    const auto &key = iter.first;
    const auto &vars = iter.second;
    if (std::all_of(vars.begin(), vars.end(), [](mlir::Operation *op) {
          for (auto *user : op->getUsers()) {
            if (!OnlyHasReadOrNoEffect(user)) return false;
          }
          return true;
        })) {
      read_only_vars.insert(key);
    }
  }

  mlir::TF::SideEffectAnalysis side_effect_analysis(module);

  mlir::OpBuilder builder(&module.getBodyRegion());
  // "_tfrt_resource_init" is the special function that executes all invariant
  // ops (eg. read-only variables) used in the model. This function should be
  // executed after user-specified initialization.
  auto init_func_op = builder.create<mlir::func::FuncOp>(
      module.getLoc(), "_tfrt_resource_init",
      mlir::FunctionType::get(module.getContext(), /*inputs=*/{},
                              /*results=*/{}));
  auto *block = init_func_op.addEntryBlock();
  builder.setInsertionPointToStart(block);

  HoistInfo module_hoist_info;

  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    // Skips hoisting if this function is an init function or any callees,
    // including recursive ones, of an init functions, because otherwise the
    // hoisted values won't be initialized when this function is called.
    if (IsSessionInitializer(func) ||
        init_callees.contains(func.getSymName()) || func == init_func_op ||
        xla_launch_callees.contains(func.getSymName()))
      continue;

    // Skips hoisting if this function runs on TPU. This is will happen when
    // fallback to TPUPartitionedCallOp is enabled for SPMD.
    // TODO(b/214039254): remove this once tfrt support native SPMD.
    bool has_tpu_op = false;
    func.walk([&has_tpu_op](mlir::Operation *op) {
      if (op->hasAttr("_tpu_replicate")) has_tpu_op = true;
    });
    if (has_tpu_op) continue;

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
  builder.create<mlir::func::ReturnOp>(init_func_op.getLoc());

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

void LowerTFSavedModelPass::ReplaceHoistedValues(
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

      llvm::SmallVector<mlir::Value> new_values;

      if (fuse_get_resource_ops_) {
        auto get_resource_op = builder.create<mlir::TF::_TfrtGetResourceOp>(
            block->getParentOp()->getLoc(), old_values.getTypes(),
            builder.getI64ArrayAttr(indices),
            builder.getStrArrayAttr(shared_name_arr),
            builder.getStrArrayAttr(container_arr));
        get_resource_op->setAttr("device", builder.getStringAttr(device));
        new_values = get_resource_op.getResults();
      } else {
        for (int i = 0; i < old_values.size(); ++i) {
          auto get_resource_op = builder.create<mlir::TF::_TfrtGetResourceOp>(
              block->getParentOp()->getLoc(),
              mlir::TypeRange(old_values[i].getType()),
              builder.getI64ArrayAttr(indices[i]),
              builder.getStrArrayAttr(shared_name_arr[i]),
              builder.getStrArrayAttr(container_arr[i]));
          get_resource_op->setAttr("device", builder.getStringAttr(device));
          new_values.append(get_resource_op->result_begin(),
                            get_resource_op->result_end());
        }
      }

      for (auto iter : llvm::zip(old_values, new_values)) {
        auto old_value = std::get<0>(iter);
        auto new_value = std::get<1>(iter);
        old_value.replaceAllUsesWith(new_value);
      }
    }
  }
}

static llvm::SmallVector<unsigned, 4> CompareTypes(mlir::TypeRange x,
                                                   mlir::TypeRange y) {
  llvm::SmallVector<unsigned, 4> results;
  assert(x.size() == y.size());
  for (int i = 0, e = x.size(); i < e; ++i) {
    if (x[i] != y[i]) results.push_back(i);
  }
  return results;
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

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertReferenceVariableToResourceVariablePass)
};

mlir::LogicalResult ConvertReferenceVariableToResourceVariable(
    mlir::TF::VariableV2Op var_op) {
  auto tensor_type =
      mlir::TF::DropRefType(var_op.getRef().getType()).cast<mlir::TensorType>();

  llvm::SmallVector<mlir::TF::IdentityOp, 4> identity_ops;
  llvm::SmallVector<mlir::TF::AssignOp, 4> assign_ops;
  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 4>
      side_effect_free_ops;

  for (mlir::OpOperand &use : var_op.getRef().getUses()) {
    mlir::Operation *user = use.getOwner();

    if (auto identity = llvm::dyn_cast<mlir::TF::IdentityOp>(user)) {
      identity_ops.push_back(identity);
      continue;
    } else if (auto assign = llvm::dyn_cast<mlir::TF::AssignOp>(user)) {
      // Conservatively we only allow the case that the output of this tf.Assign
      // is not consumed by any other ops.
      if (assign.getOutputRef().use_empty()) {
        assign_ops.push_back(assign);
        continue;
      }
    } else if (mlir::isMemoryEffectFree(user)) {
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
      var_op.getContainer(), var_op.getSharedName());

  for (auto op : identity_ops) {
    // Set insertion point to this identity_op so that the side-effect
    // visibility is preserved.
    builder.setInsertionPoint(op);
    auto read_var_op = builder.create<mlir::TF::ReadVariableOp>(
        op.getLoc(), op.getType(), var_handle_op);
    op.replaceAllUsesWith(read_var_op.getValue());
    op.erase();
  }

  for (auto op : assign_ops) {
    // Set the insertion point after the assign op so that all operands are
    // dominating the newly created op.
    builder.setInsertionPoint(op);
    builder.create<mlir::TF::AssignVariableOp>(op.getLoc(), var_handle_op,
                                               op.getValue());
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
    op->setOperand(idx, read_var_op.getValue());
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
    if (op.getSharedName().empty()) {
      op.emitOpError()
          << "unable to convert reference variables with empty shared_names.";
      return mlir::WalkResult::interrupt();
    }

    llvm::StringRef device;
    if (auto device_attr = op->getAttrOfType<mlir::StringAttr>("device")) {
      device = device_attr.getValue();
    }

    ref_vars[{device, op.getContainer(), op.getSharedName()}].push_back(op);

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
CreateLowerTFSavedModelPass(bool hoist_invariant_ops,
                            bool fuse_get_resource_ops) {
  return std::make_unique<LowerTFSavedModelPass>(hoist_invariant_ops,
                                                 fuse_get_resource_ops);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateConvertReferenceVariableToResourceVariablePass() {
  return std::make_unique<ConvertReferenceVariableToResourceVariablePass>();
}

static mlir::PassRegistration<LowerTFSavedModelPass> saved_model_pass;

static mlir::PassRegistration<ConvertReferenceVariableToResourceVariablePass>
    ref_var_pass;

}  // namespace tensorflow

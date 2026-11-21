/* Copyright 2024 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/import/callee_symbol_manager.h"
#include "xla/python/ifrt/ir/import/passes.h"
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTPOPULATEATOMPROGRAMMETADATAPASS
#include "xla/python/ifrt/ir/import/passes.h.inc"

namespace {

IfrtDevicesAttr GetDeviceAttrPermutation(
    const llvm::DenseMap<int, int>& device_id_to_index,
    IfrtDevicesAttr array_devices_attr, mlir::OpBuilder& builder) {
  std::vector<int> device_permutation;
  device_permutation.reserve(array_devices_attr.getIds().size());
  for (int device_id : array_devices_attr.getIds()) {
    device_permutation.push_back(device_id_to_index.at(device_id));
  }
  return IfrtDevicesAttr::get(builder.getContext(), device_permutation);
}

struct CallOpDeviceCache {
  // Map from logical device ID to index within the device list. It is only
  // populated if there's an IfrtArrayType arg/result with a different devices
  // attribute than the CallOp.
  llvm::DenseMap<int, int> device_id_to_index;
  // Map from device attribute to device attribute representing the permutation
  // to apply to the CallOp's device attribute to get device attribute key.
  llvm::DenseMap<IfrtDevicesAttr, IfrtDevicesAttr> permutation_attrs;
  bool device_id_to_index_populated = false;
};

// Populates the metadata on the atom program `ModuleOp` and `main` `FuncOp`.
void PopulateMetadata(CallOp call_op, mlir::ModuleOp module_op,
                      mlir::func::FuncOp callee_op, mlir::OpBuilder& builder,
                      CallOpDeviceCache& device_cache) {
  module_op->setAttr(kIfrtNumDevicesAttrName,
                     builder.getI32IntegerAttr(call_op.getDevices().size()));
  // Copy ifrt.compile_options_key if it exists.
  if (call_op->hasAttr(kIfrtCompileOptionsKey)) {
    module_op->setAttr(kIfrtCompileOptionsKey,
                       call_op->getAttr(kIfrtCompileOptionsKey));
  }
  // Copy `ifrt.local_view` attribute if it exists.
  if (call_op->hasAttr(kIfrtLocalViewAttrName)) {
    module_op->setAttr(kIfrtLocalViewAttrName,
                       call_op->getAttr(kIfrtLocalViewAttrName));
  }

  IfrtDevicesAttr call_op_devices_attr = call_op.getDevicesAttr();
  auto get_device_permutation = [&](IfrtDevicesAttr array_devices_attr) {
    if (auto it = device_cache.permutation_attrs.find(array_devices_attr);
        it != device_cache.permutation_attrs.end()) {
      return it->second;
    }
    if (!device_cache.device_id_to_index_populated) {
      for (const auto& [idx, device_id] :
           llvm::enumerate(call_op_devices_attr.getIds())) {
        device_cache.device_id_to_index[device_id] = idx;
      }
      device_cache.device_id_to_index_populated = true;
    }
    IfrtDevicesAttr permutation = GetDeviceAttrPermutation(
        device_cache.device_id_to_index, array_devices_attr, builder);
    device_cache.permutation_attrs[array_devices_attr] = permutation;
    return permutation;
  };

  // Attach sharding to inputs.
  for (const auto& [i, input] : llvm::enumerate(call_op.getInputs())) {
    const IfrtArrayType array_type = GetArrayType(input);
    // It is faster to get all the attributes and add the new ones than
    // setting the new attributes one-by-one. This is because the logic that
    // sets an attribute converts the attr dict to a NamedAttrList, and then
    // linearly searches for the attr.
    llvm::SmallVector<mlir::NamedAttribute, 16> arg_attrs;
    if (mlir::DictionaryAttr arg_attr_dict = callee_op.getArgAttrDict(i);
        arg_attr_dict != nullptr) {
      arg_attrs.append(arg_attr_dict.begin(), arg_attr_dict.end());
    }
    arg_attrs.push_back(builder.getNamedAttr(kIfrtShardingAttrName,
                                             array_type.getShardingAttr()));
    // Only attach a devices attribute if the array's devices attribute is
    // different from the devices attribute on the CallOp. Otherwise, the
    // attribute is redundant.
    if (IfrtDevicesAttr array_devices_attr = array_type.getDevicesAttr();
        array_devices_attr != call_op_devices_attr) {
      arg_attrs.push_back(builder.getNamedAttr(
          kIfrtDevicesAttrName, get_device_permutation(array_devices_attr)));
    }
    if (array_type.getMemoryKindAttr()) {
      arg_attrs.push_back(builder.getNamedAttr(kIfrtMemoryKindAttrName,
                                               array_type.getMemoryKindAttr()));
    }
    callee_op.setArgAttrs(i, arg_attrs);
  }

  // Attach sharding to outputs.
  for (const auto& [i, output] : llvm::enumerate(call_op.getOutputs())) {
    const IfrtArrayType array_type = GetArrayType(output);
    llvm::SmallVector<mlir::NamedAttribute, 16> res_attrs;
    if (mlir::DictionaryAttr res_attr_dict = callee_op.getResultAttrDict(i);
        res_attr_dict != nullptr) {
      res_attrs.append(res_attr_dict.begin(), res_attr_dict.end());
    }
    res_attrs.push_back(builder.getNamedAttr(kIfrtShardingAttrName,
                                             array_type.getShardingAttr()));
    // Only attach a devices attribute if the array's devices attribute is
    // different from the devices attribute on the CallOp. Otherwise, the
    // attribute is redundant.
    if (IfrtDevicesAttr array_devices_attr = array_type.getDevicesAttr();
        array_devices_attr != call_op_devices_attr) {
      res_attrs.push_back(builder.getNamedAttr(
          kIfrtDevicesAttrName, get_device_permutation(array_devices_attr)));
    }
    if (array_type.getMemoryKindAttr()) {
      res_attrs.push_back(builder.getNamedAttr(kIfrtMemoryKindAttrName,
                                               array_type.getMemoryKindAttr()));
    }
    callee_op.setResultAttrs(i, res_attrs);
  }

  // Alias inputs.
  for (const IoAlias& io_alias : call_op.getIoAliasPairs()) {
    callee_op.setArgAttr(io_alias.input_index, kAliasingOutputAttrName,
                         builder.getI32IntegerAttr(io_alias.output_index));
  }
  for (const int32_t idx : call_op.getDonatedInputIndices()) {
    callee_op.setArgAttr(idx, kBufferDonationAttrName,
                         builder.getBoolAttr(true));
  }
}

class IfrtPopulateAtomProgramMetadataPass
    : public impl::IfrtPopulateAtomProgramMetadataPassBase<
          IfrtPopulateAtomProgramMetadataPass> {
 public:
  void runOnOperation() override;
};

void IfrtPopulateAtomProgramMetadataPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  mlir::SymbolTableCollection symbol_table;
  mlir::FailureOr<CalleeSymbolManager> symbol_manager =
      CalleeSymbolManager::Create(module_op, &symbol_table);
  if (mlir::failed(symbol_manager)) {
    signalPassFailure();
    return;
  }
  mlir::OpBuilder builder(&getContext());
  mlir::func::FuncOp main_func = GetMainFunction(module_op);

  // Group `CallOp`s by `(callee, metadata)` in forward program order using
  // `IfrtCallOpInfo`, scoping `group_idx_by_call` to the read-only scan so it
  // is destroyed before any `CallOp`'s callee attribute is mutated. Because
  // each callee's earliest group appears first in `all_groups`, iterating
  // `llvm::reverse(all_groups)` clones later variants first and lets the
  // earliest group claim the original private callee in place (with zero clones
  // when all calls share one variant).
  llvm::SmallVector<llvm::SmallVector<CallOp, 2>> all_groups;
  {
    llvm::DenseMap<CallOp, size_t, IfrtCallOpInfo> group_idx_by_call;
    for (CallOp call_op : main_func.getOps<CallOp>()) {
      if (mlir::failed(GetOutlinedAtomProgramModule(call_op, symbol_table))) {
        signalPassFailure();
        return;
      }
      auto [group_it, group_inserted] =
          group_idx_by_call.insert({call_op, all_groups.size()});
      if (group_inserted) {
        all_groups.push_back({call_op});
      } else {
        all_groups[group_it->second].push_back(call_op);
      }
    }
  }

  llvm::DenseMap<IfrtDevicesAttr, CallOpDeviceCache> device_permutation_caches;
  for (const llvm::SmallVector<CallOp, 2>& calls : llvm::reverse(all_groups)) {
    mlir::FailureOr<mlir::func::FuncOp> target_callee =
        symbol_manager->GetOrCloneCalleeForMutation(calls);
    if (mlir::failed(target_callee)) {
      signalPassFailure();
      return;
    }
    CallOp representative_call = calls.front();
    mlir::ModuleOp target_module =
        llvm::cast<mlir::ModuleOp>((*target_callee)->getParentOp());
    PopulateMetadata(
        representative_call, target_module, *target_callee, builder,
        device_permutation_caches[representative_call.getDevicesAttr()]);
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

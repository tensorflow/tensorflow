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

#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTPOPULATEATOMPROGRAMMETADATAPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

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

// Populates the metadata on the atom program ModuleOp and `main` FuncOp.
mlir::LogicalResult PopulateMetadata(CallOp call_op, mlir::ModuleOp module_op,
                                     mlir::func::FuncOp callee_op,
                                     mlir::OpBuilder& builder,
                                     CallOpDeviceCache& device_cache) {
  module_op->setAttr(kIfrtNumDevicesAttrName,
                     builder.getI32IntegerAttr(call_op.getDevices().size()));
  // Copy ifrt.compile_options_key if it exists.
  if (call_op->hasAttr(kIfrtCompileOptionsKey)) {
    module_op->setAttr(kIfrtCompileOptionsKey,
                       call_op->getAttr(kIfrtCompileOptionsKey));
  }
  // Copy `ifrt.local_view` attribute if it exists.
  if (call_op->hasAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName)) {
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
  for (const auto& raw_io_alias :
       call_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    llvm::ArrayRef<int> io_alias_as_array = raw_io_alias.asArrayRef();
    callee_op.setArgAttr(io_alias_as_array[0], kAliasingOutputAttrName,
                         builder.getI32IntegerAttr(io_alias_as_array[1]));
  }
  for (const int32_t idx : call_op.getDonatedInputIndices()) {
    callee_op.setArgAttr(idx, kBufferDonationAttrName,
                         builder.getBoolAttr(true));
  }
  return mlir::success();
}

class IfrtPopulateAtomProgramMetadataPass
    : public impl::IfrtPopulateAtomProgramMetadataPassBase<
          IfrtPopulateAtomProgramMetadataPass> {
 public:
  void runOnOperation() override;
};

void IfrtPopulateAtomProgramMetadataPass::runOnOperation() {
  mlir::MLIRContext& context = getContext();
  mlir::SymbolTableCollection symbol_table;
  mlir::OpBuilder builder(&context);
  mlir::func::FuncOp main_func = GetMainFunction(getOperation());

  // Construct a map from callee `SymbolRefAttr` to a set of `CallOps` calling
  // it. This map is used to decide if an atom program module must be cloned
  // before populating its metadata (i.e., used more than once).
  llvm::DenseMap<mlir::SymbolRefAttr, llvm::DenseSet<CallOp, IfrtCallOpInfo>>
      callee_to_callops;
  for (CallOp call_op : main_func.getOps<CallOp>()) {
    callee_to_callops[call_op.getCallee()].insert(call_op);
  }

  llvm::DenseMap<IfrtDevicesAttr, CallOpDeviceCache> device_permutation_caches;
  llvm::DenseMap<CallOp, mlir::SymbolRefAttr, IfrtCallOpInfo> visited_call_ops;
  // Walk the CallOps in reverse order to ensure that the first CallOp using a
  // callee uses the original callee. Otherwise, the walk would modify the name
  // of the default callee.
  mlir::WalkResult result = main_func.walk<mlir::WalkOrder::PreOrder,
                                           mlir::ReverseIterator>(
      [&](CallOp call_op) -> mlir::WalkResult {
        mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
        if (callee == nullptr) {
          call_op->emitOpError()
              << "can't find callee `" << call_op.getCalleeAttr() << "`";
          return mlir::WalkResult::interrupt();
        }
        mlir::ModuleOp callee_module =
            llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
        if (callee.getSymName() != kCalleeMainFuncName ||
            callee_module == nullptr) {
          call_op.emitOpError()
              << "requires callee outlined as `" << kCalleeMainFuncName
              << "` function in a ModuleOp. Actual callee name: "
              << callee.getSymName()
              << ". Actual callee parent: " << callee->getParentOp()->getName();
          return mlir::WalkResult::interrupt();
        }

        if (auto call_op_it = visited_call_ops.find(call_op);
            call_op_it != visited_call_ops.end()) {
          // Set the callee attribute to the existing callee for this
          // CallOp.
          call_op.setCalleeAttr(call_op_it->second);
          return mlir::WalkResult::advance();
        }

        callee_to_callops[call_op.getCallee()].erase(call_op);
        if (callee_to_callops[call_op.getCallee()].empty()) {
          // There's only one CallOp for this callee, so we can populate the
          // metadata in place.
          if (mlir::failed(PopulateMetadata(
                  call_op, callee_module, callee, builder,
                  device_permutation_caches[call_op.getDevicesAttr()]))) {
            return mlir::WalkResult::interrupt();
          }
          visited_call_ops[call_op.clone()] = call_op.getCalleeAttr();
          return mlir::WalkResult::advance();
        }

        // Clone the callee module because there are multiple CallOps
        // calling it.
        mlir::ModuleOp cloned_module = callee_module.clone();
        mlir::func::FuncOp cloned_callee = GetMainFunction(cloned_module);
        // Insert new cloned atom program module in the SymbolTable.
        symbol_table
            .getSymbolTable(
                callee_module->getParentWithTrait<mlir::OpTrait::SymbolTable>())
            .insert(cloned_module);
        if (mlir::failed(PopulateMetadata(
                call_op, cloned_module, cloned_callee, builder,
                device_permutation_caches[call_op.getDevicesAttr()]))) {
          return mlir::WalkResult::interrupt();
        }
        mlir::SymbolRefAttr callee_attr = mlir::SymbolRefAttr::get(
            cloned_module.getSymNameAttr(),
            mlir::SymbolRefAttr::get(cloned_callee.getSymNameAttr()));
        // Clone the CallOp because it will be modified next.
        visited_call_ops[call_op.clone()] = callee_attr;
        call_op.setCalleeAttr(callee_attr);

        return mlir::WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    signalPassFailure();
  }

  // Erase the cloned CallOp because they were used only as keys of the map.
  for (auto& [call_op, unused] : visited_call_ops) {
    call_op.erase();
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla

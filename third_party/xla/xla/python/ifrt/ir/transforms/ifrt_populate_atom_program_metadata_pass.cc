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

#include <memory>
#include <utility>

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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTPOPULATEATOMPROGRAMMETADATAPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Populates the metadata on the atom program ModuleOp and `main` FuncOp.
mlir::LogicalResult PopulateMetadata(CallOp call_op, mlir::ModuleOp module_op,
                                     mlir::func::FuncOp callee_op,
                                     mlir::OpBuilder& builder) {
  module_op->setAttr(kIfrtNumDevicesAttrName,
                     builder.getI32IntegerAttr(call_op.getDevices().size()));
  // Copy `ifrt.local_view` attribute if it exists.
  if (call_op->hasAttrOfType<mlir::UnitAttr>(kIfrtLocalViewAttrName)) {
    module_op->setAttr(kIfrtLocalViewAttrName,
                       call_op->getAttr(kIfrtLocalViewAttrName));
  }

  // Attach sharding to inputs.
  for (const auto& [i, input] : llvm::enumerate(call_op.getInputs())) {
    const auto array = mlir::dyn_cast_or_null<IfrtArrayType>(input.getType());
    if (array == nullptr) {
      return call_op->emitOpError()
             << "requires all inputs to be IfrtArrayType. Input #" << i << ": "
             << input.getType();
    }
    // It is faster to get all the attributes and add the new ones than
    // setting the new attributes one-by-one. This is because the logic that
    // sets an attribute converts the attr dict to a NamedAttrList, and then
    // linearly searches for the attr.
    llvm::SmallVector<mlir::NamedAttribute, 16> arg_attrs;
    auto arg_attr_dict = callee_op.getArgAttrDict(i);
    if (arg_attr_dict != nullptr) {
      arg_attrs.append(arg_attr_dict.begin(), arg_attr_dict.end());
    }
    arg_attrs.push_back(
        builder.getNamedAttr(kIfrtShardingAttrName, array.getShardingAttr()));
    if (array.getMemoryKindAttr()) {
      arg_attrs.push_back(builder.getNamedAttr(kIfrtMemoryKindAttrName,
                                               array.getMemoryKindAttr()));
    }
    callee_op.setArgAttrs(i, arg_attrs);
  }

  // Attach sharding to outputs.
  for (const auto& [i, output] : llvm::enumerate(call_op.getOutputs())) {
    const auto array = mlir::dyn_cast_or_null<IfrtArrayType>(output.getType());
    if (array == nullptr) {
      return call_op->emitOpError()
             << "requires all outputs to be IfrtArrayType. Input #" << i << ": "
             << output.getType();
    }
    llvm::SmallVector<mlir::NamedAttribute, 16> res_attrs;
    auto res_attr_dict = callee_op.getResultAttrDict(i);
    if (res_attr_dict != nullptr) {
      res_attrs.append(res_attr_dict.begin(), res_attr_dict.end());
    }
    res_attrs.push_back(
        builder.getNamedAttr(kIfrtShardingAttrName, array.getShardingAttr()));
    if (array.getMemoryKindAttr()) {
      res_attrs.push_back(builder.getNamedAttr(kIfrtMemoryKindAttrName,
                                               array.getMemoryKindAttr()));
    }
    callee_op.setResultAttrs(i, res_attrs);
  }

  // Alias inputs.
  for (const auto& raw_io_alias :
       call_op.getIoAliases().getAsRange<mlir::DenseI32ArrayAttr>()) {
    llvm::ArrayRef<int> io_alias_as_array = raw_io_alias.asArrayRef();
    callee_op.setArgAttr(io_alias_as_array[0], "tf.aliasing_output",
                         builder.getI32IntegerAttr(io_alias_as_array[1]));
  }
  for (const auto idx : call_op.getDonatedInputIndices()) {
    callee_op.setArgAttr(idx, "jax.buffer_donor", builder.getBoolAttr(true));
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

  // Construct a map from callee `SymbolRefAttr` to the unique `CallOps`
  // using it. This map is used to decide if an atom program module must be
  // cloned before populating its metadata (i.e., used more than once).
  llvm::DenseMap<mlir::SymbolRefAttr, llvm::DenseSet<CallOp, IfrtCallOpInfo>>
      callee_call_count;
  for (CallOp call_op : main_func.getOps<CallOp>()) {
    callee_call_count[call_op.getCallee()].insert(call_op);
  }

  llvm::DenseMap<CallOp, mlir::SymbolRefAttr, IfrtCallOpInfo> visited_call_ops;
  // Walk the CallOps in reverse order to ensure that the first CallOp using a
  // callee uses the original callee. Otherwise, the walk would modify the name
  // of the default callee.
  auto result = main_func.walk<mlir::WalkOrder::PreOrder,
                               mlir::ReverseIterator>([&](CallOp call_op)
                                                          -> mlir::WalkResult {
    mlir::func::FuncOp callee = call_op.getCalleeOp(symbol_table);
    if (callee == nullptr) {
      return call_op->emitOpError()
             << "can't find callee `" << call_op.getCalleeAttr() << "`";
    }
    auto callee_module = llvm::dyn_cast<mlir::ModuleOp>(callee->getParentOp());
    if (callee.getSymName() != kCalleeMainFuncName ||
        callee_module == nullptr) {
      return call_op.emitOpError()
             << "requires callee outlined as `" << kCalleeMainFuncName
             << "` function in a ModuleOp. Actual callee name: "
             << callee.getSymName()
             << ". Actual callee parent: " << callee->getParentOp()->getName();
    }

    if (auto call_op_it = visited_call_ops.find(call_op);
        call_op_it != visited_call_ops.end()) {
      call_op.setCalleeAttr(call_op_it->second);
    } else {
      callee_call_count[call_op.getCallee()].erase(call_op);
      if (!callee_call_count[call_op.getCallee()].empty()) {
        // Only clone the callee if it is used more than once.
        mlir::ModuleOp cloned_module = callee_module.clone();
        mlir::func::FuncOp cloned_callee = GetMainFunction(cloned_module);
        // Insert new cloned atom program module in the SymbolTable.
        symbol_table
            .getSymbolTable(
                callee_module->getParentWithTrait<mlir::OpTrait::SymbolTable>())
            .insert(cloned_module);
        mlir::SymbolRefAttr callee_attr = mlir::SymbolRefAttr::get(
            cloned_module.getSymNameAttr(),
            mlir::SymbolRefAttr::get(cloned_callee.getSymNameAttr()));
        auto populate_result =
            PopulateMetadata(call_op, cloned_module, cloned_callee, builder);
        if (mlir::failed(populate_result)) {
          return populate_result;
        }
        // Clone the CallOp because it will be modified next.
        visited_call_ops[call_op.clone()] = callee_attr;
        call_op.setCalleeAttr(callee_attr);
      } else {
        auto populate_result = PopulateMetadata(
            call_op, callee_module, GetMainFunction(callee_module), builder);
        if (mlir::failed(populate_result)) {
          return populate_result;
        }
        visited_call_ops[call_op.clone()] = call_op.getCalleeAttr();
      }
    }
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPopulateAtomProgramMetadataPass() {
  return std::make_unique<IfrtPopulateAtomProgramMetadataPass>();
}

}  // namespace ifrt
}  // namespace xla

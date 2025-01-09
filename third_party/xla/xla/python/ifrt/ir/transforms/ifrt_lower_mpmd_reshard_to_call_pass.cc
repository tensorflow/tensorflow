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
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/transforms/utils.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTLOWERMPMDRESHARDTOCALLPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Returns a fingerprint of the input and output types of a ReshardOp.
uint64_t ReshardFingerprint(ReshardOp reshard_op) {
  std::string s;
  llvm::raw_string_ostream os(s);
  for (const auto& input : reshard_op.getInputs()) {
    os << input.getType();
  }
  for (const auto& output : reshard_op.getOutputs()) {
    os << output.getType();
  }
  // Whether the input is donated does not need to be included in the
  // fingerprint  because that does not affect the computations generated.
  return tsl::Fingerprint64(os.str());
}

class IfrtLowerMpmdReshardToCallPass
    : public impl::IfrtLowerMpmdReshardToCallPassBase<
          IfrtLowerMpmdReshardToCallPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::SymbolTable symbol_table(module_op);
    mlir::OpBuilder builder(&getContext());
    mlir::func::FuncOp main_func = GetMainFunction(module_op);
    auto result = main_func->walk([&](ReshardOp reshard_op) {
      // Uniquify the devices the MPMD reshard executes on. It is unclear what
      // the devices order should be, but it is fine to return them sorted
      // because they are used only used for debugging purposes.
      absl::btree_set<int> device_set;
      bool does_reshard = false;
      for (const auto& [idx, pair] : llvm::enumerate(
               llvm::zip(reshard_op.getInputs(), reshard_op.getOutputs()))) {
        auto in_array_type =
            mlir::cast<IfrtArrayType>(std::get<0>(pair).getType());
        if (in_array_type == nullptr) {
          reshard_op.emitOpError()
              << "requires all inputs to be `IfrtArrayType`. Input #" << idx
              << ": " << std::get<0>(pair).getType();
          return mlir::WalkResult::interrupt();
        }
        auto out_array_type =
            mlir::cast<IfrtArrayType>(std::get<1>(pair).getType());
        if (out_array_type == nullptr) {
          reshard_op.emitOpError()
              << "requires all outputs to be `IfrtArrayType`. Output #" << idx
              << ": " << std::get<1>(pair).getType();
          return mlir::WalkResult::interrupt();
        }
        if (IsReshard(in_array_type, out_array_type)) {
          does_reshard = true;
        }
        device_set.insert(in_array_type.getDevices().begin(),
                          in_array_type.getDevices().end());
        device_set.insert(out_array_type.getDevices().begin(),
                          out_array_type.getDevices().end());
      }

      if (!does_reshard) {
        reshard_op.emitOpError()
            << "does not reshard any arrays. Use CopyArraysOp instead";
        return mlir::WalkResult::interrupt();
      }

      std::vector<int> devices(device_set.begin(), device_set.end());
      std::string module_sym_name =
          absl::StrCat("reshard_", ReshardFingerprint(reshard_op));

      auto reshard_module_op = mlir::dyn_cast_or_null<mlir::ModuleOp>(
          module_op.lookupSymbol(module_sym_name));
      mlir::func::FuncOp reshard_func = nullptr;
      if (reshard_module_op == nullptr) {
        // Create a module corresponding to the reshard op.
        builder.setInsertionPointToEnd(module_op.getBody());
        reshard_module_op = builder.create<mlir::ModuleOp>(
            mlir::UnknownLoc::get(builder.getContext()), module_sym_name);
        reshard_module_op.setVisibility(mlir::SymbolTable::Visibility::Private);
        reshard_module_op->setAttr(kIfrtNumDevicesAttrName,
                                   builder.getI32IntegerAttr(devices.size()));

        // Create the main func in the reshard module, and add the ReshardOp
        // to it.
        mlir::OpBuilder reshard_builder(reshard_module_op.getBodyRegion());
        reshard_func = reshard_builder.create<mlir::func::FuncOp>(
            reshard_module_op->getLoc(), kCalleeMainFuncName,
            mlir::FunctionType::get(reshard_builder.getContext(),
                                    reshard_op.getInputs().getTypes(),
                                    reshard_op.getOutputs().getTypes()));
        reshard_func.setVisibility(mlir::SymbolTable::Visibility::Public);
        reshard_func->setAttr(kIfrtReshardFunctionAttrName,
                              builder.getUnitAttr());
        mlir::Block* entryBlock = reshard_func.addEntryBlock();
        reshard_builder.setInsertionPointToEnd(entryBlock);
        auto inner_reshard_op = reshard_builder.create<ReshardOp>(
            reshard_op.getLoc(), /*outputs=*/reshard_op.getOutputs().getTypes(),
            /*control_output=*/reshard_op.getControlOutput().getType(),
            /*inputs=*/reshard_func.getArguments(),
            /*donated=*/reshard_op.getDonated(),
            /*control_inputs=*/mlir::ValueRange());
        reshard_builder.create<mlir::func::ReturnOp>(
            reshard_func.getLoc(), inner_reshard_op.getOutputs());
      }

      // Replace the ReshardOp with a CallOp.
      builder.setInsertionPoint(reshard_op);
      mlir::SymbolRefAttr reshard_func_symbol = mlir::SymbolRefAttr::get(
          reshard_module_op.getSymNameAttr(),
          mlir::SymbolRefAttr::get(GetMainFunction(reshard_module_op)));
      llvm::SmallVector<int32_t> donated_input_indices;
      if (reshard_op.getDonated()) {
        donated_input_indices.resize(reshard_op.getInputs().size());
        std::iota(donated_input_indices.begin(), donated_input_indices.end(),
                  0);
      }
      auto call_op = builder.create<CallOp>(
          reshard_op.getLoc(), /*outputs=*/reshard_op.getOutputs().getTypes(),
          /*control_output=*/reshard_op.getControlOutput().getType(),
          /*inputs=*/reshard_op.getInputs(),
          /*control_inputs=*/reshard_op.getControlInputs(),
          /*callee=*/reshard_func_symbol,
          /*devices=*/devices,
          /*io_aliases=*/builder.getArrayAttr({}),
          /*donated_input_indices=*/
          builder.getDenseI32ArrayAttr(donated_input_indices));
      call_op->setAttr(kIfrtModuleTypeAttrName,
                       builder.getStringAttr(kIfrtModuleTypeMpmdReshard));
      reshard_op.replaceAllUsesWith(call_op.getResults());
      reshard_op.erase();
      return mlir::WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtLowerMpmdReshardToCallPass() {
  return std::make_unique<IfrtLowerMpmdReshardToCallPass>();
}

}  // namespace ifrt
}  // namespace xla

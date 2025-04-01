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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTVERIFYDONATIONPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Verifies that if the value is an input to the IR, then it has been donated.
mlir::LogicalResult VerifyIfInputAndDonated(mlir::Operation* op, int idx,
                                            mlir::Value arg) {
  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(arg);
  mlir::func::FuncOp func_op = block_arg
                                   ? mlir::dyn_cast<mlir::func::FuncOp>(
                                         block_arg.getOwner()->getParentOp())
                                   : nullptr;
  if (func_op && func_op.getArgAttr(block_arg.getArgNumber(),
                                    kIfrtDonatedArgAttrName) == nullptr) {
    return op->emitOpError()
           << "input #" << idx << " has not been donated to the program.";
  }
  return mlir::success();
}

template <typename T>
mlir::LogicalResult verifyCallOpAliasesAndDonations(
    T op, llvm::DenseMap<mlir::Value, mlir::Operation*>& donated_value_to_op) {
  llvm::DenseSet<int> donated_input_idxs;
  // Verify if a donated input is an argument of the main func, then it has
  // also been donated by the user.
  for (const auto idx : op.getDonatedInputIndices()) {
    donated_input_idxs.insert(idx);
    auto donated_value = op.getInputs()[idx];
    auto donated_it = donated_value_to_op.try_emplace(donated_value, op);
    if (!donated_it.second) {
      op.emitOpError() << "input #" << idx << " of " << op.getCalleeAttr()
                       << " was already donated or aliased to the op at "
                       << donated_it.first->second->getLoc();
      return mlir::failure();
    }
    if (mlir::failed(VerifyIfInputAndDonated(op, idx, donated_value))) {
      return mlir::failure();
    }
  }

  for (const auto& io_alias :
       op.getIoAliases().template getAsRange<mlir::DenseI32ArrayAttr>()) {
    mlir::ArrayRef<int> io_alias_as_array = io_alias.asArrayRef();
    donated_input_idxs.insert(io_alias_as_array[0]);
    auto aliased_value = op.getInputs()[io_alias_as_array[0]];
    auto donated_it = donated_value_to_op.try_emplace(aliased_value, op);
    if (!donated_it.second) {
      op.emitOpError() << "input #" << io_alias_as_array[0] << " of "
                       << op.getCalleeAttr()
                       << " was already donated or aliased to the op at "
                       << donated_it.first->second->getLoc();
      return mlir::failure();
    }
    if (mlir::failed(
            VerifyIfInputAndDonated(op, io_alias_as_array[0], aliased_value))) {
      return mlir::failure();
    }
  }

  // Verify non-donated inputs after donated inputs have been
  // added to also catch instances such as
  // `ifrt.Call(%arg0 {ifrt.donated}, %arg0})`.
  for (const auto [idx, input] : llvm::enumerate(op.getInputs())) {
    if (!donated_input_idxs.contains(idx)) {
      auto donated_it = donated_value_to_op.find(input);
      if (donated_it != donated_value_to_op.end()) {
        op.emitOpError() << "input #" << idx << " of " << op.getCalleeAttr()
                         << " was already donated to the op at "
                         << donated_it->second->getLoc();
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}

template <typename T>
mlir::LogicalResult verifyCopyRemapAndReshardOpsDonation(
    T op, llvm::DenseMap<mlir::Value, mlir::Operation*>& donated_value_to_op) {
  // Verify that no inputs have already been donated.
  for (const auto [idx, input] : llvm::enumerate(op.getInputs())) {
    auto donated_it = donated_value_to_op.find(input);
    if (donated_it != donated_value_to_op.end()) {
      op.emitOpError() << "input #" << idx << " of op at " << op.getLoc()
                       << " was already donated to the op at "
                       << donated_it->second->getLoc();
      return mlir::failure();
    }
  }
  if (op.getDonated()) {
    // Add the donated inputs to the map and verify that all the
    // donated inputs are also donated to the main func.
    for (const auto [idx, input] : llvm::enumerate(op.getInputs())) {
      donated_value_to_op.try_emplace(input, op);
      if (mlir::failed(VerifyIfInputAndDonated(op, idx, input))) {
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}

// Verifies that no array is donated more than once, and that all arrays donated
// to reshard or atom programs, which are also inputs to the main func have
// `ifrt.donated` attribute set.
class IfrtVerifyDonationPass
    : public impl::IfrtVerifyDonationPassBase<IfrtVerifyDonationPass> {
 public:
  void runOnOperation() override;
};

void IfrtVerifyDonationPass::runOnOperation() {
  mlir::func::FuncOp func_op = getOperation();
  // We only need to run this pass on IFRT functions.
  if (!func_op->hasAttr(kIfrtFunctionAttrName) &&
      !func_op->hasAttr(kIfrtReshardFunctionAttrName)) {
    return;
  }
  llvm::DenseMap<mlir::Value, mlir::Operation*> donated_value_to_op;
  mlir::WalkResult result = func_op.walk([&](mlir::Operation* op)
                                             -> mlir::WalkResult {
    auto result =
        llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<CallOp, CallLoadedExecutableOp>([&](auto& op) {
              return verifyCallOpAliasesAndDonations(op, donated_value_to_op);
            })
            .Case<CopyArraysOp, RemapArraysOp, ReshardOp>([&](auto& op) {
              return verifyCopyRemapAndReshardOpsDonation(op,
                                                          donated_value_to_op);
            })
            .Case<mlir::func::ReturnOp>([&](mlir::func::ReturnOp return_op) {
              for (const auto& [idx, result] :
                   llvm::enumerate(return_op.getOperands())) {
                auto donated_it = donated_value_to_op.find(result);
                if (donated_it != donated_value_to_op.end()) {
                  return_op.emitOpError()
                      << "result #" << idx << " of op at " << return_op.getLoc()
                      << " was already donated to the op at "
                      << donated_it->second->getLoc();
                  return mlir::failure();
                }
              }
              return mlir::success();
            })
            .Default(mlir::success());

    if (mlir::failed(result)) {
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateIfrtVerifyDonationPass() {
  return std::make_unique<IfrtVerifyDonationPass>();
}

}  // namespace ifrt
}  // namespace xla

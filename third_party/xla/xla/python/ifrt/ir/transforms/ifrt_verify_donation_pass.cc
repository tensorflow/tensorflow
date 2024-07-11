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
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTVERIFYDONATIONPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Verifies that if the value is an input to the IR, then it has been donated.
mlir::LogicalResult VerifyIfInputAndDonated(mlir::Operation* op,
                                            mlir::Value arg) {
  auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(arg);
  mlir::func::FuncOp func_op = block_arg
                                   ? mlir::dyn_cast<mlir::func::FuncOp>(
                                         block_arg.getOwner()->getParentOp())
                                   : nullptr;
  if (func_op &&
      func_op.getArgAttr(block_arg.getArgNumber(),
                         xla::ifrt::kIfrtDonatedArgAttrName) == nullptr) {
    return op->emitOpError() << "input has not been donated to the program.";
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
  mlir::ModuleOp module_op = getOperation();
  xla::ifrt::ReshardOp reshard_op;
  llvm::DenseSet<mlir::Value> donated_values;
  mlir::WalkResult result = module_op.walk([&](mlir::Operation* op)
                                               -> mlir::WalkResult {
    auto result =
        llvm::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
            .Case<xla::ifrt::CallOp, xla::ifrt::CallLoadedExecutableOp>(
                [&](auto& op) {
                  for (const auto& io_alias :
                       op.getIoAliases()
                           .template getAsRange<mlir::DenseI32ArrayAttr>()) {
                    mlir::ArrayRef<int> io_alias_as_array =
                        io_alias.asArrayRef();
                    auto donated_value = op.getInputs()[io_alias_as_array[0]];
                    if (!donated_values.insert(donated_value).second) {
                      op.emitOpError() << "input #" << io_alias_as_array[0]
                                       << " already donated.";
                      return mlir::failure();
                    }

                    if (mlir::failed(
                            VerifyIfInputAndDonated(op, donated_value))) {
                      return mlir::failure();
                    }
                  }
                  return mlir::success();
                })
            .Case<xla::ifrt::ReshardOp>([&](auto& op) {
              if (op.getDonated()) {
                auto donated_value = op.getInput();
                if (!donated_values.insert(donated_value).second) {
                  op.emitOpError() << "input already donated.";
                  return mlir::failure();
                }
                if (mlir::failed(VerifyIfInputAndDonated(op, donated_value))) {
                  return mlir::failure();
                }
              }
              return mlir::success();
            })
            .Case<xla::ifrt::RemapArraysOp>([&](auto& op) {
              if (op.getDonated()) {
                for (const auto [idx, input] :
                     llvm::enumerate(op.getInputs())) {
                  if (!donated_values.insert(input).second) {
                    op.emitOpError() << "input #" << idx << " already donated.";
                    return mlir::failure();
                  }
                  if (mlir::failed(VerifyIfInputAndDonated(op, input))) {
                    return mlir::failure();
                  }
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyDonationPass() {
  return std::make_unique<IfrtVerifyDonationPass>();
}

}  // namespace ifrt
}  // namespace xla

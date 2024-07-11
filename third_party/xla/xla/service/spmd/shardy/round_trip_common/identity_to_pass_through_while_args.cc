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

#include "xla/service/spmd/shardy/round_trip_common/identity_to_pass_through_while_args.h"

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "shardy/dialect/sdy/ir/utils.h"  // from @shardy
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::StringRef;

using ::mlir::func::FuncOp;

// For every block argument of an `mhlo::WhileOp` that is directly returned by
// the body of the op (pass-through), add an `sdy::IdentityOp` between the block
// argument and the return op.
//
// This will prevent canonicalization from replacing these block arguments with
// the corresponding operands as free variables.
class AddIdentityToPassThroughWhileArgsPass
    : public mlir::PassWrapper<AddIdentityToPassThroughWhileArgsPass,
                               mlir::OperationPass<FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AddIdentityToPassThroughWhileArgsPass)

  void runOnOperation() final {
    FuncOp funcOp = getOperation();
    mlir::IRRewriter rewriter(funcOp);

    funcOp.walk([&](mlir::mhlo::WhileOp op) {
      mlir::Operation* returnOp = mlir::sdy::getBodyTerminator(op);
      rewriter.setInsertionPoint(returnOp);
      for (mlir::Value returnValue : returnOp->getOperands()) {
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(returnValue);
            blockArg && blockArg.getOwner() == &op.getBody().front()) {
          auto identityOp = rewriter.create<mlir::sdy::IdentityOp>(
              returnValue.getLoc(), returnValue);
          rewriter.replaceUsesWithIf(returnValue, identityOp,
                                     [returnOp](mlir::OpOperand& use) {
                                       return use.getOwner() == returnOp;
                                     });
        }
      }
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-add-identity-to-pass-through-while-args";
  }

  StringRef getDescription() const override {
    return "Adds an identity op between pass-through block arguments of a "
           "while op.";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createAddIdentityToPassThroughWhileArgsPass() {
  return std::make_unique<AddIdentityToPassThroughWhileArgsPass>();
}

void registerAddIdentityToPassThroughWhileArgsPass() {
  mlir::registerPass(createAddIdentityToPassThroughWhileArgsPass);
}

}  // namespace sdy
}  // namespace xla

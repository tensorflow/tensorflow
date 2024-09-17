/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/Support/Casting.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
#define GEN_PASS_DEF_LIFTCALLSITELOCCALLERPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

namespace {

// JAX bridge generates a func.call for each op lowering
// These are inlined but loc will be messed up after the inline pass. This pass
// normalize the loc after inline pass.

class LiftCallSiteLocCallerPass
    : public impl::LiftCallSiteLocCallerPassBase<LiftCallSiteLocCallerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCallSiteLocCallerPass);

  void runOnOperation() override {
    getOperation()->walk([](mlir::Operation* op) {
      while (true) {
        auto loc = mlir::dyn_cast_or_null<CallSiteLoc>(op->getLoc());
        if (loc == nullptr) {
          return;
        }

        if (llvm::isa<mlir::UnknownLoc>(loc.getCallee())) {
          op->setLoc(loc.getCaller());
        } else {
          op->setLoc(loc.getCallee());
        }
      }
    });
  }
};

}  // namespace
}  // namespace odml
}  // namespace mlir

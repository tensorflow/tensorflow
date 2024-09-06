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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
    getOperation()->walk([](func::FuncOp func_op) {
      for (Operation& op : func_op.getOps()) {
        if (!mlir::isa<CallSiteLoc>(op.getLoc())) {
          continue;
        }

        auto loc = op.getLoc().dyn_cast<CallSiteLoc>();
        op.setLoc(loc.getCaller());
      }
    });
  }
};

}  // namespace
}  // namespace odml
}  // namespace mlir

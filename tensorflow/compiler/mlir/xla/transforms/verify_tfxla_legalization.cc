/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project

namespace mlir {
namespace mhlo {

namespace {

#define GEN_PASS_DEF_VERIFYTFXLALEGALIZATION
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

class VerifyTFXLALegalization
    : public impl::VerifyTFXLALegalizationBase<VerifyTFXLALegalization> {
 public:
  void runOnOperation() override;
};

void VerifyTFXLALegalization::runOnOperation() {}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyTFXLALegalizationPass() {
  return std::make_unique<VerifyTFXLALegalization>();
}

}  // namespace mhlo
}  // namespace mlir

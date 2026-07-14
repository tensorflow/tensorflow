/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/op_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

#define GEN_PASS_DEF_DTENSORREMOVEDTENSORLAYOUTPASS
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

class DTensorRemoveDTensorLayoutPass
    : public impl::DTensorRemoveDTensorLayoutPassBase<
          DTensorRemoveDTensorLayoutPass> {
 public:
  void runOnOperation() override {
    RemoveDTensorLayoutOps(getOperation(), /*remove_xla_spmd_layouts=*/true);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorRemoveDTensorLayoutPass() {
  return std::make_unique<DTensorRemoveDTensorLayoutPass>();
}

}  // namespace dtensor
}  // namespace tensorflow

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/set_tpu_infeed_layout.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_api.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/type_to_shape.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_INFEEDOPSXLAADJUSTLAYOUT
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes.h.inc"

class InfeedsOpsXlaAdjustLayout
    : public impl::InfeedOpsXlaAdjustLayoutBase<InfeedsOpsXlaAdjustLayout> {
 public:
  void runOnOperation() override;

 private:
  static void runOnInfeedOp(::mlir::mhlo::InfeedOp op) {
    OpBuilder builder(op.getContext());
    SmallVector<Type> result_types(op.getResultTypes().begin(),
                                   op.getResultTypes().end());
    if (!op->getAttr("layout")) {
      auto layout = mlir::GetTPUInfeedLayout(result_types, builder);
      if (failed(layout)) return;

      op->setAttr("layout", layout.value());
    }
  }
};

void InfeedsOpsXlaAdjustLayout::runOnOperation() {
  getOperation().walk(runOnInfeedOp);
}

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
CreateInfeedsOpsXlaAdjustLayoutPass() {
  return std::make_unique<InfeedsOpsXlaAdjustLayout>();
}

}  // namespace mhlo
}  // namespace mlir

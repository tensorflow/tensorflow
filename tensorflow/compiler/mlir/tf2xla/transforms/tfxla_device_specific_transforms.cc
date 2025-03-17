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
#include <optional>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/tf2xla/kernels/rng_converter_utils.h"
#include "xla/xla_data.pb.h"

namespace mlir {
namespace mhlo {

namespace {

#define GEN_PASS_DEF_TFXLADEVICESPECIFICTRANSFORMS
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

class TFXLADeviceSpecificTransforms
    : public impl::TFXLADeviceSpecificTransformsBase<
          TFXLADeviceSpecificTransforms> {
 public:
  explicit TFXLADeviceSpecificTransforms(std::optional<StringRef> device_type) {
    if (device_type.has_value()) {
      device_type_ = device_type.value().str();
    }
  }
  void runOnOperation() override;

 private:
  LogicalResult ConvertGetAlgOp(TF::StatelessRandomGetAlgOp get_alg_op);
};

LogicalResult TFXLADeviceSpecificTransforms::ConvertGetAlgOp(
    TF::StatelessRandomGetAlgOp get_alg_op) {
  if (!device_type_.hasValue()) return failure();

  xla::RandomAlgorithm xla_rng =
      tensorflow::DefaultRngAlgForDeviceType(device_type_);
  tensorflow::Algorithm tensorflow_rng =
      tensorflow::ToTensorflowAlgorithm(xla_rng);

  OpBuilder opbuilder(get_alg_op);

  auto tf_const = opbuilder.create<TF::ConstOp>(
      get_alg_op->getLoc(), opbuilder.getI32IntegerAttr((int)tensorflow_rng));

  get_alg_op->replaceAllUsesWith(tf_const);
  get_alg_op->erase();
  return success();
}

void TFXLADeviceSpecificTransforms::runOnOperation() {
  if (!device_type_.hasValue()) return;
  auto func_op = getOperation();

  auto walk_result = func_op->walk([&](TF::StatelessRandomGetAlgOp op) {
    if (failed(ConvertGetAlgOp(op))) {
      op->emitOpError(
          "Could not convert and remove Device specific information");
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  if (walk_result.wasInterrupted()) signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateTFXLADeviceSpecificTransformsPass(std::optional<StringRef> device_type) {
  return std::make_unique<TFXLADeviceSpecificTransforms>(device_type);
}

}  // namespace mhlo
}  // namespace mlir

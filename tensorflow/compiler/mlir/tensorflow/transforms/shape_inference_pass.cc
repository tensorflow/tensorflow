/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/shape_inference.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_TENSORFLOWSHAPEINFERENCEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// This transformation pass propagate shapes on the TensorFlow graph.
// It is a ModulePass in order to be able to change function types.
class ShapeInference
    : public impl::TensorFlowShapeInferencePassBase<ShapeInference> {
 public:
  ShapeInference() = default;
  explicit ShapeInference(ArrayRef<ArrayRef<int64_t>> input_shapes)
      : input_shapes_(input_shapes) {}
  void runOnOperation() override {
    // Parse `input_arg_shapes_` if provided (test only)
    SmallVector<ArrayRef<int64_t>> input_shapes_vec;
    absl::StatusOr<SmallVector<SmallVector<int64_t>>> parsed_shapes;
    if (!input_arg_shapes_.empty()) {
      parsed_shapes = ParseArgumentShapes(input_arg_shapes_);
      if (!parsed_shapes.ok()) {
        getOperation().emitError() << parsed_shapes.status().message();
        return signalPassFailure();
      }
      input_shapes_vec = SmallVector<ArrayRef<int64_t>>{parsed_shapes->begin(),
                                                        parsed_shapes->end()};
      input_shapes_ = input_shapes_vec;
    }

    auto failure_or_converged = InferModuleShape(
        getOperation(), max_iterations_, /*ops_to_skip=*/{}, input_shapes_);
    if (failed(failure_or_converged)) return signalPassFailure();
    if (!failure_or_converged.value()) {
      getOperation().emitError()
          << "shape inference pass did not reach convergence after "
          << max_iterations_;
      return signalPassFailure();
    }
  }

 private:
  ArrayRef<ArrayRef<int64_t>> input_shapes_;
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTFShapeInferencePass(
    ArrayRef<ArrayRef<int64_t>> input_shapes) {
  return std::make_unique<ShapeInference>(input_shapes);
}

}  // namespace TF
}  // namespace mlir

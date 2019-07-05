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

// This transformation pass applies quantization propagation on TFLite dialect.

#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/quantization_utils.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {
// Applies prepare quantization on the model in TFL dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantizaton rule for some operations in the quantization-awre
// training quantization simpler.
struct PrepareQuantizePass : public FunctionPass<PrepareQuantizePass> {
  void runOnFunction() override;
};

void PrepareQuantizePass::runOnFunction() {
  ApplyQuantizationParamsPropagation(getFunction());
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
FunctionPassBase *CreatePrepareQuantizePass() {
  return new PrepareQuantizePass();
}

static PassRegistration<PrepareQuantizePass> pass(
    "tfl-prepare-quantize", "Prepare TFL dialect for quantization");

}  // namespace TFL
}  // namespace mlir

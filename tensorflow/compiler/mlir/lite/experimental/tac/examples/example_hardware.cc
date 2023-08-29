/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/examples/example_hardware.h"

#include <memory>

#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_patterns.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {

constexpr char ExampleHardware::kId[];  // Define kId.

mlir::RewritePatternSet ExampleHardware::GetTransformations(
    MLIRContext* context) const {
  mlir::RewritePatternSet patterns(context);

  patterns.add<LowerPackIntoConcatReshape, UnrollSplit, UnrollSplitV, PadSlice,
               PadConcat>(context);
  return patterns;
}

std::unique_ptr<TargetHardware> CreateExampleHardware() {
  return std::make_unique<ExampleHardware>();
}

TargetHardwareRegistration<ExampleHardware> example_hardware(
    "Example device", CreateExampleHardware);

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

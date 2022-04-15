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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_EXAMPLES_EXAMPLE_HARDWARE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_EXAMPLES_EXAMPLE_HARDWARE_H_

#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/simple_hardware.h"

namespace mlir {
namespace TFL {
namespace tac {

class ExampleHardware : public SimpleHardware {
 public:
  static constexpr char kId[] = "ExampleHardware";

  mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const override;

  mlir::TypeID GetTypeId() const override {
    return mlir::TypeID::get<ExampleHardware>();
  }

  bool IsNotSupportedOp(mlir::Operation* op) const override { return false; }

  float AdvantageOverCPU() const override { return 5.0; }
};

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_EXAMPLES_EXAMPLE_HARDWARE_H_

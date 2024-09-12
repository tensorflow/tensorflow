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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_GPU_HARDWARE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_GPU_HARDWARE_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
// Gpu hardware class which handles GPU capabilities in TFLite.
// This is used by TAC to get op supported/ op cost estimates on GPU.
class GpuHardware : public TargetHardware {
 public:
  static constexpr char kId[] = "GPU";
  mlir::RewritePatternSet GetTransformations(
      MLIRContext* context) const override;

  mlir::TypeID GetTypeId() const override {
    return mlir::TypeID::get<GpuHardware>();
  }

  double GetHardwareSwitchingCost(const TargetHardware* from,
                                  size_t buffer_size) const override;
};
}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_GPU_HARDWARE_H_

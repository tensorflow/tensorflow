/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_XLA_CPU_DEVICE_TARGET_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_XLA_CPU_DEVICE_TARGET_H_

#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/device_target.h"

namespace mlir {
namespace xla_hlo {

// Target specs for cpu kernels
class CpuDeviceTarget : public quant::DeviceTarget {
 public:
  explicit CpuDeviceTarget(MLIRContext* ctx);

 private:
  LogicalResult HandleMultiplyAccumulateScale(
      quant::QuantizeContext* ctx, Operation* op,
      quant::AdjacentOperations* new_items, bool* changed);
};

}  // namespace xla_hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_XLA_CPU_DEVICE_TARGET_H_

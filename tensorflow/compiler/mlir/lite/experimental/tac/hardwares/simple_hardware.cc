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

#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/simple_hardware.h"

#include <cstddef>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"

namespace mlir {
namespace TFL {
namespace tac {

bool SimpleHardware::IsOpSupported(mlir::Operation* op) const {
  if (IsNotSupportedOp(op)) {
    return false;
  }
  const TargetHardware* cpu = GetTargetHardware("CPU");
  return cpu->IsOpSupported(op);
}

double SimpleHardware::GetHardwareSwitchingCost(const TargetHardware* from,
                                                size_t buffer_size) const {
  auto from_type = from->GetTypeId();
  auto to_type = GetTypeId();
  if (from_type == to_type) return 0.0f;

  // TODO(renjieliu): Implement a better version for different hardware cases.
  return buffer_size * kCrossHardwareTransferPerByteCost / 8.0 +
         kCrossHardwareTransferFixedCost;
}

double SimpleHardware::GetOpCost(mlir::Operation* op) const {
  const TargetHardware* cpu = GetTargetHardware("CPU");
  return cpu->GetOpCost(op) / AdvantageOverCPU();
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_SIMPLE_HARDWARE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_SIMPLE_HARDWARE_H_

#include <cstddef>

#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"

namespace mlir {
namespace TFL {
namespace tac {

// A simple hardware is an interface makes you add a target backend easily if
// you don't want too much customization.
//
// It allows you to easily specify the ops capabilities (by
// specifying the denylist), the rest ops will be considered supported. Also you
// can also specify the advantage over CPU.
//
// If you need more customization, e.g., if you have your own hardware dialect,
// please consider use TargetHardware directly.
class SimpleHardware : public TargetHardware {
 public:
  // This is essentially a denylist.
  // TODO(renjieliu): Consider whether we want an allowlist for custom op as
  // well.
  virtual bool IsNotSupportedOp(mlir::Operation* op) const = 0;

  // The larger the value is, the more preferrable over CPU.
  // If the value > 1, means the hardware has advantage over CPU.
  // If the value < 1, means CPU is more preferred.
  // If we specify 10.0, meaning the hardware is 10x faster than CPU.
  // The value should be > 0.
  // TODO(renjieliu): Consider add an interface for more detailed customization,
  // for example, users should be able to specify some ops are preferred and
  // some are less preferred.
  virtual float AdvantageOverCPU() const = 0;

 private:
  bool IsOpSupported(mlir::Operation* op) const override;

  double GetHardwareSwitchingCost(const TargetHardware* from,
                                  size_t buffer_size) const override;

  double GetOpCost(mlir::Operation* op) const override;
};

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_HARDWARES_SIMPLE_HARDWARE_H_

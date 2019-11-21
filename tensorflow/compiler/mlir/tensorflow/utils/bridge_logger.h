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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_BRIDGE_LOGGER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_BRIDGE_LOGGER_H_

#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassInstrumentation.h"  // TF:local_config_mlir

namespace tensorflow {

// Logger for logging/dumping MLIR modules before and after passes in bridge
// targeting TPUs.
class BridgeLogger : public mlir::PassInstrumentation {
 public:
  void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override;
  void runAfterPass(mlir::Pass* pass, mlir::Operation* op) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_BRIDGE_LOGGER_H_

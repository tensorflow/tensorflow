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

#ifndef TENSORFLOW_CORE_IR_INTERFACES_H_
#define TENSORFLOW_CORE_IR_INTERFACES_H_

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectInterface.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"

// Include generated declarations.
#include "tensorflow/core/ir/interfaces.h.inc"

namespace mlir {
namespace tfg {
// The dialect fallback model for the TensorFlow registry interface.
class TensorFlowRegistryInterfaceBase
    : public TensorFlowRegistryInterface::FallbackModel<
          TensorFlowRegistryInterfaceBase>,
      public DialectInterface::Base<TensorFlowRegistryInterfaceBase> {
 public:
  explicit TensorFlowRegistryInterfaceBase(Dialect *dialect)
      : DialectInterface::Base<TensorFlowRegistryInterfaceBase>(dialect) {}

  // Returns whether the operation is stateful.
  virtual bool isStateful(Operation *op) const = 0;
};
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_INTERFACES_H_

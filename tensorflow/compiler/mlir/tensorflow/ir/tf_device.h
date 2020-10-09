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

// This file defines the tf_device dialect: it contains operations that model
// TensorFlow's actions to launch computations on accelerator devices.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DEVICE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DEVICE_H_

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace mlir {
namespace tf_device {

// The TensorFlow Device dialect.
//
// This dialect contains operations to describe/launch computations on devices.
// These operations do not map 1-1 to TensorFlow ops and requires a lowering
// pass later to transform them into Compile/Run op pairs, like XlaCompile and
// XlaRun.
class TensorFlowDeviceDialect : public Dialect {
 public:
  static StringRef getDialectNamespace() { return "tf_device"; }
  // Constructing TensorFlowDevice dialect under an non-null MLIRContext.
  explicit TensorFlowDeviceDialect(MLIRContext* context);
};

}  // namespace tf_device
}  // namespace mlir

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DEVICE_H_

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_DIALECT_REGISTRATION_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_DIALECT_REGISTRATION_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/core/ir/ops.h"

namespace mlir {
// Inserts all the TensorFlow dialects in the provided registry. This is
// intended for tools that need to register dialects before parsing .mlir files.
inline void RegisterAllTensorFlowDialects(DialectRegistry &registry) {
  registry
      .insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
              mlir::ml_program::MLProgramDialect, mlir::TF::TensorFlowDialect,
              mlir::tf_type::TFTypeDialect, mlir::cf::ControlFlowDialect,
              mlir::tf_device::TensorFlowDeviceDialect,
              mlir::tf_executor::TensorFlowExecutorDialect,
              mlir::tf_saved_model::TensorFlowSavedModelDialect,
              mlir::tfg::TFGraphDialect>();
}
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_DIALECT_REGISTRATION_H_

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

#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"

namespace mlir {
// Inserts all the TensorFlow dialects in the provided registry. This is
// intended for tools that need to register dialects before parsing .mlir files.
inline void RegisterAllTensorFlowDialects(DialectRegistry &registry) {
  registry.insert<mlir::StandardOpsDialect, mlir::TF::TensorFlowDialect,
                  mlir::tf_type::TFTypeDialect, mlir::complex::ComplexDialect,
                  mlir::tf_device::TensorFlowDeviceDialect,
                  mlir::tf_executor::TensorFlowExecutorDialect,
                  mlir::tf_saved_model::TensorFlowSavedModelDialect>();
}
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_DIALECT_REGISTRATION_H_

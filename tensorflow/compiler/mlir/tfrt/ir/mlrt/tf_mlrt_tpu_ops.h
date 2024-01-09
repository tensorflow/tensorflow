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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_TPU_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_TPU_OPS_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

namespace tensorflow {
namespace tf_mlrt_tpu {

class TensorflowMlrtTpuDialect : public mlir::Dialect {
 public:
  explicit TensorflowMlrtTpuDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "tf_mlrt_tpu"; }
};

}  // namespace tf_mlrt_tpu
}  // namespace tensorflow

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_tpu_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_IR_MLRT_TF_MLRT_TPU_OPS_H_

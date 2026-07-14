/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.cc.inc"

namespace mlir {
namespace quant {

void RegisterOpsHook(TF::TensorFlowDialect &dialect) {
  dialect.addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.cc.inc"
      >();
}

void RegisterOps() {
  TF_DIALECT_REGISTER_ADDITIONAL_OPERATIONS(RegisterOpsHook);
}

static auto kRegistration = (RegisterOps(), true);

}  // namespace quant
}  // namespace mlir

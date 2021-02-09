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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_XLA_THUNKS_OPS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_XLA_THUNKS_OPS_H_

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project

namespace mlir {
class OpBuilder;

#include "tensorflow/compiler/xla/service/gpu/ir/xla_thunks_structs.h.inc"

namespace xla_thunks {

class XLAThunksDialect : public Dialect {
 public:
  explicit XLAThunksDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "xla_thunks"; }
};

}  // namespace xla_thunks

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/service/gpu/ir/xla_thunks_ops.h.inc"

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_IR_XLA_THUNKS_OPS_H_

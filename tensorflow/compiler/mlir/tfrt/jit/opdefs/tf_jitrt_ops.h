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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_OPDEFS_TF_JITRT_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_OPDEFS_TF_JITRT_OPS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tfrt/compiler/opdefs/tfrt_op_interfaces.h"  // from @tf_runtime
#include "tfrt/compiler/opdefs/tfrt_traits.h"  // from @tf_runtime

namespace mlir {
namespace tf_jitrt {

class JitRuntimeDialect : public mlir::Dialect {
 public:
  explicit JitRuntimeDialect(mlir::MLIRContext *context);
  static mlir::StringRef getDialectNamespace() { return "tf_jitrt"; }
};

// Returns the maximum size of ranked tensor argument of the `func`. Returns `1`
// if all arguments are unranked.
//
// To get the cost of the `tf_jitrt.execute` operations, when compiled cluster
// has unranked inputs, we use the maximum size of the arguments of the parent
// function as an estimate (see TFRT_CostFunctionInterface).
int64_t GetMaxArgSize(mlir::func::FuncOp func);

}  // namespace tf_jitrt
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/tf_jitrt_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_OPDEFS_TF_JITRT_OPS_H_

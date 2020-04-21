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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_IR_CHLO_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_IR_CHLO_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffects.h"  // from @llvm-project

namespace mlir {
namespace xla_chlo {

class XlaHloClientDialect : public Dialect {
 public:
  explicit XlaHloClientDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "xla_chlo"; }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/chlo_ops.h.inc"

}  // namespace xla_chlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_CHLO_OPS_H_

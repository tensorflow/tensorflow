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

// This file defines the operations used in the LXLA dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_IR_LHLO_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_IR_LHLO_OPS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Dialect.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir

namespace mlir {
class OpBuilder;

namespace xla_lhlo {

class XlaLhloDialect : public Dialect {
 public:
  explicit XlaLhloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "xla_lhlo"; }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h.inc"

}  // namespace xla_lhlo
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_IR_LHLO_OPS_H_

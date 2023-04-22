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

// This file defines the operations used in the MLIR TensorFlow Lite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_

#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_structs.h.inc"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mlir {
namespace TFL {

class TensorFlowLiteDialect : public Dialect {
 public:
  explicit TensorFlowLiteDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "tfl"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_interface.h.inc"

}  // end namespace TFL
}  // end namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_

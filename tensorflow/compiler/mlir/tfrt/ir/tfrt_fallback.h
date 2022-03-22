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
#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

using namespace mlir;  // NOLINT

namespace tfrt {
namespace fallback {

// Dialect for fallback operations.
class FallbackDialect : public Dialect {
 public:
  explicit FallbackDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfrt_fallback"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &os) const override;
};

// The MLIR type represents a tensorflow::Tensor.
class TFTensorType : public Type::TypeBase<TFTensorType, Type, TypeStorage> {
 public:
  using Base::Base;
};

// The MLIR type represents a tensorflow::Allocator.
class TFAllocatorType
    : public Type::TypeBase<TFAllocatorType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace fallback
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_IR_TFRT_FALLBACK_H_

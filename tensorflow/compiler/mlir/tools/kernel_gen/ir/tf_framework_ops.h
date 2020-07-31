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

// This file defines the operations used in the TFFramework dialect.
//
#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

namespace TFFrameworkTypes {
enum Kind {
  // TODO(pifon): Replace enum value with
  // OpKernelContextType = Type::FIRST_TF_FRAMEWORK_TYPE,
  // after DialectSymbolRegistry.def is updated.
  OpKernelContextType = Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
};
}  // namespace TFFrameworkTypes

/// OpKernelContextType corresponds to C++ class OpKernelContext defined in
/// tensorflow/core/framework/op_kernel.h
class OpKernelContextType
    : public Type::TypeBase<OpKernelContextType, Type, TypeStorage> {
 public:
  using Base::Base;

  static OpKernelContextType get(MLIRContext *context) {
    return Base::get(context, TFFrameworkTypes::Kind::OpKernelContextType);
  }

  /// Support method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) {
    return kind == TFFrameworkTypes::Kind::OpKernelContextType;
  }
};

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_dialect.h.inc"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h.inc"

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_

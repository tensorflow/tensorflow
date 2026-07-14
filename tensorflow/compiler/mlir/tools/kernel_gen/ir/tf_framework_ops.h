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

#include "absl/status/status.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/ControlFlowInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_status.h.inc"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

/// OpKernelContextType corresponds to C++ class OpKernelContext defined in
/// tensorflow/core/framework/op_kernel.h
class OpKernelContextType
    : public Type::TypeBase<OpKernelContextType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name =
      "kernel_gen.tf_framework.op_kernel_context";
};

class JITCallableType
    : public Type::TypeBase<JITCallableType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name = "kernel_gen.tf_framework.jit_callable";
};

absl::StatusCode ConvertAttrToEnumValue(ErrorCode error_code);

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_dialect.h.inc"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_IR_TF_FRAMEWORK_OPS_H_

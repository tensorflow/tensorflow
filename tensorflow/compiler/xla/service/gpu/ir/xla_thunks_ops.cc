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

// This file defines the operations used in the Thunk dialect.

#include "tensorflow/compiler/xla/service/gpu/ir/xla_thunks_ops.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
#include "tensorflow/compiler/xla/service/gpu/ir/xla_thunks_structs.cc.inc"
namespace xla_thunks {

XLAThunksDialect::XLAThunksDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<XLAThunksDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/xla/service/gpu/ir/xla_thunks_ops.cc.inc"
      >();
}

}  // namespace xla_thunks

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/service/gpu/ir/xla_thunks_ops.cc.inc"

}  // namespace mlir

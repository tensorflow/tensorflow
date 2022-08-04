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

#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project

//===----------------------------------------------------------------------===//
// RT Dialect
//===----------------------------------------------------------------------===//

#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_dialect.cpp.inc"

namespace xla {
namespace runtime {

void RuntimeDialect::initialize() {
  allowUnknownTypes();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/xla/mlir/ir/runtime//rt_ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_types.cpp.inc"
      >();
}

}  // namespace runtime
}  // end namespace xla

#define GET_OP_CLASSES
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_ops.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/xla/mlir/ir/runtime/rt_types.cpp.inc"

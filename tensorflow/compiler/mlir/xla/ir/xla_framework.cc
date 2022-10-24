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

// This file defines the operations used in the xla_framework dialect.
#include "tensorflow/compiler/mlir/xla/ir/xla_framework.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project

// Generated dialect definitions.
#include "tensorflow/compiler/mlir/xla/ir/xla_framework_dialect.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/xla_framework_types.cc.inc"

namespace mlir {
namespace xla_framework {

// Setup operations and types
void XLAFrameworkDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/xla_framework.cc.inc"
#undef GET_OP_LIST
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "tensorflow/compiler/mlir/xla/ir/xla_framework_types.cc.inc"
#undef GET_TYPEDEF_LIST
      >();
}

}  // namespace xla_framework
}  // namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/xla_framework.cc.inc"

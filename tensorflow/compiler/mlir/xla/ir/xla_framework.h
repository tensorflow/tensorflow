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

// This file defines the operations and types used in the XLAFramework dialect.
//
#ifndef TENSORFLOW_COMPILER_MLIR_XLA_XLA_FRAMEWORK_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_XLA_FRAMEWORK_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project

#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/xla_framework_types.h.inc"
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/xla_framework.h.inc"
#include "tensorflow/compiler/mlir/xla/ir/xla_framework_dialect.h.inc"

#undef GET_OP_CLASSES

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_XLA_FRAMEWORK_H_

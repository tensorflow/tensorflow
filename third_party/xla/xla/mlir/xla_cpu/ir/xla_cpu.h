/* Copyright 2022 The OpenXLA Authors.

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
#ifndef XLA_MLIR_XLA_CPU_IR_XLA_CPU_H_
#define XLA_MLIR_XLA_CPU_IR_XLA_CPU_H_

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project

#define GET_OP_CLASSES
#include "xla/mlir/xla_cpu/ir/xla_cpu.h.inc"
#include "xla/mlir/xla_cpu/ir/xla_cpu_dialect.h.inc"
#include "xla/mlir/xla_cpu/ir/xla_cpu_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/mlir/xla_cpu/ir/xla_cpu_attrdefs.h.inc"
#undef GET_OP_CLASSES

#endif  // XLA_MLIR_XLA_CPU_IR_XLA_CPU_H_

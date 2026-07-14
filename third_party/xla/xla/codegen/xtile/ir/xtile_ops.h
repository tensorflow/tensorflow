/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_XTILE_IR_XTILE_OPS_H_
#define XLA_CODEGEN_XTILE_IR_XTILE_OPS_H_

#include <optional>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/Interfaces/CallInterfaces.h"  // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h"  // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_attrs.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_dialect.h"  // IWYU pragma: keep
#include "xla/hlo/analysis/indexing_map.h"  // IWYU pragma: keep

#define GET_OP_CLASSES
#include "xla/codegen/xtile/ir/xtile_interface_ops.h.inc"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h.inc"  // IWYU pragma: keep

#endif  // XLA_CODEGEN_XTILE_IR_XTILE_OPS_H_

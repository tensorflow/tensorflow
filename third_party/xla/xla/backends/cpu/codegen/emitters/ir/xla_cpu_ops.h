/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_IR_XLA_CPU_OPS_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_IR_XLA_CPU_OPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"  // IWYU pragma: keep
#include "mlir/IR/Attributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"  // IWYU pragma: keep
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_types.h"  // IWYU pragma: keep

#define GET_OP_CLASSES
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.h.inc"

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_IR_XLA_CPU_OPS_H_

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

#ifndef XLA_MLIR_RUNTIME_IR_TESTS_TESTLIB_H_
#define XLA_MLIR_RUNTIME_IR_TESTS_TESTLIB_H_

#include <cstdint>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project

// clang-format off
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "xla/mlir/runtime/ir/tests/testlib_dialect.h.inc"
#include "xla/mlir/runtime/ir/tests/testlib_enums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "xla/mlir/runtime/ir/tests/testlib_attrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "xla/mlir/runtime/ir/tests/testlib_types.h.inc"

namespace xla {
namespace runtime {

inline mlir::Type ConvertValueType(ValueType type) {
  return mlir::LLVM::LLVMPointerType::get(type.getContext());
}

inline void AddTestlibTypeConversions(mlir::TypeConverter& converter) {
  converter.addConversion(ConvertValueType);
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_MLIR_RUNTIME_IR_TESTS_TESTLIB_H_

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_SORT_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_SORT_UTIL_H_

#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace llvm_ir {
// Emits llvm IR to sort the 'dimension_to_sort' dimension of 'keys_array' into
// ascending order.
Status EmitSortInPlace(int64 dimension_to_sort, const IrArray& keys_array,
                       tensorflow::StringPiece name,
                       llvm::IRBuilder<>* ir_builder);
}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_SORT_UTIL_H_

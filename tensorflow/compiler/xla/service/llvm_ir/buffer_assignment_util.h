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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_BUFFER_ASSIGNMENT_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_BUFFER_ASSIGNMENT_UTIL_H_

#include "tensorflow/compiler/xla/service/buffer_assignment.h"

namespace xla {
namespace llvm_ir {
// In XLA:GPU we map constant buffer allocations to globals in the generated
// LLVM IR.  This function gives us the name of the global variable a constant
// buffer is mapped to.  Not used on XLA:CPU.
string ConstantBufferAllocationToGlobalName(const BufferAllocation& allocation);

// Returns the Literal corresponding to `allocation`, which must be a constant
// allocation.
const Literal& LiteralForConstantAllocation(const BufferAllocation& allocation);
}  // namespace llvm_ir
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_BUFFER_ASSIGNMENT_UTIL_H_

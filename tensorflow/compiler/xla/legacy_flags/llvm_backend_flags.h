/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_LLVM_BACKEND_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_LLVM_BACKEND_FLAGS_H_

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Append to *flag_list flag definitions associated with XLA's use of LLVM for
// code generation.
void AppendLlvmBackendFlags(std::vector<tensorflow::Flag>* flag_list);

// The values of flags associated with XLA's use of LLVM for code generation.
typedef struct {
  // Allows llvm to make transformations that reduce the precision of
  // floating-point computations, but it *does not* allow it to disregard signed
  // zero or assume that NaN and Inf never appear.
  //
  // Controls the "UnsafeFPMath" LLVM target option and
  // llvm::FastMathFlags::allowReciprocal.  This is equivalent to clang's
  // -funsafe-math-optimizations flag.
  bool xla_precision_losing_optimizations;

  // Unleashes the full power of LLVM's unsafe floating-point optimizations.
  // Everything is fair game, including disregarding signed zero and assuming
  // that NaN and Inf never appear.
  //
  // This implies xla_precision_losing_optimizations, and is equivalent to
  // clang's -ffast-math flag.
  bool xla_fast_math;
} LlvmBackendFlags;

// Return a pointer to the LlvmBackendFlags struct.  Repeated calls return the
// same pointer.  This should be called only after Flags::Parse() has returned.
LlvmBackendFlags* GetLlvmBackendFlags();

}  // namespace legacy_flags
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_LLVM_BACKEND_FLAGS_H_

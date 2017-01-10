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

// Legacy flags associated with XLA's use of LLVM for code generation.

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/llvm_backend_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static LlvmBackendFlags* flags;
static std::vector<tensorflow::Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new LlvmBackendFlags;
  flags->xla_fast_math = true;
  flags->xla_precision_losing_optimizations = true;
  flag_list = new std::vector<tensorflow::Flag>({
      tensorflow::Flag(
          "xla_precision_losing_optimizations",
          &flags->xla_precision_losing_optimizations,
          "Allows llvm to make transformations that reduce the precision of "
          "floating-point computations. This is equivalent to clang's "
          "-funsafe-math-optimizations flag."),
      tensorflow::Flag(
          "xla_fast_math", &flags->xla_fast_math,
          "Allows llvm to make all manner of unsafe floating-point "
          "optimizations, including assuming that NaN and Inf don't appear.  "
          "This is equivalent to clang's -ffast-math flag."),
  });
  ParseFlagsFromEnv(*flag_list);
}

void AppendLlvmBackendFlags(std::vector<tensorflow::Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

LlvmBackendFlags* GetLlvmBackendFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace xla

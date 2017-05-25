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

// Legacy flags for XLA's cpu_compiler module.

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static CpuCompilerFlags* flags;
static std::vector<tensorflow::Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new CpuCompilerFlags;
  flags->xla_cpu_llvm_opt_level = 2;
  flags->xla_cpu_llvm_cl_opts = "";
  flags->xla_cpu_embed_ir = false;
  flags->xla_cpu_parallel = false;
  flag_list = new std::vector<tensorflow::Flag>({
      tensorflow::Flag(
          "xla_cpu_llvm_opt_level", &flags->xla_cpu_llvm_opt_level,
          "The LLVM optimization level for the CPU XLA backend. "
          "Valid range is from 0 to 3 where 0 means no optimizations."),
      tensorflow::Flag(
          "xla_cpu_llvm_cl_opts", &flags->xla_cpu_llvm_cl_opts,
          "Comma-separated list of command line options to pass to LLVM."),
      tensorflow::Flag(
          "xla_cpu_embed_ir", &flags->xla_cpu_embed_ir,
          "Embed the LLVM IR module string in the resultant CpuExecutable."),
      tensorflow::Flag("xla_cpu_parallel", &flags->xla_cpu_parallel,
                       "Use the multi-threaded CPU backend."),
  });
  ParseFlagsFromEnv(*flag_list);
}

// Append to *append_to flag definitions associated with XLA's cpu_compiler
// module.
void AppendCpuCompilerFlags(std::vector<tensorflow::Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the CpuCompilerFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
CpuCompilerFlags* GetCpuCompilerFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace xla

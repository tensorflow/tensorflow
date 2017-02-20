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

// Legacy flags for XLA's gpu_backend_lib module.

#include <mutex>  // NOLINT(build/c++11): only using std::call_once, not mutex.
#include <vector>

#include "tensorflow/compiler/xla/legacy_flags/gpu_backend_lib_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static GpuBackendLibFlags* flags;
static std::vector<tensorflow::Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new GpuBackendLibFlags;
  flags->dump_temp_products_to = "";
  flags->ftz = false;
  flags->fma = true;
  flags->gpu_architecture = "compute_35";
  flags->verbose_ptx_asm = false;
  flags->kernel = "";
  flags->llvm_dump_passes = false;
  flags->llvm_cl_opts = "";
  flags->dump_ir_before_passes = false;
  flags->opt_level = 3;
  flag_list = new std::vector<tensorflow::Flag>({
      tensorflow::Flag("dump_temp_products_to", &flags->dump_temp_products_to,
                       "dump temporary compilation products to this directory. "
                       "If empty, no dump is produced"),
      tensorflow::Flag("ftz", &flags->ftz, "flush to zero semantics"),
      tensorflow::Flag("fma", &flags->fma, "use FMA synthesis"),
      tensorflow::Flag("gpu_architecture", &flags->gpu_architecture,
                       "GPU architecture"),
      tensorflow::Flag("verbose_ptx_asm", &flags->verbose_ptx_asm,
                       "emit PTX assembly with extra comments"),
      tensorflow::Flag("kernel", &flags->kernel,
                       "only emit the IR and PTX for this kernel"),
      tensorflow::Flag("llvm_dump_passes", &flags->llvm_dump_passes,
                       "dump the passes LLVM runs to stderr"),
      tensorflow::Flag(
          "llvm_cl_opts", &flags->llvm_cl_opts,
          "comma-separated list of command line options to pass to "
          "LLVM.  For example, --llvm_cl_opts=--print-before=loop-unroll"),
      tensorflow::Flag("dump_ir_before_passes", &flags->dump_ir_before_passes,
                       "dump the IR before each optimization pass in "
                       "sequentially-named files."),
      tensorflow::Flag("opt_level", &flags->opt_level,
                       "optimization level (default to 3)"),
  });
  ParseFlagsFromEnv(*flag_list);
}

// Append to *append_to flag definitions associated with XLA's gpu_backend_lib
// module.
void AppendGpuBackendLibFlags(std::vector<tensorflow::Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the GpuBackendLibFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
GpuBackendLibFlags* GetGpuBackendLibFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace xla

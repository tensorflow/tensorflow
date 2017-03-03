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

#ifndef TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_GPU_BACKEND_LIB_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_GPU_BACKEND_LIB_FLAGS_H_

// Legacy flags for XLA's gpu_backend_lib module.

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Append to *flag_list flag definitions associated with XLA's gpu_backend_lib
// module.
void AppendGpuBackendLibFlags(std::vector<tensorflow::Flag>* flag_list);

// The values of flags associated with XLA's gpu_backend_lib module.
typedef struct {
  string dump_temp_products_to;  // temporary compilation products dir
  bool ftz;                      // flush to zero semantics
  bool fma;                      // use FMA synthesis
  bool verbose_ptx_asm;          // emit PTX assembly with extra comments
  string kernel;                 // only emit the IR and PTX for this kernel
  bool llvm_dump_passes;         // dump the passes LLVM runs to stderr
  string llvm_cl_opts;           // comma-separated list of LLVM options
  bool dump_ir_before_passes;    // dump IR before each pass
  int32 opt_level;               // optimization level
} GpuBackendLibFlags;

// Return a pointer to the GpuBackendLibFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
GpuBackendLibFlags* GetGpuBackendLibFlags();

}  // namespace legacy_flags
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_GPU_BACKEND_LIB_FLAGS_H_

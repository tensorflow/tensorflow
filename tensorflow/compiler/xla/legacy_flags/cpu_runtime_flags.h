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

#ifndef TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_CPU_RUNTIME_FLAGS_H_
#define TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_CPU_RUNTIME_FLAGS_H_

// Legacy flags for the XLA's cpu_runtime module.

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace xla {
namespace legacy_flags {

// Append to *flag_list flag definitions associated with XLA's cpu_runtime
// module.
void AppendCpuRuntimeFlags(std::vector<tensorflow::Flag>* flag_list);

// The values of flags associated with XLA's cpu_runtime module.
typedef struct {
  // Use Eigen for matrix multiply on the CPU platform. This is a useful hack
  // for performance comparisons against XLA's implementation.
  bool xla_cpu_use_eigen;
  // When generating calls to Eigen for matmul and conv, should single or
  // multi-threaded eigen be used?  Only used when --xla_cpu_use_eigen is true.
  bool xla_cpu_multi_thread_eigen;
} CpuRuntimeFlags;

// Return a pointer to the CpuRuntimeFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
CpuRuntimeFlags* GetCpuRuntimeFlags();

}  // namespace legacy_flags
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LEGACY_FLAGS_CPU_RUNTIME_FLAGS_H_

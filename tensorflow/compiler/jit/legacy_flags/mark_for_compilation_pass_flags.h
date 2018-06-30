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

#ifndef TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_MARK_FOR_COMPILATION_PASS_FLAGS_H_
#define TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_MARK_FOR_COMPILATION_PASS_FLAGS_H_

// Legacy flags for the XLA bridge's mark_for_compilation_pass module.

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Append to *flag_list flag definitions associated with the XLA bridge's
// mark_for_compilation_pass module.
void AppendMarkForCompilationPassFlags(
    std::vector<tensorflow::Flag>* flag_list);

// The values of flags associated with the XLA bridge's
// mark_for_compilation_pass module.
typedef struct {
  int32 tf_xla_auto_jit;  // Control compilation of operators into XLA
                          // computations on CPU and GPU devices.  0 = use
                          // ConfigProto setting; -1 = off; 1 = on for things
                          // very likely to be improved; 2 = on for everything.
                          // Experimental.
  int32 tf_xla_min_cluster_size;  // Minimum number of operators in an XLA
                                  // compilation. Ignored for operators placed
                                  // on an XLA device or operators explicitly
                                  // marked for compilation.
  int32 tf_xla_max_cluster_size;  // Maximum number of operators in an XLA
                                  // compilation.
  bool tf_xla_clustering_debug;   // Dump graphs during XLA compilation.
  bool tf_xla_cpu_global_jit;     // Enables global JIT compilation for CPU
                                  // via SessionOptions.
  int64 tf_xla_clustering_fuel;   // "Compiler fuel" for clustering.  Only this
                                  // many ops will be marked as eligible for
                                  // clustering.
  bool tf_xla_fusion_only;  // This flag is effective only when global_jit_level
                            // is set to ON* and overrides its behavior. If
                            // true, enable fusion of element-wise operations
                            // only using XLA.
} MarkForCompilationPassFlags;

// Return a pointer to the MarkForCompilationPassFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
MarkForCompilationPassFlags* GetMarkForCompilationPassFlags();

}  // namespace legacy_flags
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_MARK_FOR_COMPILATION_PASS_FLAGS_H_

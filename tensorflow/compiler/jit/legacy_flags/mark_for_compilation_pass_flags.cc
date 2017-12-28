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

// Legacy flags for the XLA bridge's mark_for_compilation_pass module.

#include <mutex>
#include <vector>

#include "tensorflow/compiler/jit/legacy_flags/mark_for_compilation_pass_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static MarkForCompilationPassFlags* flags;
static std::vector<Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new MarkForCompilationPassFlags;
  flags->tf_xla_auto_jit = 0;
  flags->tf_xla_min_cluster_size = 2;
  flags->tf_xla_max_cluster_size = std::numeric_limits<int32>::max();
  flags->tf_xla_clustering_debug = false;
  flags->tf_xla_cpu_global_jit = false;
  flag_list = new std::vector<Flag>(
      {Flag("tf_xla_auto_jit", &flags->tf_xla_auto_jit,
            "Control compilation of operators into XLA computations on CPU and "
            "GPU devices.  0 = use ConfigProto setting; -1 = off; 1 = on for "
            "things very likely to be improved; 2 = on for everything.  "
            "Experimental."),
       Flag("tf_xla_min_cluster_size", &flags->tf_xla_min_cluster_size,
            "Minimum number of operators in an XLA compilation. Ignored for "
            "operators placed on an XLA device or operators explicitly marked "
            "for compilation."),
       Flag("tf_xla_max_cluster_size", &flags->tf_xla_max_cluster_size,
            "Maximum number of operators in an XLA compilation."),
       Flag("tf_xla_clustering_debug", &flags->tf_xla_clustering_debug,
            "Dump graphs during XLA compilation."),
       Flag("tf_xla_cpu_global_jit", &flags->tf_xla_cpu_global_jit,
            "Enables global JIT compilation for CPU via SessionOptions.")});
  xla::legacy_flags::ParseFlagsFromEnv(*flag_list);
}

// Append to *append_to flag definitions associated with the XLA bridge's
// mark_for_compilation_pass module.
void AppendMarkForCompilationPassFlags(std::vector<Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the MarkForCompilationPassFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
MarkForCompilationPassFlags* GetMarkForCompilationPassFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace tensorflow

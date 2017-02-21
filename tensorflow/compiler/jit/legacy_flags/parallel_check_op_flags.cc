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

// Legacy flags for the XLA bridge's parallel_check_op module.

#include <mutex>
#include <vector>

#include "tensorflow/compiler/jit/legacy_flags/parallel_check_op_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static ParallelCheckOpFlags* flags;
static std::vector<Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new ParallelCheckOpFlags;
  flags->parallel_check_failfast = true;
  flags->parallel_check_atol = "1e-5";
  flags->parallel_check_rtol = "1e-5";
  flag_list = new std::vector<Flag>({
      Flag("parallel_check_failfast", &flags->parallel_check_failfast,
           "Fail immediately on first parallel-check comparison error."),
      Flag("parallel_check_atol", &flags->parallel_check_atol,
           "Absolute error tolerance for parallel-check comparison."),
      Flag("parallel_check_rtol", &flags->parallel_check_rtol,
           "Relative error tolerance for parallel-check comparison."),
  });
  xla::legacy_flags::ParseFlagsFromEnv(*flag_list);
}

// Append to *append_to flag definitions associated with the XLA bridge's
// parallel_check_op module.
void AppendParallelCheckOpFlags(std::vector<Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the ParallelCheckOpFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
ParallelCheckOpFlags* GetParallelCheckOpFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace tensorflow

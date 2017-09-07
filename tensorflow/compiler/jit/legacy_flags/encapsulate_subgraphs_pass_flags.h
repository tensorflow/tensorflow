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

#ifndef TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_ENCAPSULATE_SUBGRAPHS_PASS_FLAGS_H_
#define TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_ENCAPSULATE_SUBGRAPHS_PASS_FLAGS_H_

// Legacy flags for the XLA bridge's encapsulate_subgraphs_pass module.

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Append to *flag_list flag definitions associated with the XLA bridge's
// encapsulate_subgraphs_pass module.
void AppendEncapsulateSubgraphsPassFlags(
    std::vector<tensorflow::Flag>* flag_list);

// The values of flags associated with the XLA bridge's
// encapsulate_subgraphs_pass module.
typedef struct {
  bool tf_xla_parallel_checking;  // Debug tool. Runs both JIT-compiled and
                                  // interpreted graphs in parallel and verifies
                                  // they produce the same outputs.
} EncapsulateSubgraphsPassFlags;

// Return a pointer to the EncapsulateSubgraphsPassFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
EncapsulateSubgraphsPassFlags* GetEncapsulateSubgraphsPassFlags();

}  // namespace legacy_flags
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_ENCAPSULATE_SUBGRAPHS_PASS_FLAGS_H_

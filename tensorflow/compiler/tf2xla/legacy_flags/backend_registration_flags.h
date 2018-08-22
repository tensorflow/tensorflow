/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LEGACY_FLAGS_BACKEND_REGISTRATION_FLAGS_H_
#define TENSORFLOW_COMPILER_TF2XLA_LEGACY_FLAGS_BACKEND_REGISTRATION_FLAGS_H_

// Legacy flags for the XLA bridge's backend registration modules.

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Append to *flag_list flag definitions associated with the XLA bridge's
// backend registration modules.
void AppendBackendRegistrationFlags(std::vector<tensorflow::Flag>* append_to);

// The values of flags associated with the XLA bridge's backend registration
// module.
typedef struct {
  // Whether to enable RandomUniform op on GPU backend.
  // TODO (b/32333178): Remove this flag or set its default to true.
  bool tf_enable_prng_ops_gpu;
} BackendRegistrationFlags;

// Return a pointer to the BackendRegistrationFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
BackendRegistrationFlags* GetBackendRegistrationFlags();

}  // namespace legacy_flags
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LEGACY_FLAGS_BACKEND_REGISTRATION_FLAGS_H_

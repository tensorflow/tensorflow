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

#ifndef TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_XLA_DEVICE_FLAGS_H_
#define TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_XLA_DEVICE_FLAGS_H_

// Legacy flags for the XLA bridge's xla_device module.

#include <vector>

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// The values of flags associated with the XLA bridge's
// xla_device module.
typedef struct {
  // Switch the CPU device into "on-demand" mode, where instead of
  // autoclustering ops are compiled one by one just-in-time.
  // Enabling this mode by a legacy flag is a temporary mechanism. When this
  // feature is battle-tested, we will switch this to be a session option.
  bool tf_xla_compile_on_demand;
} XlaDeviceFlags;

// Return a pointer to the XlaDeviceFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
XlaDeviceFlags* GetXlaDeviceFlags();

}  // namespace legacy_flags
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_LEGACY_FLAGS_XLA_DEVICE_FLAGS_H_

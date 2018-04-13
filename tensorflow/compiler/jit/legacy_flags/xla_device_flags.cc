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

// Legacy flags for the XLA bridge's xla_device module.

#include <mutex>
#include <vector>

#include "tensorflow/compiler/jit/legacy_flags/xla_device_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static XlaDeviceFlags* flags;
static std::vector<Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new XlaDeviceFlags;
  flags->tf_xla_compile_on_demand = false;
  flag_list = new std::vector<Flag>({
      Flag("tf_xla_compile_on_demand", &flags->tf_xla_compile_on_demand,
           "Switch a device into 'on-demand' mode, where instead of "
           "autoclustering ops are compiled one by one just-in-time."),
  });
  xla::legacy_flags::ParseFlagsFromEnv(*flag_list);
}

// Return a pointer to the XlaDeviceFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
XlaDeviceFlags* GetXlaDeviceFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace tensorflow

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

// Legacy flags for the XLA bridge's dump_graph module.

#include <mutex>
#include <vector>

#include "tensorflow/compiler/tf2xla/dump_graph_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/parse_flags_from_env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace tensorflow {
namespace legacy_flags {

// Pointers to the parsed value of the flags and flag descriptors, initialized
// via flags_init.
static DumpGraphFlags* flags;
static std::vector<Flag>* flag_list;
static std::once_flag flags_init;

// Allocate *flags.  Called via call_once(&flags_init,...).
static void AllocateFlags() {
  flags = new DumpGraphFlags;
  flags->tf_dump_graph_prefix = "/tmp/";
  flag_list = new std::vector<Flag>({
      Flag("tf_dump_graph_prefix", &flags->tf_dump_graph_prefix,
           "Path prefix to which graphs dumped during debugging should be "
           "written."),
  });
  xla::legacy_flags::ParseFlagsFromEnv(*flag_list);
}

// Append to *append_to flag definitions associated with the XLA bridge's
// dump_graph module.
void AppendDumpGraphFlags(std::vector<Flag>* append_to) {
  std::call_once(flags_init, &AllocateFlags);
  append_to->insert(append_to->end(), flag_list->begin(), flag_list->end());
}

// Return a pointer to the DumpGraphFlags struct;
// repeated calls return the same pointer.
// This should be called only after Flags::Parse() has returned.
DumpGraphFlags* GetDumpGraphFlags() {
  std::call_once(flags_init, &AllocateFlags);
  return flags;
}

}  // namespace legacy_flags
}  // namespace tensorflow

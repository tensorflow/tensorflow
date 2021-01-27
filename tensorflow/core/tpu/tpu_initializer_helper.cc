/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_initializer_helper.h"

#include <stdlib.h>

#include "absl/strings/str_split.h"

namespace tensorflow {
namespace tpu {

std::vector<const char*> GetLibTpuInitArguments() {
  std::vector<const char*> argv_ptr;
  std::vector<absl::string_view> argv;

  // Retrieve arguments from environment if applicable
  char* env = getenv("LIBTPU_INIT_ARGS");
  if (env != nullptr) {
    // TODO(frankchn): Handles quotes properly if necessary.
    // env pointer is already pointing to an allocated memory block.
    // absl::StrSplit returns a string_view that returns a vector of pointers
    // into that memory block. This means that we don't need to manage memory.
    argv = absl::StrSplit(env, ' ');
  }

  argv_ptr.reserve(argv.size());
  for (int i = 0; i < argv.size(); ++i) {
    argv_ptr.push_back(argv[i].data());
  }
  argv_ptr.push_back(nullptr);

  return argv_ptr;
}

}  // namespace tpu
}  // namespace tensorflow

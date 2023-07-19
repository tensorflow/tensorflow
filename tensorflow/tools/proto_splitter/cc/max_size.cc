/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/tools/proto_splitter/cc/max_size.h"

#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace tools::proto_splitter {
ABSL_CONST_INIT absl::Mutex global_mutex(absl::kConstInit);

// The default max size is set to a bit less than 2GB, since the proto splitter
// isn't extremely precise.
uint64_t ProtoMaxSize = ((uint64_t)1 << 31) - 500;

uint64_t GetMaxSize() { return ProtoMaxSize; }

void DebugSetMaxSize(uint64_t size) { ProtoMaxSize = size; }

}  // namespace tools::proto_splitter
}  // namespace tensorflow

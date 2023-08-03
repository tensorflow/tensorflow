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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_MAX_SIZE_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_MAX_SIZE_H_

#include <cstdint>

namespace tensorflow {
namespace tools::proto_splitter {

// Get max size allowed by each chunk in the proto splitter.
uint64_t GetMaxSize();

// Set the max size. Should only be used for testing purposes.
void DebugSetMaxSize(uint64_t size);

// Heuristic for determine whether to greedily generate a chunk.
#define LARGE_SIZE_CHECK(size, max_size) size > max_size / 3

}  // namespace tools::proto_splitter
}  // namespace tensorflow
#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_MAX_SIZE_H_

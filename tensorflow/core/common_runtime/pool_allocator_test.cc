/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/pool_allocator.h"

#include <limits>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

// This chunk prefix size should match ::ChunkPrefix in pool_allocator.cc.
static const size_t kChunkPrefixSize = sizeof(size_t) + sizeof(void*);
static const size_t kPoolAlignment = kChunkPrefixSize;

TEST(PoolAllocatorTest, Overflow) {
  PoolAllocator pool(0 /*pool_size_limit*/, false /*auto_resize*/,
                     new BasicCPUAllocator(0 /*numa_node*/, {}, {}),
                     new NoopRounder, "pool");

  // num_bytes + sizeof(ChunkPrefix) overflows.
  EXPECT_EQ(nullptr, pool.AllocateRaw(1, std::numeric_limits<size_t>::max() -
                                             kChunkPrefixSize + 1));

  // alignment + num_bytes overflows, when alignment > kPoolAlignment
  EXPECT_EQ(nullptr, pool.AllocateRaw(
                         kPoolAlignment + 1,
                         std::numeric_limits<size_t>::max() - kPoolAlignment));
}

}  // namespace
}  // namespace tensorflow

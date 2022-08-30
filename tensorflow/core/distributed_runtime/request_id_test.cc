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

#include "tensorflow/core/distributed_runtime/request_id.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Try requesting some request_ids and verify that none are zero.
TEST(GetUniqueRequestId, Basic) {
  for (int i = 0; i < 1000000; ++i) {
    EXPECT_NE(GetUniqueRequestId(), 0);
  }
}

TEST(GetShardedUniqueRequestId, Basic) {
  ShardUniqueRequestIdGenerator generator_0(3, 0);
  ShardUniqueRequestIdGenerator generator_1(4, 3);
  ShardUniqueRequestIdGenerator generator_2(6, 5);
  for (int i = 0; i < 1000000; ++i) {
    auto id = generator_0.GetUniqueRequestId();
    EXPECT_NE(id, 0);
    EXPECT_EQ(id & 0x0000000000000003, 0);
    id = generator_1.GetUniqueRequestId();
    EXPECT_NE(id, 0);
    EXPECT_EQ(id & 0x0000000000000003, 3);
    id = generator_2.GetUniqueRequestId();
    EXPECT_NE(id, 0);
    EXPECT_EQ(id & 0x0000000000000007, 5);
  }
}

}  // namespace tensorflow

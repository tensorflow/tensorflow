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

#include <vector>

#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

TEST(Hash, SignedUnsignedIssue) {
  const unsigned char d1[1] = {0x62};
  const unsigned char d2[2] = {0xc3, 0x97};
  const unsigned char d3[3] = {0xe2, 0x99, 0xa5};
  const unsigned char d4[4] = {0xe1, 0x80, 0xb9, 0x32};
  const unsigned char d5[48] = {
      0x01, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x18, 0x28, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  struct Case {
    uint32 hash32;
    uint64 hash64;
    const unsigned char* data;
    size_t size;
    uint32 seed;
  };

  for (Case c : std::vector<Case>{
           {0x471a8188u, 0x4c61ea3eeda4cb87ull, nullptr, 0, 0xbc9f1d34},
           {0xd615eba5u, 0x091309f7ef916c8aull, d1, sizeof(d1), 0xbc9f1d34},
           {0x0c3cccdau, 0xa815bcdf1d1af01cull, d2, sizeof(d2), 0xbc9f1d34},
           {0x3ba37e0eu, 0x02167564e4d06430ull, d3, sizeof(d3), 0xbc9f1d34},
           {0x16174eb3u, 0x8f7ed82ffc21071full, d4, sizeof(d4), 0xbc9f1d34},
           {0x98b1926cu, 0xce196580c97aff1eull, d5, sizeof(d5), 0x12345678},
       }) {
    EXPECT_EQ(c.hash32,
              Hash32(reinterpret_cast<const char*>(c.data), c.size, c.seed));
    EXPECT_EQ(c.hash64,
              Hash64(reinterpret_cast<const char*>(c.data), c.size, c.seed));

    // Check hashes with inputs aligned differently.
    for (int align = 1; align <= 7; align++) {
      std::string input(align, 'x');
      input.append(reinterpret_cast<const char*>(c.data), c.size);
      EXPECT_EQ(c.hash32, Hash32(&input[align], c.size, c.seed));
      EXPECT_EQ(c.hash64, Hash64(&input[align], c.size, c.seed));
    }
  }
}

static void BM_Hash32(int iters, int len) {
  std::string input(len, 'x');
  uint32 h = 0;
  for (int i = 0; i < iters; i++) {
    h = Hash32(input.data(), len, 1);
  }
  testing::BytesProcessed(static_cast<int64>(iters) * len);
  VLOG(1) << h;
}
BENCHMARK(BM_Hash32)->Range(1, 1024);

}  // namespace tensorflow

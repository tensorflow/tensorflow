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

#include "tensorflow/core/platform/cloud/file_block_cache.h"
#include <cstring>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(FileBlockCacheTest, PassThrough) {
  const uint64 want_offset = 42;
  const size_t want_n = 1024;
  int calls = 0;
  auto fetcher = [&calls, want_offset, want_n](uint64 got_offset, size_t got_n,
                                               std::vector<char>* out) {
    EXPECT_EQ(got_offset, want_offset);
    EXPECT_EQ(got_n, want_n);
    calls++;
    out->resize(got_n, 'x');
    return Status::OK();
  };
  // If block_size, block_count, or both are zero, the cache is a pass-through.
  FileBlockCache cache1(1, 0, fetcher);
  FileBlockCache cache2(0, 1, fetcher);
  FileBlockCache cache3(0, 0, fetcher);
  std::vector<char> out;
  TF_EXPECT_OK(cache1.Read(want_offset, want_n, &out));
  EXPECT_EQ(calls, 1);
  TF_EXPECT_OK(cache2.Read(want_offset, want_n, &out));
  EXPECT_EQ(calls, 2);
  TF_EXPECT_OK(cache3.Read(want_offset, want_n, &out));
  EXPECT_EQ(calls, 3);
}

TEST(FileBlockCacheTest, BlockAlignment) {
  // Initialize a 256-byte buffer.  This is the file underlying the reads we'll
  // do in this test.
  const size_t size = 256;
  std::vector<char> buf;
  for (int i = 0; i < size; i++) {
    buf.push_back(i);
  }
  // The fetcher just fetches slices of the buffer.
  auto fetcher = [&buf](uint64 offset, size_t n, std::vector<char>* out) {
    if (offset < buf.size()) {
      if (offset + n > buf.size()) {
        out->insert(out->end(), buf.begin() + offset, buf.end());
      } else {
        out->insert(out->end(), buf.begin() + offset, buf.begin() + offset + n);
      }
    }
    return Status::OK();
  };
  for (uint64_t block_size = 2; block_size <= 4; block_size++) {
    // Make a cache of N-byte block size (1 block) and verify that reads of
    // varying offsets and lengths return correct data.
    FileBlockCache cache(block_size, 1, fetcher);
    for (uint64_t offset = 0; offset < 10; offset++) {
      for (size_t n = block_size - 2; n <= block_size + 2; n++) {
        std::vector<char> got;
        TF_EXPECT_OK(cache.Read(offset, n, &got));
        // Verify the size of the read.
        if (offset + n <= size) {
          // Expect a full read.
          EXPECT_EQ(got.size(), n) << "block size = " << block_size
                                   << ", offset = " << offset << ", n = " << n;
        } else {
          // Expect a partial read.
          EXPECT_EQ(got.size(), size - offset)
              << "block size = " << block_size << ", offset = " << offset
              << ", n = " << n;
        }
        // Verify the contents of the read.
        std::vector<char>::const_iterator begin = buf.begin() + offset;
        std::vector<char>::const_iterator end =
            offset + n > buf.size() ? buf.end() : begin + n;
        std::vector<char> want(begin, end);
        EXPECT_EQ(got, want) << "block size = " << block_size
                             << ", offset = " << offset << ", n = " << n;
      }
    }
  }
}

TEST(FileBlockCacheTest, CacheHits) {
  const uint64 block_size = 16;
  std::set<uint64_t> calls;
  auto fetcher = [&calls, block_size](uint64 offset, size_t n,
                                      std::vector<char>* out) {
    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset % block_size, 0);
    EXPECT_EQ(calls.find(offset), calls.end()) << "at offset " << offset;
    calls.insert(offset);
    out->resize(n, 'x');
    return Status::OK();
  };
  const uint32 block_count = 256;
  FileBlockCache cache(block_size, block_count, fetcher);
  std::vector<char> out;
  // The cache has space for `block_count` blocks. The loop with i = 0 should
  // fill the cache, and the loop with i = 1 should be all cache hits. The
  // fetcher checks that it is called once and only once for each offset (to
  // fetch the corresponding block).
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < block_count; j++) {
      TF_EXPECT_OK(cache.Read(block_size * j, block_size, &out));
    }
  }
}

TEST(FileBlockCacheTest, OutOfRange) {
  // Tests reads of a 24-byte file with block size 16.
  const uint64 block_size = 16;
  const uint64 file_size = 24;
  bool first_block = false;
  bool second_block = false;
  auto fetcher = [block_size, file_size, &first_block, &second_block](
                     uint64 offset, size_t n, std::vector<char>* out) {
    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset % block_size, 0);
    if (offset == 0) {
      // The first block (16 bytes) of the file.
      out->resize(n, 'x');
      first_block = true;
    } else if (offset == block_size) {
      // The second block (8 bytes) of the file.
      out->resize(file_size - block_size, 'x');
      second_block = true;
    }
    return Status::OK();
  };
  FileBlockCache cache(block_size, 1, fetcher);
  std::vector<char> out;
  // Reading the first 16 bytes should be fine.
  TF_EXPECT_OK(cache.Read(0, block_size, &out));
  EXPECT_TRUE(first_block);
  EXPECT_EQ(out.size(), block_size);
  // Reading at offset file_size + 4 will read the second block (since the read
  // at file_size + 4 = 28 will be aligned to an offset of 16) but will return
  // OutOfRange because the offset is past the end of the 24-byte file.
  Status status = cache.Read(file_size + 4, 4, &out);
  EXPECT_EQ(status.code(), error::OUT_OF_RANGE);
  EXPECT_TRUE(second_block);
  EXPECT_EQ(out.size(), 0);
  // Reading the second full block will return 8 bytes, from a cache hit.
  second_block = false;
  TF_EXPECT_OK(cache.Read(block_size, block_size, &out));
  EXPECT_FALSE(second_block);
  EXPECT_EQ(out.size(), file_size - block_size);
}

TEST(FileBlockCacheTest, Inconsistent) {
  // Tests the detection of interrupted reads leading to partially filled blocks
  // where we expected complete blocks.
  const uint64 block_size = 16;
  // This fetcher returns OK but only fills in one byte for any offset.
  auto fetcher = [block_size](uint64 offset, size_t n, std::vector<char>* out) {
    EXPECT_EQ(n, block_size);
    EXPECT_EQ(offset % block_size, 0);
    out->resize(1, 'x');
    return Status::OK();
  };
  FileBlockCache cache(block_size, 2, fetcher);
  std::vector<char> out;
  // Read the second block; this should yield an OK status and a single byte.
  TF_EXPECT_OK(cache.Read(block_size, block_size, &out));
  EXPECT_EQ(out.size(), 1);
  // Now read the first block; this should yield FAILED_PRECONDITION because we
  // had already cached a partial block at a later position.
  Status status = cache.Read(0, block_size, &out);
  EXPECT_EQ(status.code(), error::FAILED_PRECONDITION);
}

TEST(FileBlockCacheTest, LRU) {
  const uint64 block_size = 16;
  std::list<uint64_t> calls;
  auto fetcher = [&calls, block_size](uint64 offset, size_t n,
                                      std::vector<char>* out) {
    EXPECT_EQ(n, block_size);
    EXPECT_FALSE(calls.empty()) << "at offset = " << offset;
    if (!calls.empty()) {
      EXPECT_EQ(offset, calls.front());
      calls.pop_front();
    }
    out->resize(n, 'x');
    return Status::OK();
  };
  const uint32 block_count = 2;
  FileBlockCache cache(block_size, block_count, fetcher);
  std::vector<char> out;
  // Read blocks from the cache, and verify the LRU behavior based on the
  // fetcher calls that the cache makes.
  calls.push_back(0);
  // Cache miss - drains an element from `calls`.
  TF_EXPECT_OK(cache.Read(0, 1, &out));
  // Cache hit - does not drain an element from `calls`.
  TF_EXPECT_OK(cache.Read(0, 1, &out));
  calls.push_back(block_size);
  // Cache miss followed by cache hit.
  TF_EXPECT_OK(cache.Read(block_size, 1, &out));
  TF_EXPECT_OK(cache.Read(block_size, 1, &out));
  calls.push_back(2 * block_size);
  // Cache miss followed by cache hit.  Causes eviction of LRU element.
  TF_EXPECT_OK(cache.Read(2 * block_size, 1, &out));
  TF_EXPECT_OK(cache.Read(2 * block_size, 1, &out));
  // LRU element was at offset 0.  Cache miss.
  calls.push_back(0);
  TF_EXPECT_OK(cache.Read(0, 1, &out));
  // Element at 2 * block_size is still in cache, and this read should update
  // its position in the LRU list so it doesn't get evicted by the next read.
  TF_EXPECT_OK(cache.Read(2 * block_size, 1, &out));
  // Element at block_size was evicted.  Reading this element will also cause
  // the LRU element (at 0) to be evicted.
  calls.push_back(block_size);
  TF_EXPECT_OK(cache.Read(block_size, 1, &out));
  // Element at 0 was evicted again.
  calls.push_back(0);
  TF_EXPECT_OK(cache.Read(0, 1, &out));
}

}  // namespace
}  // namespace tensorflow

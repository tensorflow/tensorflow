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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/pool_allocator.h"

#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
namespace tensorflow {
namespace {

TEST(PoolAllocatorTest, ZeroSizeBuffers) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  PoolAllocator pool(
      2 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .value(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");

  EXPECT_EQ(nullptr, pool.AllocateRaw(4 /*alignment*/, 0 /*num_bytes*/));
  pool.DeallocateRaw(nullptr);  // Should not crash.
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());
}

TEST(PoolAllocatorTest, ZeroSizePool) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  PoolAllocator pool(
      0 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .value(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");

  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());

  // All allocations should bypass the pool and return valid pointers.
  for (int i = 0; i < 3; ++i) {
    void* p0 = pool.AllocateRaw(4, 0);
    void* p4 = pool.AllocateRaw(4, 4);
    void* p12 = pool.AllocateRaw(4, 12);
    EXPECT_EQ(nullptr, p0);
    EXPECT_NE(nullptr, p4);
    EXPECT_NE(nullptr, p12);
    pool.DeallocateRaw(p0);
    pool.DeallocateRaw(p4);
    pool.DeallocateRaw(p12);
  }
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());
}

TEST(PoolAllocatorTest, Alignment) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  PoolAllocator pool(
      0 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .value(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");
  for (int i = 0; i < 16; ++i) {
    size_t alignment = 1 << i;
    void* p = pool.AllocateRaw(alignment, 111);
    EXPECT_TRUE(p != nullptr);
    EXPECT_EQ(0, reinterpret_cast<int64_t>(p) & (alignment - 1))
        << "ptr: " << p << " alignment " << alignment;
    // Intentionally don't deallocate, to test that destruction of
    // the PoolAllocator frees all pending memory.
  }
}

TEST(PoolAllocatorTest, AutoResize) {
  PoolAllocator pool(2 /*pool_size_limit*/, true /*auto_resize*/,
                     new BasicCPUAllocator(0 /*numa_node*/, {}, {}),
                     new NoopRounder, "pool");

  // Alloc/dealloc 10 sizes just a few times, confirming pool size
  // stays at 2.
  for (int i = 0; i < 10; ++i) {
    void* p = pool.AllocateRaw(4, 64 << i);
    pool.DeallocateRaw(p);
  }
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(10, pool.allocated_count());
  EXPECT_EQ(10, pool.put_count());
  EXPECT_EQ(8, pool.evicted_count());
  EXPECT_EQ(2, pool.size_limit());

  // Then repeat 1200 times.  Pool size limit should jump to 100.
  for (int j = 0; j < 120; ++j) {
    for (int i = 0; i < 10; ++i) {
      void* p = pool.AllocateRaw(4, 64 << i);
      pool.DeallocateRaw(p);
    }
  }
  EXPECT_EQ(100, pool.size_limit());
}

TEST(PoolAllocatorTest, CudaHostAllocator) {
  int alloc_count = 0;
  int64_t alloc_size = 0;
  SubAllocator::Visitor alloc_visitor =
      [&alloc_count, &alloc_size](void* ptr, int numa_node, int64_t size) {
        ++alloc_count;
        alloc_size += size;
      };
  int free_count = 0;
  int64_t free_size = 0;
  SubAllocator::Visitor free_visitor =
      [&free_count, &free_size](void* ptr, int numa_node, int64_t size) {
        ++free_count;
        free_size += size;
      };
  se::Platform* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  DeviceHostAllocator* sub_allocator = new DeviceHostAllocator(
      platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0)).value(),
      0 /*numa_node*/, {alloc_visitor}, {free_visitor});
  PoolAllocator pool(2 /*pool_size_limit*/, false /*auto_resize*/,
                     sub_allocator, new NoopRounder, "pool");
  EXPECT_EQ(0, alloc_count);
  EXPECT_EQ(0, alloc_size);
  EXPECT_EQ(0, free_count);
  EXPECT_EQ(0, free_size);

  // Repeatedly Get a 16-byte value, confirming that there's only
  // one real allocation.
  void* p1_16 = pool.AllocateRaw(4, 16);
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(1, pool.allocated_count());
  EXPECT_NE(nullptr, p1_16);
  EXPECT_EQ(1, alloc_count);  // Underlying suballoc of 16 bytes
  // Each suballocation includes a 16B ChunkPrefix.
  static const int kChunkPrefixSize = 16;
  EXPECT_EQ(16 + (alloc_count * kChunkPrefixSize), alloc_size);
  pool.DeallocateRaw(p1_16);
  // Pool contents {16}
  EXPECT_EQ(1, pool.put_count());
  void* p2_16 = pool.AllocateRaw(4, 16);  // Get it again.
  EXPECT_EQ(1, pool.get_from_pool_count());
  EXPECT_EQ(1, pool.allocated_count());
  EXPECT_EQ(p1_16, p2_16);    // Same pointer value
  pool.DeallocateRaw(p2_16);  // Put it back.
  // Pool contents {16}
  EXPECT_EQ(2, pool.put_count());
  EXPECT_EQ(1, alloc_count);  // Underlying suballoc of 16 bytes
  EXPECT_EQ(16 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(0, free_count);

  // Get two more values of different sizes.
  void* p3_4 = pool.AllocateRaw(4, 4);
  EXPECT_EQ(2, pool.allocated_count());
  EXPECT_NE(p1_16, p3_4);  // Different pointer value
  EXPECT_NE(nullptr, p3_4);
  pool.DeallocateRaw(p3_4);  // Put it back. Pool is now full.
  // Pool contents {4, 16}
  EXPECT_EQ(3, pool.put_count());
  void* p4_2 = pool.AllocateRaw(4, 2);  // Get a third size buffer.
  EXPECT_NE(nullptr, p4_2);
  EXPECT_EQ(0, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(0, free_count);

  // The pool is full: when we put back p4_2, the 16-byte buffer
  // should be evicted since it was least recently inserted.
  pool.DeallocateRaw(p4_2);
  // Pool contents {2, 4}
  EXPECT_EQ(4, pool.put_count());
  EXPECT_EQ(1, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(1, free_count);
  EXPECT_EQ(16 + (free_count * kChunkPrefixSize), free_size);

  // Re-getting and putting size 2 or 4 should not alter pool size or
  // num-evicted.
  void* p5_4 = pool.AllocateRaw(4, 4);
  EXPECT_NE(nullptr, p5_4);
  pool.DeallocateRaw(p5_4);
  void* p6_2 = pool.AllocateRaw(4, 2);
  EXPECT_NE(nullptr, p6_2);
  pool.DeallocateRaw(p6_2);
  EXPECT_EQ(3, pool.get_from_pool_count());
  EXPECT_EQ(6, pool.put_count());
  EXPECT_EQ(3, pool.allocated_count());
  EXPECT_EQ(1, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(1, free_count);
  EXPECT_EQ(16 + (free_count * kChunkPrefixSize), free_size);

  pool.Clear();
  EXPECT_EQ(0, pool.get_from_pool_count());
  EXPECT_EQ(0, pool.put_count());
  EXPECT_EQ(0, pool.allocated_count());
  EXPECT_EQ(0, pool.evicted_count());
  EXPECT_EQ(3, alloc_count);
  EXPECT_EQ(16 + 4 + 2 + (alloc_count * kChunkPrefixSize), alloc_size);
  EXPECT_EQ(3, free_count);
  EXPECT_EQ(16 + 4 + 2 + (free_count * kChunkPrefixSize), free_size);
}

TEST(PoolAllocatorTest, Pow2Rounder) {
  Pow2Rounder rounder;
  EXPECT_EQ(1, rounder.RoundUp(1));
  EXPECT_EQ(2, rounder.RoundUp(2));
  EXPECT_EQ(16, rounder.RoundUp(9));
  EXPECT_EQ(16, rounder.RoundUp(16));
  EXPECT_EQ(65536, rounder.RoundUp(41234));
  EXPECT_EQ(65536, rounder.RoundUp(65535));
  EXPECT_EQ(65536, rounder.RoundUp(65536));
}

TEST(PoolAllocatorTest, Name) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  PoolAllocator pool(
      2 /*pool_size_limit*/, false /*auto_resize*/,
      new DeviceHostAllocator(
          platform->GetExecutor(se::StreamExecutorConfig(/*ordinal=*/0))
              .value(),
          0 /*numa_node*/, {}, {}),
      new NoopRounder, "pool");
  EXPECT_EQ("pool", pool.Name());
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

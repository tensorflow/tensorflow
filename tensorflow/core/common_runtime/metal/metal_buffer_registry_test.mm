/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/metal/metal_buffer_registry.h"

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace tensorflow {
namespace metal {
namespace {

id<MTLDevice> DefaultDeviceOrNil() { return MTLCreateSystemDefaultDevice(); }

id<MTLBuffer> NewSharedBuffer(id<MTLDevice> device, size_t size) {
  return [device newBufferWithLength:size
                             options:MTLResourceStorageModeShared];
}

class MetalBufferRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = DefaultDeviceOrNil();
    if (device_ == nil) {
      GTEST_SKIP() << "No Metal device available on this machine.";
    }
  }
  id<MTLDevice> device_ = nil;
};

TEST_F(MetalBufferRegistryTest, RegisterReturnsHostVisibleContents) {
  id<MTLBuffer> buffer = NewSharedBuffer(device_, 1024);
  ASSERT_NE(buffer, nil);

  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);
  // The address core sees must be the buffer's own contents pointer, which is
  // what makes it safe for core to do pointer arithmetic on it.
  EXPECT_EQ(address, [buffer contents]);

  EXPECT_TRUE(MetalBufferRegistry::Global().Unregister(address));
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, LookupResolvesBaseAddress) {
  id<MTLBuffer> buffer = NewSharedBuffer(device_, 4096);
  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);

  id<MTLBuffer> found = nil;
  size_t offset = 12345;
  ASSERT_TRUE(MetalBufferRegistry::Global().Lookup(address, &found, &offset));
  EXPECT_EQ(found, buffer);
  EXPECT_EQ(offset, 0u);

  MetalBufferRegistry::Global().Unregister(address);
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, LookupResolvesInteriorPointer) {
  constexpr size_t kSize = 4096;
  constexpr size_t kOffset = 1728;
  id<MTLBuffer> buffer = NewSharedBuffer(device_, kSize);
  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);

  // This is the case the registry exists for: the BFC allocator sub-divides
  // an allocation and kernels see pointers into the middle of it.
  void* interior = static_cast<char*>(address) + kOffset;
  id<MTLBuffer> found = nil;
  size_t offset = 0;
  ASSERT_TRUE(MetalBufferRegistry::Global().Lookup(interior, &found, &offset));
  EXPECT_EQ(found, buffer);
  EXPECT_EQ(offset, kOffset);

  MetalBufferRegistry::Global().Unregister(address);
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, LookupRejectsOnePastTheEnd) {
  constexpr size_t kSize = 2048;
  id<MTLBuffer> buffer = NewSharedBuffer(device_, kSize);
  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);

  void* past_end = static_cast<char*>(address) + kSize;
  EXPECT_FALSE(MetalBufferRegistry::Global().Lookup(past_end, nullptr,
                                                    nullptr));

  MetalBufferRegistry::Global().Unregister(address);
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, LookupFailsAfterUnregister) {
  id<MTLBuffer> buffer = NewSharedBuffer(device_, 512);
  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);
  ASSERT_TRUE(MetalBufferRegistry::Global().Unregister(address));

  EXPECT_FALSE(MetalBufferRegistry::Global().Lookup(address, nullptr, nullptr));
  // Unregistering twice must not double-release the buffer.
  EXPECT_FALSE(MetalBufferRegistry::Global().Unregister(address));
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, UnregisterRejectsInteriorPointer) {
  id<MTLBuffer> buffer = NewSharedBuffer(device_, 1024);
  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);

  void* interior = static_cast<char*>(address) + 8;
  EXPECT_FALSE(MetalBufferRegistry::Global().Unregister(interior));
  EXPECT_TRUE(MetalBufferRegistry::Global().Unregister(address));
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, StatsTrackBytesInUse) {
  constexpr size_t kSize = 8192;
  const auto before = MetalBufferRegistry::Global().GetStats();

  id<MTLBuffer> buffer = NewSharedBuffer(device_, kSize);
  void* address = MetalBufferRegistry::Global().Register(buffer);
  ASSERT_NE(address, nullptr);

  const auto during = MetalBufferRegistry::Global().GetStats();
  EXPECT_EQ(during.num_allocs, before.num_allocs + 1);
  EXPECT_EQ(during.bytes_in_use,
            before.bytes_in_use + static_cast<int64_t>(kSize));
  EXPECT_GE(during.peak_bytes_in_use, during.bytes_in_use);
  EXPECT_GE(during.largest_alloc_size, static_cast<int64_t>(kSize));

  MetalBufferRegistry::Global().Unregister(address);
  const auto after = MetalBufferRegistry::Global().GetStats();
  EXPECT_EQ(after.bytes_in_use, before.bytes_in_use);
  // Peak is a high-water mark and must not fall back on deallocation.
  EXPECT_EQ(after.peak_bytes_in_use, during.peak_bytes_in_use);
  [buffer release];
}

TEST_F(MetalBufferRegistryTest, RegisterRejectsNil) {
  EXPECT_EQ(MetalBufferRegistry::Global().Register(nil), nullptr);
}

TEST_F(MetalBufferRegistryTest, RegisterRejectsPrivateStorage) {
  // Private buffers have no host-visible contents, so registering one would
  // hand core an address it cannot legally offset.
  id<MTLBuffer> buffer =
      [device_ newBufferWithLength:256
                           options:MTLResourceStorageModePrivate];
  ASSERT_NE(buffer, nil);
  EXPECT_EQ(MetalBufferRegistry::Global().Register(buffer), nullptr);
  [buffer release];
}

}  // namespace
}  // namespace metal
}  // namespace tensorflow

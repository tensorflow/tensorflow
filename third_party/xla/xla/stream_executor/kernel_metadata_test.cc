/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/kernel_metadata.h"

#include <optional>

#include <gtest/gtest.h>

namespace stream_executor {
namespace {

TEST(KernelMetadataTest, DefaultConstructor) {
  KernelMetadata metadata;
  EXPECT_EQ(metadata.registers_per_thread(), std::nullopt);
  EXPECT_EQ(metadata.shared_memory_bytes(), std::nullopt);
}

TEST(KernelMetadataTest, SetRegistersPerThread) {
  KernelMetadata metadata;
  metadata.set_registers_per_thread(10);
  EXPECT_EQ(metadata.registers_per_thread(), 10);
}

TEST(KernelMetadataTest, SetSharedMemoryBytes) {
  KernelMetadata metadata;
  metadata.set_shared_memory_bytes(1024);
  EXPECT_EQ(metadata.shared_memory_bytes(), 1024);
}

}  // namespace
}  // namespace stream_executor

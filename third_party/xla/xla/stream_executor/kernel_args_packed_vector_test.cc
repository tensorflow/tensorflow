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

#include "xla/stream_executor/kernel_args_packed_vector.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/types/span.h"

namespace stream_executor {
namespace {
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Ne;
using ::testing::SizeIs;

TEST(KernelArgsPackedVectorTest, StoresSharedMemoryBytes) {
  KernelArgsPackedVector args(std::vector<std::vector<char>>(),
                              /*shared_memory_bytes=*/42);
  EXPECT_EQ(args.number_of_shared_bytes(), 42);
  EXPECT_EQ(args.number_of_arguments(), 1);
}

TEST(KernelArgsPackedVectorTest, StoresArgumentAddresses) {
  KernelArgsPackedVector args = []() {
    std::vector<std::vector<char>> storage;
    storage.push_back(std::vector<char>({10}));
    storage.push_back(std::vector<char>({20, 21}));
    storage.push_back(std::vector<char>({30, 31, 32}));
    return KernelArgsPackedVector(std::move(storage),
                                  /*shared_memory_bytes=*/0);
  }();

  EXPECT_EQ(args.number_of_arguments(), 3);
  ASSERT_THAT(args.argument_addresses(), SizeIs(3));
  ASSERT_THAT(args.argument_addresses(), Each(Ne(nullptr)));

  EXPECT_THAT(
      absl::Span<const char>(
          absl::bit_cast<const char*>(args.argument_addresses().at(0)), 1),
      ElementsAre(10));
  EXPECT_THAT(
      absl::Span<const char>(
          absl::bit_cast<const char*>(args.argument_addresses().at(1)), 2),
      ElementsAre(20, 21));
  EXPECT_THAT(
      absl::Span<const char>(
          absl::bit_cast<const char*>(args.argument_addresses().at(2)), 3),
      ElementsAre(30, 31, 32));
}

TEST(KernelArgsPackedVectorTest,
     ConsidersSharedMemoryBytesInNumberOfArguments) {
  KernelArgsPackedVector args({{}, {}},
                              /*shared_memory_bytes=*/42);

  // Two arguments and one shared memory argument.
  EXPECT_EQ(args.number_of_arguments(), 3);
}

}  // namespace
}  // namespace stream_executor

// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/accelerator.h"

#include <gtest/gtest.h>

namespace litert::internal {
namespace {

TEST(AcceleratorRegistryTest, CreateEmptyAcceleratorWorks) {
  [[maybe_unused]]
  auto accelerator_squeleton = AcceleratorRegistry::CreateEmptyAccelerator();
}

TEST(AcceleratorRegistryTest, AcceleratorCanBeRegisteredAndRetrieved) {
  AcceleratorRegistry registry;

  auto registered_accelerator1 = registry.RegisterAccelerator(
      AcceleratorRegistry::CreateEmptyAccelerator());
  ASSERT_TRUE(registered_accelerator1);

  auto registered_accelerator2 = registry.RegisterAccelerator(
      AcceleratorRegistry::CreateEmptyAccelerator());
  ASSERT_TRUE(registered_accelerator2);

  ASSERT_NE(registered_accelerator1, registered_accelerator2);

  auto queried_accelerator1 = registry.Get(0);
  ASSERT_TRUE(queried_accelerator1);
  EXPECT_EQ(queried_accelerator1, registered_accelerator1);

  auto queried_accelerator2 = registry.Get(1);
  ASSERT_TRUE(queried_accelerator2);
  EXPECT_EQ(queried_accelerator2, registered_accelerator2);

  EXPECT_FALSE(registry.Get(2));
  EXPECT_FALSE(registry.Get(-1));

  auto idx1 = registry.FindAcceleratorIndex(queried_accelerator1.Value());
  ASSERT_TRUE(idx1);
  EXPECT_EQ(idx1.Value(), 0);

  auto idx2 = registry.FindAcceleratorIndex(queried_accelerator2.Value());
  ASSERT_TRUE(idx2);
  EXPECT_EQ(idx2.Value(), 1);
}

}  // namespace
}  // namespace litert::internal

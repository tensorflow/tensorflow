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

#include "xla/service/maybe_owning_device_memory.h"

#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using MaybeOwningDeviceMemoryTest = ::testing::Test;

TEST(MaybeOwningDeviceMemoryTest, DefaultConstructed) {
  MaybeOwningDeviceMemory memory;
  EXPECT_FALSE(memory.HasOwnership());
  EXPECT_EQ(memory.AsDeviceMemoryBase().opaque(), nullptr);
  EXPECT_EQ(memory.AsDeviceMemoryBase().size(), 0);
}

//===-----------------------------------------------------------------------===/
// Performance benchmarks below.
//===-----------------------------------------------------------------------===/

void BM_DefaultConstructed(benchmark::State& state) {
  for (auto s : state) {
    MaybeOwningDeviceMemory memory;
    benchmark::DoNotOptimize(memory);
  }
}

BENCHMARK(BM_DefaultConstructed);

}  // namespace
}  // namespace xla

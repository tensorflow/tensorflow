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

#include "xla/stream_executor/kernel_symbol_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"

namespace stream_executor {
namespace {

using absl_testing::IsOk;
using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;

TEST(KernelSymbolRegistryTest, RegisterSymbol) {
  KernelSymbolRegistry registry;
  EXPECT_THAT(registry.RegisterSymbol("symbol_name", cuda::kCudaPlatformId,
                                      /*symbol=*/nullptr),
              IsOk());
  EXPECT_THAT(registry.RegisterSymbol("symbol_name", cuda::kCudaPlatformId,
                                      /*symbol=*/nullptr),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

void PretendsToBeAKernel(int* x) { *x = 42; }

TEST(KernelSymbolRegistryTest, FindSymbol) {
  KernelSymbolRegistry registry;
  EXPECT_THAT(registry.FindSymbol("symbol_name", cuda::kCudaPlatformId),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(registry.RegisterSymbol("symbol_name", cuda::kCudaPlatformId,
                                      &PretendsToBeAKernel),
              IsOk());
  EXPECT_THAT(registry.FindSymbol("symbol_name", cuda::kCudaPlatformId),
              IsOkAndHolds(absl::bit_cast<void*>(&PretendsToBeAKernel)));
}

KERNEL_SYMBOL_REGISTRY_REGISTER_SYMBOL_STATICALLY(static_test_symbol,
                                                  cuda::kCudaPlatformId,
                                                  &PretendsToBeAKernel);

TEST(KernelSymbolRegistryTest, StaticRegistration) {
  EXPECT_THAT(KernelSymbolRegistry::GetGlobalInstance().FindSymbol(
                  "static_test_symbol", cuda::kCudaPlatformId),
              IsOkAndHolds(absl::bit_cast<void*>(&PretendsToBeAKernel)));
}

}  // namespace
}  // namespace stream_executor

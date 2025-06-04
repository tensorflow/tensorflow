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

#include "xla/stream_executor/platform/platform_object_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor {
namespace {

struct TestTrait {
  using Type = int;
};

struct OtherTestTrait {
  using Type = float;
};

struct StaticTestTrait {
  using Type = int;
};

using tsl::testing::IsOk;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

TEST(PlatformObjectRegistryTest, RegisterObject) {
  PlatformObjectRegistry registry;

  // Can register a simple kernel
  EXPECT_THAT(registry.RegisterObject<TestTrait>(
                  stream_executor::cuda::kCudaPlatformId, 42),
              IsOk());

  // Can register another simple kernel - no clash
  EXPECT_THAT(registry.RegisterObject<OtherTestTrait>(
                  stream_executor::cuda::kCudaPlatformId, 42.0f),
              IsOk());

  // Can register a different kernel under the same trait for a different
  // platform.
  EXPECT_THAT(registry.RegisterObject<TestTrait>(
                  stream_executor::rocm::kROCmPlatformId, 44),
              IsOk());

  // Can't register a kernel if it already exists in the registry.
  EXPECT_THAT(registry.RegisterObject<TestTrait>(
                  stream_executor::cuda::kCudaPlatformId, 44),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(PlatformObjectRegistryTest, RegisterObjectConcurrently) {
  // This test will show races in the registry implementation when run with
  // `--config=tsan`.

  PlatformObjectRegistry registry;

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&] {
    int cuda_value = 11;
    // Can register a simple kernel
    EXPECT_THAT(registry.RegisterObject<TestTrait>(
                    stream_executor::cuda::kCudaPlatformId, cuda_value),
                IsOk());
  });

  pool.Schedule([&] {
    // Can register a different kernel under the same trait for a different
    // platform.
    EXPECT_THAT(registry.RegisterObject<TestTrait>(
                    stream_executor::rocm::kROCmPlatformId, 42),
                IsOk());
  });
}

TEST(PlatformObjectRegistryTest, FindObject) {
  PlatformObjectRegistry registry;

  ASSERT_THAT(registry.RegisterObject<TestTrait>(
                  stream_executor::cuda::kCudaPlatformId, 33),
              IsOk());

  EXPECT_THAT(
      registry.FindObject<TestTrait>(stream_executor::cuda::kCudaPlatformId),
      IsOkAndHolds(33));

  // No registered kernel for ROCM.
  EXPECT_THAT(
      registry.FindObject<TestTrait>(stream_executor::rocm::kROCmPlatformId),
      StatusIs(absl::StatusCode::kNotFound));

  // No registered kernel for the other trait.
  EXPECT_THAT(registry.FindObject<OtherTestTrait>(
                  stream_executor::cuda::kCudaPlatformId),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(PlatformObjectRegistryTest, FindObjectConcurrently) {
  // This test will show races in the registry implementation when run with
  // `--config=tsan`.

  PlatformObjectRegistry registry;

  ASSERT_THAT(registry.RegisterObject<TestTrait>(
                  stream_executor::cuda::kCudaPlatformId, 333),
              IsOk());

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&] {
    EXPECT_THAT(
        registry.FindObject<TestTrait>(stream_executor::cuda::kCudaPlatformId),
        IsOkAndHolds(333));
  });

  pool.Schedule([&] {
    EXPECT_THAT(
        registry.FindObject<TestTrait>(stream_executor::cuda::kCudaPlatformId),
        IsOkAndHolds(333));
  });
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(StaticTestTraitRegistration,
                                           StaticTestTrait,
                                           cuda::kCudaPlatformId, 142);

TEST(PlatformObjectRegistryTest, FindStaticallyRegisteredObject) {
  EXPECT_THAT(
      PlatformObjectRegistry::GetGlobalRegistry().FindObject<StaticTestTrait>(
          stream_executor::cuda::kCudaPlatformId),
      IsOkAndHolds(142));
}

}  // namespace
}  // namespace stream_executor

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

#include "xla/stream_executor/gpu/gpu_kernel_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor::gpu {
namespace {

struct TestKernelTrait {
  using KernelType = TypedKernel<>;
};

struct OtherTestKernelTrait {
  using KernelType = TypedKernel<>;
};

using testing::Property;
using tsl::testing::IsOk;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

TEST(GpuKernelRegistryTest, RegisterKernel) {
  PlatformObjectRegistry object_registry;
  GpuKernelRegistry registry{&object_registry};
  KernelLoaderSpec cuda_spec =
      KernelLoaderSpec::CreateInProcessSymbolSpec(nullptr, "kernel_name", 1);
  KernelLoaderSpec rocm_spec =
      KernelLoaderSpec::CreateInProcessSymbolSpec(nullptr, "kernel_name", 42);

  // Can register a simple kernel
  EXPECT_THAT(registry.RegisterKernel<TestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId, cuda_spec),
              IsOk());

  // Can register another simple kernel - no clash
  EXPECT_THAT(registry.RegisterKernel<OtherTestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId, cuda_spec),
              IsOk());

  // Can register a different kernel under the same trait for a different
  // platform.
  EXPECT_THAT(registry.RegisterKernel<TestKernelTrait>(
                  stream_executor::rocm::kROCmPlatformId, rocm_spec),
              IsOk());

  // Can't register a kernel if it already exists in the registry.
  EXPECT_THAT(registry.RegisterKernel<TestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId, cuda_spec),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(GpuKernelRegistryTest, RegisterKernelConcurrently) {
  // This test will show races in the registry implementation when run with
  // `--config=tsan`.

  PlatformObjectRegistry object_registry;
  GpuKernelRegistry registry{&object_registry};

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&] {
    KernelLoaderSpec cuda_spec =
        KernelLoaderSpec::CreateInProcessSymbolSpec(nullptr, "kernel_name", 1);
    // Can register a simple kernel
    EXPECT_THAT(registry.RegisterKernel<TestKernelTrait>(
                    stream_executor::cuda::kCudaPlatformId, cuda_spec),
                IsOk());
  });

  pool.Schedule([&] {
    KernelLoaderSpec rocm_spec =
        KernelLoaderSpec::CreateInProcessSymbolSpec(nullptr, "kernel_name", 42);
    // Can register a different kernel under the same trait for a different
    // platform.
    EXPECT_THAT(registry.RegisterKernel<TestKernelTrait>(
                    stream_executor::rocm::kROCmPlatformId, rocm_spec),
                IsOk());
  });
}

TEST(GpuKernelRegistryTest, FindKernel) {
  PlatformObjectRegistry object_registry;
  GpuKernelRegistry registry{&object_registry};
  KernelLoaderSpec spec =
      KernelLoaderSpec::CreateInProcessSymbolSpec(nullptr, "kernel_name", 333);

  ASSERT_THAT(registry.RegisterKernel<TestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId, spec),
              IsOk());

  EXPECT_THAT(registry.FindKernel<TestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId),
              IsOkAndHolds(Property(&KernelLoaderSpec::arity, 333)));

  // No registered kernel for ROCM.
  EXPECT_THAT(registry.FindKernel<TestKernelTrait>(
                  stream_executor::rocm::kROCmPlatformId),
              StatusIs(absl::StatusCode::kNotFound));

  // No registered kernel for the other trait.
  EXPECT_THAT(registry.FindKernel<OtherTestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(GpuKernelRegistryTest, FindKernelConcurrently) {
  // This test will show races in the registry implementation when run with
  // `--config=tsan`.

  PlatformObjectRegistry object_registry;
  GpuKernelRegistry registry{&object_registry};
  KernelLoaderSpec spec =
      KernelLoaderSpec::CreateInProcessSymbolSpec(nullptr, "kernel_name", 333);

  ASSERT_THAT(registry.RegisterKernel<TestKernelTrait>(
                  stream_executor::cuda::kCudaPlatformId, spec),
              IsOk());

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&] {
    EXPECT_THAT(registry.FindKernel<TestKernelTrait>(
                    stream_executor::cuda::kCudaPlatformId),
                IsOkAndHolds(Property(&KernelLoaderSpec::arity, 333)));
  });

  pool.Schedule([&] {
    EXPECT_THAT(registry.FindKernel<TestKernelTrait>(
                    stream_executor::cuda::kCudaPlatformId),
                IsOkAndHolds(Property(&KernelLoaderSpec::arity, 333)));
  });
}

}  // namespace
}  // namespace stream_executor::gpu

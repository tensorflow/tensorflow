/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/locality_domain.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {
namespace {

using ::absl_testing::IsOk;
using ::testing::Each;

class LocalityDomainTest : public ::testing::Test {
 public:
  CudaExecutor* executor_;

 private:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(Platform * platform,
                         PlatformManager::PlatformWithId(
                             stream_executor::cuda::kCudaPlatformId));
    ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                         platform->ExecutorForDevice(0));
    executor_ = reinterpret_cast<CudaExecutor*>(executor);
  }
};

// The locality-domain count query reports at least one domain when it succeeds.
TEST_F(LocalityDomainTest, GetLocalityDomainCountReturnsAtLeastOne) {
  CUdevice device;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  absl::StatusOr<int> count_or = GetLocalityDomainCount(device);
  if (absl::IsUnimplemented(count_or.status())) {
    GTEST_SKIP() << count_or.status();
  }
  ASSERT_OK_AND_ASSIGN(int count, std::move(count_or));
  EXPECT_GE(count, 1);
}

// CreateLocalityDomains produces one domain per locality domain, in domain-id
// order, each backed by a non-empty SM partition.
TEST_F(LocalityDomainTest, CreateLocalityDomainsMatchesCountAndIds) {
  CUdevice device;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  absl::StatusOr<int> count_or = GetLocalityDomainCount(device);
  if (absl::IsUnimplemented(count_or.status())) {
    GTEST_SKIP() << count_or.status();
  }
  ASSERT_OK_AND_ASSIGN(int count, std::move(count_or));
  if (count <= 1) {
    GTEST_SKIP() << "Requires a multi-die GPU with more than one locality "
                    "domain";
  }

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(auto domains, CreateLocalityDomains(device));
  ASSERT_EQ(static_cast<int>(domains.size()), count);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(domains[i]->locality_domain_id(), i);
    EXPECT_GT(domains[i]->sm_count(), 0);
  }
}

// The domains partition the device's SMs: each has a positive SM count, their
// sum does not exceed the device total, and the domain's SM count agrees with
// its backing green context.
TEST_F(LocalityDomainTest, LocalityDomainSmPartition) {
  CUdevice device;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  absl::StatusOr<int> count_or = GetLocalityDomainCount(device);
  if (absl::IsUnimplemented(count_or.status())) {
    GTEST_SKIP() << count_or.status();
  }
  ASSERT_OK_AND_ASSIGN(int count, std::move(count_or));
  if (count <= 1) {
    GTEST_SKIP() << "Requires a multi-die GPU with more than one locality "
                    "domain";
  }

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(auto domains, CreateLocalityDomains(device));
  const int total_sms = executor_->GetDeviceDescription().core_count();
  ASSERT_GT(total_sms, 0);

  int sm_sum = 0;
  for (const auto& domain : domains) {
    EXPECT_GT(domain->sm_count(), 0);
    EXPECT_EQ(domain->green_context().sm_count(), domain->sm_count());
    sm_sum += domain->sm_count();
  }
  EXPECT_LE(sm_sum, total_sms);
}

// A locality domain can create a stream onto its SM partition.
TEST_F(LocalityDomainTest, LocalityDomainCreateStream) {
  CUdevice device;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  absl::StatusOr<int> count_or = GetLocalityDomainCount(device);
  if (absl::IsUnimplemented(count_or.status())) {
    GTEST_SKIP() << count_or.status();
  }
  ASSERT_OK_AND_ASSIGN(int count, std::move(count_or));
  if (count <= 1) {
    GTEST_SKIP() << "Requires a multi-die GPU with more than one locality "
                    "domain";
  }

  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  ASSERT_OK_AND_ASSIGN(auto domains, CreateLocalityDomains(device));
  ASSERT_FALSE(domains.empty());

  ASSERT_OK_AND_ASSIGN(CUstream stream,
                       domains[0]->CreateStream(/*priority=*/0));
  EXPECT_NE(stream, nullptr);
  EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
}

// On a multi-die GPU, each locality domain is exposed as its own green context
// and can independently launch a kernel. Exercises the executor-cached path
// (GetLocalityDomains / CreateStreamInLocalityDomain).
TEST_F(LocalityDomainTest, PerDomainKernelLaunch) {
  CUdevice device;
  ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
  absl::StatusOr<int> count_or = GetLocalityDomainCount(device);
  if (absl::IsUnimplemented(count_or.status())) {
    GTEST_SKIP() << count_or.status();
  }
  ASSERT_TRUE(count_or.ok()) << count_or.status();

  ASSERT_OK_AND_ASSIGN(
      absl::Span<const std::unique_ptr<LocalityDomain>> domains,
      executor_->GetLocalityDomains());
  if (domains.size() <= 1) {
    GTEST_SKIP() << "Device exposes " << domains.size()
                 << " locality domain(s); a multi-die GPU is required.";
  }

  ASSERT_OK_AND_ASSIGN(auto add, LoadAddI32TestKernel(executor_));
  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;

  for (int i = 0; i < static_cast<int>(domains.size()); ++i) {
    EXPECT_EQ(domains[i]->locality_domain_id(), i);
    EXPECT_GT(domains[i]->sm_count(), 0);

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                         executor_->CreateStreamInLocalityDomain(i));

    DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(kLength, 0);
    DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(kLength, 0);
    DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(kLength, 0);

    EXPECT_THAT(stream->Memset32(&a, 1, kByteLength), IsOk());
    EXPECT_THAT(stream->Memset32(&b, 2, kByteLength), IsOk());
    EXPECT_THAT(stream->MemZero(&c, kByteLength), IsOk());
    EXPECT_THAT(
        add.Launch(ThreadDim(), BlockDim(kLength), stream.get(), a, b, c),
        IsOk());

    std::array<int32_t, kLength> host;
    EXPECT_THAT(stream->MemcpyD2H(c, absl::MakeSpan(host)), IsOk());
    EXPECT_THAT(stream->BlockHostUntilDone(), IsOk());
    EXPECT_THAT(host, Each(3));
  }
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor

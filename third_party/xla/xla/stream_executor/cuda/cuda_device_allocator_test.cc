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

#include "xla/stream_executor/cuda/cuda_device_allocator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

CudaDeviceAllocator::Options MakeTestOptions(bool enable_rdma) {
  CudaDeviceAllocator::Options options;
  options.enable_rdma = enable_rdma;
  return options;
}

class CudaDeviceAllocatorTest : public ::testing::TestWithParam<bool> {};

TEST_P(CudaDeviceAllocatorTest, AllocateAndFree) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaDeviceAllocator> allocator,
      CudaDeviceAllocator::Create(executor, MakeTestOptions(GetParam())));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(1024));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_GE(allocation->address().size(), 1024);
}

TEST_P(CudaDeviceAllocatorTest, AllocateZeroBytes) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaDeviceAllocator> allocator,
      CudaDeviceAllocator::Create(executor, MakeTestOptions(GetParam())));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(0));
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(allocation->address().opaque(), nullptr);
}

TEST_P(CudaDeviceAllocatorTest, MemcpyRoundTrip) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaDeviceAllocator> allocator,
      CudaDeviceAllocator::Create(executor, MakeTestOptions(GetParam())));

  constexpr int kSize = 1024;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(kSize));

  // Write a pattern to host buffer and copy to device.
  std::vector<uint8_t> host_src(kSize);
  for (int i = 0; i < kSize; i++) {
    host_src[i] = static_cast<uint8_t>(i);
  }

  // Use a DeviceAddress sized to kSize (not the padded allocation size) so
  // the memcpy transfers exactly the bytes we care about.
  DeviceAddress<uint8_t> addr(
      DeviceAddressBase(allocation->address().opaque(), kSize));
  ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(host_src), &addr));

  // Copy back from device to host.
  std::vector<uint8_t> host_dst(kSize, 0);
  ASSERT_OK(stream->MemcpyD2H(addr, absl::MakeSpan(host_dst)));
  ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(host_src, host_dst);
}

TEST_P(CudaDeviceAllocatorTest, PeerAccessEnabled) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaDeviceAllocator::Options options = MakeTestOptions(GetParam());
  options.enable_peer_access = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CudaDeviceAllocator> allocator,
                       CudaDeviceAllocator::Create(executor, options));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(4096));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_GE(allocation->address().size(), 4096);
}

TEST_P(CudaDeviceAllocatorTest, DeallocateDuringStreamCapture) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CudaDeviceAllocator> allocator,
      CudaDeviceAllocator::Create(executor, MakeTestOptions(GetParam())));

  // Allocate two buffers prior to stream capture: one to be freed during
  // active stream capture, and one to be used inside the captured graph.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> alloc_to_free,
                       allocator->Allocate(4096));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> alloc_for_graph,
                       allocator->Allocate(1024));

  DeviceAddress<uint8_t> dst(
      DeviceAddressBase(alloc_for_graph->address().opaque(), 1024));

  // Trace a command buffer while freeing `alloc_to_free` mid-capture.
  // Without capture-aware queued VMM deallocation, `alloc_to_free.reset()`
  // calls `cuCtxSynchronize()` during active stream capture, which fails with
  // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED and invalidates the graph capture.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CommandBuffer> cmd_buf,
      TraceCommandBufferFactory::Create(
          executor, [&](Stream* capture_stream) -> absl::Status {
            alloc_to_free.reset();
            return capture_stream->MemZero(&dst, 1024);
          }));
  EXPECT_NE(cmd_buf, nullptr);
}

// Regress the 4 GiB exportable-allocation failure found on WSL. Run on all CUDA
// platforms to catch similar failures in other configurations, using POSIX-FD
// handles where supported. These handles allow sharing allocations between
// processes. Requesting them exercises the allocation mode that triggered the
// bug; disabling them could hide it and remove interprocess sharing support.
// Verify that each backing allocation preserves exportability after fallback.
// The extra byte covers a short final chunk when the fallback runs.
TEST(CudaDeviceAllocatorVmmTest, LargeAllocation) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  std::unique_ptr<ActivateContext> activation = executor->Activate();
  CUdevice device;
  ASSERT_OK(cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));
  int posix_fd_supported = 0;
  ASSERT_OK(cuda::ToStatus(cuDeviceGetAttribute(
      &posix_fd_supported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
      device)));
  CudaDeviceAllocator::Options options = MakeTestOptions(false);
  options.enable_posix_fd_handle = posix_fd_supported != 0;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CudaDeviceAllocator> allocator,
                       CudaDeviceAllocator::Create(executor, options));
  constexpr uint64_t kGiB = uint64_t{1} << 30;
  constexpr uint64_t kRequestedSize = 4 * kGiB + 1;
  constexpr uint32_t kPattern = 0x5a17c3e9;
  int64_t free_memory = 0;
  int64_t total_memory = 0;
  ASSERT_TRUE(executor->DeviceMemoryUsage(&free_memory, &total_memory));
  if (free_memory < static_cast<int64_t>(kRequestedSize + kGiB / 2)) {
    GTEST_SKIP() << "Need 4.5 GiB of free GPU memory, found " << free_memory;
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(kRequestedSize));
  DeviceAddressBase address = allocation->address();
  const uint64_t size = address.size();
  ASSERT_GE(size, kRequestedSize);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());
  ASSERT_OK(stream->Memset32(&address, kPattern, size));
  ASSERT_OK(stream->BlockHostUntilDone());

  char* base_ptr = static_cast<char*>(address.opaque());
  std::vector<uint64_t> offsets = {0, size - sizeof(uint32_t)};
  for (uint64_t boundary = kGiB; boundary < size; boundary += kGiB) {
    offsets.push_back(boundary - sizeof(uint32_t));
    offsets.push_back(boundary);
  }
  for (uint64_t offset : offsets) {
    SCOPED_TRACE(offset);
    void* offset_ptr = base_ptr + offset;
    uint32_t actual = 0;
    ASSERT_OK(cuda::ToStatus(cuMemcpyDtoH(
        &actual, absl::bit_cast<CUdeviceptr>(offset_ptr), sizeof(actual))));
    EXPECT_EQ(actual, kPattern);

    CUmemGenericAllocationHandle handle = 0;
    ASSERT_OK(cuda::ToStatus(cuMemRetainAllocationHandle(&handle, offset_ptr)));
    CUmemAllocationProp properties = {};
    CUresult result =
        cuMemGetAllocationPropertiesFromHandle(&properties, handle);
    EXPECT_OK(cuda::ToStatus(cuMemRelease(handle)));
    ASSERT_OK(cuda::ToStatus(result));
    EXPECT_EQ((properties.requestedHandleTypes &
               CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0,
              options.enable_posix_fd_handle);
  }
}

INSTANTIATE_TEST_SUITE_P(RdmaSupport, CudaDeviceAllocatorTest,
                         ::testing::Bool(), [](const auto& info) {
                           return info.param ? "RdmaEnabled" : "RdmaDisabled";
                         });

TEST(CudaDeviceAllocatorNonVmmTest, AllocateMemcpyAndFreeWithoutVmm) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());

  CudaDeviceAllocator::Options options;
  options.use_vmm = false;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CudaDeviceAllocator> allocator,
                       CudaDeviceAllocator::Create(executor, options));

  constexpr int kSize = 1024;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                       allocator->Allocate(kSize));
  ASSERT_NE(allocation, nullptr);
  EXPECT_NE(allocation->address().opaque(), nullptr);
  EXPECT_EQ(allocation->address().size(), kSize);

  std::vector<uint8_t> host_src(kSize);
  for (int i = 0; i < kSize; i++) {
    host_src[i] = static_cast<uint8_t>(i);
  }

  DeviceAddress<uint8_t> addr(
      DeviceAddressBase(allocation->address().opaque(), kSize));
  ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(host_src), &addr));

  std::vector<uint8_t> host_dst(kSize, 0);
  ASSERT_OK(stream->MemcpyD2H(addr, absl::MakeSpan(host_dst)));
  ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(host_src, host_dst);
}

TEST(CudaDeviceAllocatorLocalityDomainTest, RejectsNegativeLocalityDomain) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  for (int locality_domain_id : {-1, -2}) {
    CudaDeviceAllocator::Options options;
    options.locality_domain_id = locality_domain_id;
    EXPECT_THAT(CudaDeviceAllocator::Create(executor, options),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument))
        << "locality_domain_id=" << locality_domain_id;
  }
}

TEST(CudaDeviceAllocatorLocalityDomainTest, LocalityDomainRequiresCuda134) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const DeviceDescription& device = executor->GetDeviceDescription();
  if (std::min(device.driver_version(),
               device.compile_time_toolkit_version()) >=
      SemanticVersion{13, 4, 0}) {
    GTEST_SKIP() << "Requires a CUDA toolkit or driver older than 13.4";
  }

  CudaDeviceAllocator::Options options;
  options.locality_domain_id = 0;
  EXPECT_THAT(CudaDeviceAllocator::Create(executor, options),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(CudaDeviceAllocatorLocalityDomainTest, LocalityDomainRequiresVmm) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));

  CudaDeviceAllocator::Options options;
  options.use_vmm = false;
  options.locality_domain_id = 0;
  EXPECT_THAT(CudaDeviceAllocator::Create(executor, options),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CudaDeviceAllocatorLocalityDomainTest,
     RejectsLocalityDomainBeyondDeviceCount) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const int domain_count =
      executor->GetDeviceDescription().locality_domain_count();
  if (domain_count < 2) {
    GTEST_SKIP() << "Requires a GPU with at least two locality domains";
  }

  for (int locality_domain_id : {domain_count, 255}) {
    CudaDeviceAllocator::Options options;
    options.locality_domain_id = locality_domain_id;
    EXPECT_THAT(CudaDeviceAllocator::Create(executor, options),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument))
        << "locality_domain_id=" << locality_domain_id;
  }
}

TEST(CudaDeviceAllocatorLocalityDomainTest, PlacesMemoryOnEachDomain) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("CUDA"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  const int domain_count =
      executor->GetDeviceDescription().locality_domain_count();
  if (domain_count < 2) {
    GTEST_SKIP() << "Requires a GPU with at least two locality domains";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());

  constexpr int kSize = 4096;
  for (int locality_domain_id = 0; locality_domain_id < domain_count;
       ++locality_domain_id) {
    CudaDeviceAllocator::Options options;
    options.locality_domain_id = locality_domain_id;
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<CudaDeviceAllocator> allocator,
                         CudaDeviceAllocator::Create(executor, options));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                         allocator->Allocate(kSize));
    ASSERT_NE(allocation->address().opaque(), nullptr);
    EXPECT_GE(allocation->address().size(), kSize);

#if CUDA_VERSION >= 13040
    {
      std::unique_ptr<ActivateContext> activation = executor->Activate();
      CUmemGenericAllocationHandle handle;
      ASSERT_EQ(
          cuMemRetainAllocationHandle(&handle, allocation->address().opaque()),
          CUDA_SUCCESS);
      CUmemAllocationProp properties = {};
      EXPECT_EQ(cuMemGetAllocationPropertiesFromHandle(&properties, handle),
                CUDA_SUCCESS);
      EXPECT_EQ(cuMemRelease(handle), CUDA_SUCCESS);
      EXPECT_EQ(properties.location.type,
                CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN);
      EXPECT_EQ(properties.location.localized.localityDomainId,
                locality_domain_id);
    }
#endif  // CUDA_VERSION >= 13040

    std::vector<uint8_t> host_src(kSize);
    for (int i = 0; i < kSize; ++i) {
      host_src[i] = static_cast<uint8_t>(i + locality_domain_id);
    }
    DeviceAddress<uint8_t> addr(
        DeviceAddressBase(allocation->address().opaque(), kSize));
    ASSERT_OK(stream->MemcpyH2D(absl::MakeConstSpan(host_src), &addr));
    std::vector<uint8_t> host_dst(kSize, 0);
    ASSERT_OK(stream->MemcpyD2H(addr, absl::MakeSpan(host_dst)));
    ASSERT_OK(stream->BlockHostUntilDone());
    EXPECT_EQ(host_src, host_dst);
  }
}

}  // namespace
}  // namespace stream_executor::gpu

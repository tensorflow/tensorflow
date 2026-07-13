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

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_memory_reservation.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

// 1 MB — will be rounded up to the VMM granularity (typically 2 MB).
static constexpr uint64_t kTestSize = 1024 * 1024;

class CudaMemoryReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        platform_, PlatformManager::PlatformWithId(cuda::kCudaPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
  }

  Platform* platform_ = nullptr;
  StreamExecutor* executor_ = nullptr;
};

// Verifies that MapTo grants read/write access to all P2P-capable peer devices,
// not just the local device. Without this, NVLink P2P accesses to VA-mapped
// buffers deadlock during NCCL collective operations (AllGather, ReduceScatter,
// AllReduce).
TEST_F(CudaMemoryReservationTest, SetAccessGrantsPeerDeviceAccess) {
  if (platform_->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs, found "
                 << platform_->VisibleDeviceCount();
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, CudaRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));

  CUdeviceptr base_ptr = reinterpret_cast<CUdeviceptr>(res->address().opaque());
  for (int32_t peer = 0; peer < platform_->VisibleDeviceCount(); ++peer) {
    if (!executor_->CanEnablePeerAccessTo(peer)) continue;
    CUmemLocation loc = {};
    loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
    loc.id = peer;
    uint64_t flags = 0;
    ASSERT_EQ(
        cuMemGetAccess(reinterpret_cast<unsigned long long*>(&flags),  // NOLINT
                       &loc, base_ptr),
        CUDA_SUCCESS)
        << "cuMemGetAccess failed for peer device " << peer;
    EXPECT_EQ(flags, static_cast<uint64_t>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE))
        << "Expected READWRITE access on peer device " << peer;
  }
}

}  // namespace
}  // namespace stream_executor::gpu

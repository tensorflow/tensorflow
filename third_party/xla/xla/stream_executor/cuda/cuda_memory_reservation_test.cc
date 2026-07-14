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

#include "xla/stream_executor/cuda/cuda_memory_reservation.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;

// 1 MB — will be rounded up to the VMM granularity (typically 2 MB).
static constexpr uint64_t kTestSize = 1024 * 1024;

// A minimal MemoryAllocation that is NOT a CudaRawMemoryAllocation, used to
// exercise the wrong-type error path in MapTo.
class FakeAllocation : public MemoryAllocation {
 public:
  DeviceAddressBase address() const override { return DeviceAddressBase(); }
};

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

// Verifies that Create reserves a non-null virtual address range of at least
// the requested size.
TEST_F(CudaMemoryReservationTest, CreateReservation) {
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  EXPECT_NE(res->address().opaque(), nullptr);
  EXPECT_GE(res->address().size(), kTestSize);
}

// Verifies that passing a non-CudaRawMemoryAllocation to MapTo returns an
// InvalidArgument error.
TEST_F(CudaMemoryReservationTest, MapToWrongType) {
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  FakeAllocation fake;
  EXPECT_THAT(res->MapTo(0, 0, kTestSize, fake),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Verifies the full MapTo workflow. The ScopedMapping is destroyed first,
// unmapping the reservation range, then the allocation is released.
TEST_F(CudaMemoryReservationTest, MapToSingleAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, CudaRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));

  // The mapped address starts at the reservation base (offset 0) and spans
  // the full allocation size.
  EXPECT_EQ(mapping.mapped_address().opaque(), res->address().opaque());
  EXPECT_EQ(mapping.mapped_address().size(), alloc_size);
  // ScopedMapping destructor: cuMemUnmap.
  // CudaMemoryReservation destructor: cuMemUnmap (logs error, already unmapped)
  // + cuMemAddressFree. Allocation destructor: cuMemRelease.
}

// Verifies that ScopedMapping unmaps the range on destruction, allowing a
// second mapping into the same reservation range.
TEST_F(CudaMemoryReservationTest, ScopedMappingUnmapsOnDestruction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, CudaRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  {
    TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));
    // mapping goes out of scope here, triggering cuMemUnmap.
  }

  // After the ScopedMapping is destroyed, the range is unmapped and can be
  // remapped.
  TF_ASSERT_OK_AND_ASSIGN(auto mapping2, res->MapTo(0, 0, alloc_size, *alloc));
  EXPECT_NE(mapping2.mapped_address().opaque(), nullptr);
}

// Verifies that multiple physical allocations can be mapped into a contiguous
// reservation range via the span-based MapTo overload.
TEST_F(CudaMemoryReservationTest, MapToMultipleAllocations) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc1, CudaRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc2, CudaRawMemoryAllocation::Create(executor_, kTestSize));

  const size_t size1 = alloc1->address().size();
  const size_t size2 = alloc2->address().size();

  // Reserve enough virtual space for both allocations.
  TF_ASSERT_OK_AND_ASSIGN(
      auto res, CudaMemoryReservation::Create(executor_, size1 + size2));
  ASSERT_GE(res->address().size(), size1 + size2);

  MemoryReservation::MappingDescriptor descs[] = {
      {/*reservation_offset=*/0, /*allocation_offset=*/0, size1, alloc1.get()},
      {/*reservation_offset=*/size1, /*allocation_offset=*/0, size2,
       alloc2.get()},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(absl::MakeSpan(descs)));

  // The contiguous mapping starts at the reservation base and spans both
  // allocations.
  EXPECT_EQ(mapping.mapped_address().opaque(), res->address().opaque());
  EXPECT_EQ(mapping.mapped_address().size(), size1 + size2);
}

// Verifies that a second reservation does not alias the first.
TEST_F(CudaMemoryReservationTest, TwoReservationsDifferentAddresses) {
  TF_ASSERT_OK_AND_ASSIGN(auto res1,
                          CudaMemoryReservation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res2,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  EXPECT_NE(res1->address().opaque(), res2->address().opaque());
}

// Verifies that MapTo grants read/write access to the local device via
// cuMemSetAccess, readable back via cuMemGetAccess.
TEST_F(CudaMemoryReservationTest, SetAccessGrantsLocalDeviceAccess) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, CudaRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          CudaMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));

  CUmemLocation loc = {};
  loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
  loc.id = executor_->device_ordinal();
  uint64_t flags = 0;
  CUdeviceptr base_ptr = reinterpret_cast<CUdeviceptr>(res->address().opaque());
  ASSERT_EQ(
      cuMemGetAccess(reinterpret_cast<unsigned long long*>(&flags),  // NOLINT
                     &loc, base_ptr),
      CUDA_SUCCESS);
  EXPECT_EQ(flags, static_cast<uint64_t>(CU_MEM_ACCESS_FLAGS_PROT_READWRITE));
}

}  // namespace
}  // namespace stream_executor::gpu

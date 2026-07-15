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

#include "xla/stream_executor/rocm/rocm_memory_reservation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;

static constexpr uint64_t kTestSize = 1024 * 1024;

class FakeAllocation : public MemoryAllocation {
 public:
  DeviceAddressBase address() const override { return DeviceAddressBase(); }
};

class RocmMemoryReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto platform_or = PlatformManager::PlatformWithName("ROCM");
    if (!platform_or.ok()) {
      GTEST_SKIP() << "ROCM platform not available";
    }
    auto executor_or = platform_or.value()->ExecutorForDevice(0);
    if (!executor_or.ok()) {
      GTEST_SKIP() << "ROCM executor not available: " << executor_or.status();
    }
    executor_ = executor_or.value();
  }

  StreamExecutor* executor_ = nullptr;
};

TEST_F(RocmMemoryReservationTest, CreateReservation) {
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  EXPECT_NE(res->address().opaque(), nullptr);
  EXPECT_GE(res->address().size(), kTestSize);
}

TEST_F(RocmMemoryReservationTest, MapToWrongType) {
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  FakeAllocation fake;
  EXPECT_THAT(res->MapTo(0, 0, kTestSize, fake),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(RocmMemoryReservationTest, MapToSingleAllocation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));

  EXPECT_EQ(mapping.mapped_address().opaque(), res->address().opaque());
  EXPECT_EQ(mapping.mapped_address().size(), alloc_size);
}

TEST_F(RocmMemoryReservationTest, ScopedMappingUnmapsOnDestruction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  const size_t alloc_size = alloc->address().size();
  {
    TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(0, 0, alloc_size, *alloc));
  }

  TF_ASSERT_OK_AND_ASSIGN(auto mapping2, res->MapTo(0, 0, alloc_size, *alloc));
  EXPECT_NE(mapping2.mapped_address().opaque(), nullptr);
}

TEST_F(RocmMemoryReservationTest, MapToMultipleAllocations) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc1, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc2, RocmRawMemoryAllocation::Create(executor_, kTestSize));

  const size_t size1 = alloc1->address().size();
  const size_t size2 = alloc2->address().size();

  TF_ASSERT_OK_AND_ASSIGN(
      auto res, RocmMemoryReservation::Create(executor_, size1 + size2));
  ASSERT_GE(res->address().size(), size1 + size2);

  MemoryReservation::MappingDescriptor descs[] = {
      {/*reservation_offset=*/0, /*allocation_offset=*/0, size1, alloc1.get()},
      {/*reservation_offset=*/size1, /*allocation_offset=*/0, size2,
       alloc2.get()},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(absl::MakeSpan(descs)));

  EXPECT_EQ(mapping.mapped_address().opaque(), res->address().opaque());
  EXPECT_EQ(mapping.mapped_address().size(), size1 + size2);
}

TEST_F(RocmMemoryReservationTest, TwoReservationsDifferentAddresses) {
  TF_ASSERT_OK_AND_ASSIGN(auto res1,
                          RocmMemoryReservation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(auto res2,
                          RocmMemoryReservation::Create(executor_, kTestSize));

  EXPECT_NE(res1->address().opaque(), res2->address().opaque());
}

// Remap with every slice marked remap_required=false must leave the existing
// physical backing (and therefore the data) untouched.
TEST_F(RocmMemoryReservationTest, RemapPreservesUnchangedSlices) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc_a, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc_b, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  const size_t sa = alloc_a->address().size();
  const size_t sb = alloc_b->address().size();

  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, sa + sb));

  MemoryReservation::MappingDescriptor descs[] = {
      {/*reservation_offset=*/0, /*allocation_offset=*/0, sa, alloc_a.get()},
      {/*reservation_offset=*/sa, /*allocation_offset=*/0, sb, alloc_b.get()},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(absl::MakeSpan(descs)));

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
  void* const base = res->address().opaque();
  DeviceAddressBase slice0(base, sizeof(uint64_t));
  DeviceAddressBase slice1(reinterpret_cast<uint8_t*>(base) + sa,
                           sizeof(uint64_t));

  constexpr uint64_t kPatternA = 0xAAAAAAAAAAAAAAAAULL;
  constexpr uint64_t kPatternB = 0xBBBBBBBBBBBBBBBBULL;
  ASSERT_THAT(stream->Memcpy(&slice0, &kPatternA, sizeof(kPatternA)), IsOk());
  ASSERT_THAT(stream->Memcpy(&slice1, &kPatternB, sizeof(kPatternB)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  MemoryReservation::RemappingDescriptor remaps[] = {
      {0, 0, sa, alloc_a.get(), /*remap_required=*/false},
      {sa, 0, sb, alloc_b.get(), /*remap_required=*/false},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping2,
                          std::move(mapping).Remap(absl::MakeSpan(remaps)));
  EXPECT_EQ(mapping2.mapped_address().opaque(), base);
  EXPECT_EQ(mapping2.mapped_address().size(), sa + sb);

  uint64_t v0 = 0, v1 = 0;
  ASSERT_THAT(stream->Memcpy(&v0, slice0, sizeof(v0)), IsOk());
  ASSERT_THAT(stream->Memcpy(&v1, slice1, sizeof(v1)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());
  EXPECT_EQ(v0, kPatternA);
  EXPECT_EQ(v1, kPatternB);
}

// Remap with remap_required=true for a slice must repoint that slice at the
// new physical allocation while preserving the slices left unchanged.
TEST_F(RocmMemoryReservationTest, RemapRepointsRequiredSlice) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc_a, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  TF_ASSERT_OK_AND_ASSIGN(
      auto alloc_b, RocmRawMemoryAllocation::Create(executor_, kTestSize));
  const size_t sa = alloc_a->address().size();
  const size_t sb = alloc_b->address().size();

  TF_ASSERT_OK_AND_ASSIGN(auto res,
                          RocmMemoryReservation::Create(executor_, sa + sb));

  MemoryReservation::MappingDescriptor descs[] = {
      {/*reservation_offset=*/0, /*allocation_offset=*/0, sa, alloc_a.get()},
      {/*reservation_offset=*/sa, /*allocation_offset=*/0, sb, alloc_b.get()},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping, res->MapTo(absl::MakeSpan(descs)));

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor_->CreateStream());
  void* const base = res->address().opaque();
  DeviceAddressBase slice0(base, sizeof(uint64_t));
  DeviceAddressBase slice1(reinterpret_cast<uint8_t*>(base) + sa,
                           sizeof(uint64_t));

  constexpr uint64_t kPatternA = 0xAAAAAAAAAAAAAAAAULL;
  constexpr uint64_t kPatternB = 0xBBBBBBBBBBBBBBBBULL;
  ASSERT_THAT(stream->Memcpy(&slice0, &kPatternA, sizeof(kPatternA)), IsOk());
  ASSERT_THAT(stream->Memcpy(&slice1, &kPatternB, sizeof(kPatternB)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());

  // Keep slice0 on alloc_a; repoint slice1 to alloc_a as well. After the remap
  // slice1 aliases alloc_a, so reading it must observe slice0's data
  // (kPatternA) rather than the kPatternB it previously held via alloc_b.
  MemoryReservation::RemappingDescriptor remaps[] = {
      {0, 0, sa, alloc_a.get(), /*remap_required=*/false},
      {sa, 0, sb, alloc_a.get(), /*remap_required=*/true},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto mapping2,
                          std::move(mapping).Remap(absl::MakeSpan(remaps)));
  EXPECT_EQ(mapping2.mapped_address().opaque(), base);
  EXPECT_EQ(mapping2.mapped_address().size(), sa + sb);

  uint64_t v0 = 0, v1 = 0;
  ASSERT_THAT(stream->Memcpy(&v0, slice0, sizeof(v0)), IsOk());
  ASSERT_THAT(stream->Memcpy(&v1, slice1, sizeof(v1)), IsOk());
  ASSERT_THAT(stream->BlockHostUntilDone(), IsOk());
  EXPECT_EQ(v0, kPatternA);
  EXPECT_EQ(v1, kPatternA);
}

}  // namespace
}  // namespace stream_executor::gpu

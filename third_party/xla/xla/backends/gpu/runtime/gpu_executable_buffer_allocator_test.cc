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

#include "xla/backends/gpu/runtime/gpu_executable_buffer_allocator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/buffer_allocations.h"
#include "xla/backends/gpu/runtime/gpu_constants.h"
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Pair;
using ::testing::Return;

constexpr int64_t kDefault = static_cast<int64_t>(MemorySpaceColor::kDefault);
constexpr int64_t kCollective =
    static_cast<int64_t>(MemorySpaceColor::kCollective);
constexpr int64_t kTempBuffer =
    static_cast<int64_t>(MemorySpaceColor::kTempBuffer);

// Host-backed buffers must satisfy the alignment check on every allocation
// kind.
constexpr uint64_t kAlignment = kXlaAllocatedBufferAlignBytes;

// Serves every request from aligned host memory and records the memory space
// and size of each request in the order it was made.
class RecordingAllocator final : public se::DeviceAddressAllocator {
 public:
  RecordingAllocator(const se::Platform* platform, se::Stream* stream)
      : se::DeviceAddressAllocator(platform), stream_(stream) {}

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    if (failing_memory_space_.has_value() &&
        *failing_memory_space_ == memory_space) {
      return absl::ResourceExhaustedError("injected allocation failure");
    }
    requests_.push_back({memory_space, size});
    auto storage = std::make_unique<uint8_t[]>(size + kAlignment);
    uint8_t* aligned = reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(storage.get()) + kAlignment - 1) &
        ~uintptr_t{kAlignment - 1});
    storage_.push_back(std::move(storage));
    return se::ScopedDeviceAddress<uint8_t>(
        se::DeviceAddressBase(aligned, size), device_ordinal, this);
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase mem) override {
    if (!mem.is_null()) {
      ++num_deallocations_;
    }
    return absl::OkStatus();
  }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return stream_;
  }

  // Makes every subsequent request for `memory_space` fail.
  void FailRequestsFor(int64_t memory_space) {
    failing_memory_space_ = memory_space;
  }

  // (memory_space, size) of every successful request, in request order.
  const std::vector<std::pair<int64_t, uint64_t>>& requests() const {
    return requests_;
  }

  int num_deallocations() const { return num_deallocations_; }

 private:
  se::Stream* stream_;
  std::optional<int64_t> failing_memory_space_;
  std::vector<std::pair<int64_t, uint64_t>> requests_;
  std::vector<std::unique_ptr<uint8_t[]>> storage_;
  int num_deallocations_ = 0;
};

class GpuExecutableBufferAllocatorTest : public ::testing::Test {
 protected:
  GpuExecutableBufferAllocatorTest() : allocator_(&platform_, &stream_) {
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    run_options_.set_stream(&stream_);
    service_run_options_ = ServiceExecutableRunOptions(run_options_);
  }

  // Builds the BufferAllocations for one execution of `allocations` with the
  // base (ALWAYS_UPDATE) buffer allocator and the recording device allocator.
  absl::StatusOr<BufferAllocations> Generate(
      absl::Span<const BufferAllocation* const> allocations) {
    GpuExecutableBufferAllocator executable_allocator(
        "test", allocations, ShapeUtil::MakeShape(F32, {64}), &debug_options_,
        /*thunk_executor=*/nullptr);
    auto get_parameter_buffer = [&](const BufferAllocation& allocation)
        -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
      return GpuExecutableBufferAllocator::ParameterBuffer{
          se::DeviceAddressBase(parameter_storage_, sizeof(parameter_storage_)),
          allocation.parameter_number()};
    };
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        executable_allocator.CreateExecutionScope(&service_run_options_,
                                                  &allocator_,
                                                  /*device_ordinal=*/0));
    return scope->GenerateBufferAllocations(
        &service_run_options_, get_parameter_buffer, &globals_, &allocator_);
  }

  NiceMock<se::MockPlatform> platform_;
  NiceMock<se::MockStreamExecutor> executor_;
  NiceMock<se::MockStream> stream_;
  RecordingAllocator allocator_;
  ExecutableRunOptions run_options_;
  ServiceExecutableRunOptions service_run_options_;
  DebugOptions debug_options_;
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals_;
  alignas(kAlignment) uint8_t parameter_storage_[1024];
  alignas(kAlignment) uint8_t constant_storage_[256];
};

TEST_F(GpuExecutableBufferAllocatorTest,
       AllocatesCollectiveMemoryBeforeOtherTransientBuffers) {
  BufferAllocation parameter(/*index=*/0, /*size=*/1024, kDefault);
  parameter.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  BufferAllocation default_temp(/*index=*/1, /*size=*/1024, kDefault);
  BufferAllocation collective_temp(/*index=*/2, /*size=*/2048, kCollective);
  BufferAllocation constant(/*index=*/3, /*size=*/256, kDefault);
  constant.set_constant(true);
  BufferAllocation collective_output(/*index=*/4, /*size=*/512, kCollective);
  collective_output.set_maybe_live_out(true);
  BufferAllocation default_output(/*index=*/5, /*size=*/768, kDefault);
  default_output.set_maybe_live_out(true);
  BufferAllocation thread_local_buffer(/*index=*/6, /*size=*/128, kDefault);
  thread_local_buffer.set_is_thread_local(true);
  BufferAllocation separate_temp(/*index=*/7, /*size=*/64, kTempBuffer);
  std::vector<const BufferAllocation*> allocations = {
      &parameter,           &default_temp,
      &collective_temp,     &constant,
      &collective_output,   &default_output,
      &thread_local_buffer, &separate_temp};
  globals_[3] = se::DeviceAddressBase(constant_storage_, 256);

  ASSERT_OK_AND_ASSIGN(BufferAllocations buffers, Generate(allocations));

  // Every collective request precedes every other request, and each group
  // keeps allocation index order.
  EXPECT_THAT(allocator_.requests(),
              ElementsAre(Pair(kCollective, 2048), Pair(kCollective, 512),
                          Pair(kDefault, 1024), Pair(kDefault, 768),
                          Pair(kTempBuffer, 64)));

  // The result is still indexed by allocation index.
  EXPECT_EQ(buffers.GetDeviceAddress(0).opaque(),
            static_cast<void*>(parameter_storage_));
  EXPECT_EQ(buffers.GetDeviceAddress(1).size(), 1024);
  EXPECT_EQ(buffers.GetDeviceAddress(2).size(), 2048);
  EXPECT_EQ(buffers.GetDeviceAddress(3).opaque(),
            static_cast<void*>(constant_storage_));
  EXPECT_EQ(buffers.GetDeviceAddress(4).size(), 512);
  EXPECT_EQ(buffers.GetDeviceAddress(5).size(), 768);
  EXPECT_TRUE(buffers.GetDeviceAddress(6).is_null());
  EXPECT_EQ(buffers.GetDeviceAddress(7).size(), 64);

  ASSERT_OK(buffers.TearDown(/*live_addresses=*/{}, allocations));
  EXPECT_EQ(allocator_.num_deallocations(), 5);
}

TEST_F(GpuExecutableBufferAllocatorTest,
       KeepsAllocationIndexOrderWithoutCollectiveMemory) {
  BufferAllocation first(/*index=*/0, /*size=*/256, kDefault);
  BufferAllocation second(/*index=*/1, /*size=*/512, kDefault);
  BufferAllocation third(/*index=*/2, /*size=*/1024, kDefault);
  std::vector<const BufferAllocation*> allocations = {&first, &second, &third};

  ASSERT_OK_AND_ASSIGN(BufferAllocations buffers, Generate(allocations));

  EXPECT_THAT(allocator_.requests(),
              ElementsAre(Pair(kDefault, 256), Pair(kDefault, 512),
                          Pair(kDefault, 1024)));
  ASSERT_OK(buffers.TearDown(/*live_addresses=*/{}, allocations));
}

TEST_F(GpuExecutableBufferAllocatorTest,
       ReturnsDefaultMemoryFailureAfterCollectiveMemoryIsAllocated) {
  allocator_.FailRequestsFor(kDefault);
  BufferAllocation default_temp(/*index=*/0, /*size=*/1024, kDefault);
  BufferAllocation collective_temp(/*index=*/1, /*size=*/2048, kCollective);
  std::vector<const BufferAllocation*> allocations = {&default_temp,
                                                      &collective_temp};

  EXPECT_THAT(Generate(allocations).status(),
              StatusIs(absl::StatusCode::kResourceExhausted));
  // The collective buffer was requested before the default one failed.
  EXPECT_THAT(allocator_.requests(), ElementsAre(Pair(kCollective, 2048)));
}

TEST_F(GpuExecutableBufferAllocatorTest,
       AllocatesAndInitializesPersistentBufferOnceAndOutlivesStepAllocator) {
  BufferAllocation persistent(/*index=*/0, /*size=*/16, kDefault);
  persistent.set_persistent(true);
  persistent.set_initial_data(
      {0x95, 0x23, 0x01, 0x07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  BufferAllocation zero_init_persistent(/*index=*/1, /*size=*/32, kDefault);
  zero_init_persistent.set_persistent(true);
  BufferAllocation temp(/*index=*/2, /*size=*/64, kDefault);
  std::vector<const BufferAllocation*> allocations = {
      &persistent, &zero_init_persistent, &temp};

  EXPECT_FALSE(persistent.IsPreallocatedTempBuffer());
  EXPECT_FALSE(persistent.is_readonly());

  alignas(kAlignment) uint8_t persistent_storage_0[16]{};
  alignas(kAlignment) uint8_t persistent_storage_1[32]{};
  EXPECT_CALL(executor_, Allocate(16, kDefault))
      .WillOnce(Return(se::DeviceAddressBase(persistent_storage_0, 16)));
  EXPECT_CALL(executor_, Allocate(32, kDefault))
      .WillOnce(Return(se::DeviceAddressBase(persistent_storage_1, 32)));

  int executor_deallocations = 0;
  EXPECT_CALL(executor_, Deallocate(_))
      .Times(2)
      .WillRepeatedly([&](se::DeviceAddressBase* mem) {
        if (!mem->is_null()) {
          ++executor_deallocations;
        }
      });

  std::vector<uint8_t> copied_bytes;
  EXPECT_CALL(stream_, Memcpy(_, persistent.initial_data().data(), 16))
      .WillOnce([&](se::DeviceAddressBase*, const void* src, uint64_t size) {
        const auto* bytes = static_cast<const uint8_t*>(src);
        copied_bytes.assign(bytes, bytes + size);
        return absl::OkStatus();
      });
  EXPECT_CALL(stream_, MemZero(_, 32)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream_, BlockHostUntilDone()).WillOnce(Return(absl::OkStatus()));

  {
    GpuExecutableBufferAllocator executable_allocator(
        "test", allocations, ShapeUtil::MakeShape(F32, {64}), &debug_options_,
        /*thunk_executor=*/nullptr, /*num_devices=*/1);
    auto no_params = [&](const BufferAllocation&)
        -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
      return absl::InternalError("unexpected parameter");
    };

    se::DeviceAddressBase run1_persistent_0;
    se::DeviceAddressBase run1_persistent_1;
    {
      // Simulate a per step DeviceAddressAllocator (such as TF XLA's
      // TfAllocatorAdapter) that is destroyed before executable_allocator.
      RecordingAllocator step1_allocator(&platform_, &stream_);
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope1,
          executable_allocator.CreateExecutionScope(&service_run_options_,
                                                    &step1_allocator,
                                                    /*device_ordinal=*/0));
      ASSERT_OK_AND_ASSIGN(
          BufferAllocations buffers1,
          scope1->GenerateBufferAllocations(&service_run_options_, no_params,
                                            &globals_, &step1_allocator));
      run1_persistent_0 = buffers1.GetDeviceAddress(0);
      run1_persistent_1 = buffers1.GetDeviceAddress(1);
      EXPECT_THAT(step1_allocator.requests(), ElementsAre(Pair(kDefault, 64)));
      EXPECT_OK(scope1->TearDown(/*live_addresses=*/{}, buffers1));
      EXPECT_EQ(step1_allocator.num_deallocations(), 1);
    }

    EXPECT_EQ(executor_deallocations, 0);

    {
      RecordingAllocator step2_allocator(&platform_, &stream_);
      ASSERT_OK_AND_ASSIGN(
          std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope2,
          executable_allocator.CreateExecutionScope(&service_run_options_,
                                                    &step2_allocator,
                                                    /*device_ordinal=*/0));
      ASSERT_OK_AND_ASSIGN(
          BufferAllocations buffers2,
          scope2->GenerateBufferAllocations(&service_run_options_, no_params,
                                            &globals_, &step2_allocator));
      EXPECT_EQ(run1_persistent_0.opaque(),
                buffers2.GetDeviceAddress(0).opaque());
      EXPECT_EQ(run1_persistent_1.opaque(),
                buffers2.GetDeviceAddress(1).opaque());
      EXPECT_THAT(step2_allocator.requests(), ElementsAre(Pair(kDefault, 64)));
      EXPECT_OK(scope2->TearDown(/*live_addresses=*/{}, buffers2));
      EXPECT_EQ(step2_allocator.num_deallocations(), 1);
    }

    EXPECT_EQ(executor_deallocations, 0);
  }

  // Persistent allocations are freed via StreamExecutor only when
  // executable_allocator goes out of scope.
  EXPECT_THAT(copied_bytes, ElementsAre(0x95, 0x23, 0x01, 0x07, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0));
  EXPECT_EQ(executor_deallocations, 2);
}

TEST_F(GpuExecutableBufferAllocatorTest,
       ReturnsResourceExhaustedWhenPersistentBufferAllocationFails) {
  BufferAllocation persistent(/*index=*/0, /*size=*/16, kDefault);
  persistent.set_persistent(true);
  std::vector<const BufferAllocation*> allocations = {&persistent};

  EXPECT_CALL(executor_, Allocate(16, kDefault))
      .WillOnce(Return(se::DeviceAddressBase()));

  EXPECT_THAT(Generate(allocations).status(),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("persistent buffer allocation 0")));
}

TEST_F(GpuExecutableBufferAllocatorTest,
       RetriesPersistentInitializationAfterAllocationFailure) {
  BufferAllocation first(/*index=*/0, /*size=*/16, kDefault);
  first.set_persistent(true);
  first.set_initial_data(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  BufferAllocation second(/*index=*/1, /*size=*/32, kDefault);
  second.set_persistent(true);
  std::vector<const BufferAllocation*> allocations = {&first, &second};

  alignas(kAlignment) uint8_t storage_0[16]{};
  alignas(kAlignment) uint8_t storage_1[32]{};
  EXPECT_CALL(executor_, Allocate(16, kDefault))
      .Times(2)
      .WillRepeatedly(Return(se::DeviceAddressBase(storage_0, 16)));
  EXPECT_CALL(executor_, Allocate(32, kDefault))
      .WillOnce(Return(se::DeviceAddressBase()))
      .WillOnce(Return(se::DeviceAddressBase(storage_1, 32)));
  int executor_deallocations = 0;
  ON_CALL(executor_, Deallocate(_))
      .WillByDefault([&](se::DeviceAddressBase* mem) {
        if (!mem->is_null()) {
          ++executor_deallocations;
        }
      });
  EXPECT_CALL(stream_, Memcpy(_, first.initial_data().data(), 16))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream_, MemZero(_, 32)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream_, BlockHostUntilDone()).WillOnce(Return(absl::OkStatus()));

  {
    GpuExecutableBufferAllocator executable_allocator(
        "test", allocations, ShapeUtil::MakeShape(F32, {64}), &debug_options_,
        /*thunk_executor=*/nullptr, /*num_devices=*/1);
    auto no_params = [&](const BufferAllocation&)
        -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
      return absl::InternalError("unexpected parameter");
    };
    auto generate = [&]() -> absl::StatusOr<BufferAllocations> {
      ABSL_ASSIGN_OR_RETURN(
          std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
          executable_allocator.CreateExecutionScope(&service_run_options_,
                                                    &allocator_,
                                                    /*device_ordinal=*/0));
      return scope->GenerateBufferAllocations(&service_run_options_, no_params,
                                              &globals_, &allocator_);
    };

    EXPECT_THAT(generate().status(),
                StatusIs(absl::StatusCode::kResourceExhausted,
                         HasSubstr("persistent buffer allocation 1")));
    // The buffer allocated before the failure is freed rather than cached.
    EXPECT_EQ(executor_deallocations, 1);

    ASSERT_OK_AND_ASSIGN(BufferAllocations buffers, generate());
    EXPECT_EQ(buffers.GetDeviceAddress(0).opaque(),
              static_cast<void*>(storage_0));
    EXPECT_EQ(buffers.GetDeviceAddress(1).opaque(),
              static_cast<void*>(storage_1));
    EXPECT_EQ(executor_deallocations, 1);
  }
  EXPECT_EQ(executor_deallocations, 3);
}

TEST_F(GpuExecutableBufferAllocatorTest,
       RejectsPersistentBufferInNonDefaultMemorySpace) {
  BufferAllocation persistent(/*index=*/0, /*size=*/16, kCollective);
  persistent.set_persistent(true);
  std::vector<const BufferAllocation*> allocations = {&persistent};

  EXPECT_CALL(executor_, Allocate(_, _)).Times(0);

  EXPECT_THAT(Generate(allocations).status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unsupported memory space color")));
}

TEST_F(GpuExecutableBufferAllocatorTest,
       VirtualDevicesOnSamePhysicalExecutorSharePersistentBuffers) {
  BufferAllocation persistent(/*index=*/0, /*size=*/16, kDefault);
  persistent.set_persistent(true);
  BufferAllocation temp(/*index=*/1, /*size=*/64, kDefault);
  std::vector<const BufferAllocation*> allocations = {&persistent, &temp};

  alignas(kAlignment) uint8_t persistent_storage[16]{};
  ON_CALL(executor_, device_ordinal()).WillByDefault(Return(0));
  EXPECT_CALL(executor_, Allocate(16, kDefault))
      .WillOnce(Return(se::DeviceAddressBase(persistent_storage, 16)));
  EXPECT_CALL(executor_, Deallocate(_)).Times(1);
  EXPECT_CALL(stream_, MemZero(_, 16)).WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(stream_, BlockHostUntilDone()).WillOnce(Return(absl::OkStatus()));

  GpuExecutableBufferAllocator executable_allocator(
      "test", allocations, ShapeUtil::MakeShape(F32, {64}), &debug_options_,
      /*thunk_executor=*/nullptr, /*num_devices=*/2);
  auto no_params = [&](const BufferAllocation&)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return absl::InternalError("unexpected parameter");
  };

  // Logical device ordinals 0 and 1 share physical `executor_` (ordinal 0), so
  // both scopes must resolve the same persistent buffer address (matching
  // CommandBufferThunk), while transient allocations keep their logical
  // ordinals.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope0,
      executable_allocator.CreateExecutionScope(
          &service_run_options_, &allocator_, /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(
      BufferAllocations buffers0,
      scope0->GenerateBufferAllocations(&service_run_options_, no_params,
                                        &globals_, &allocator_));
  EXPECT_EQ(buffers0.device_ordinal(), 0);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope1,
      executable_allocator.CreateExecutionScope(
          &service_run_options_, &allocator_, /*device_ordinal=*/1));
  ASSERT_OK_AND_ASSIGN(
      BufferAllocations buffers1,
      scope1->GenerateBufferAllocations(&service_run_options_, no_params,
                                        &globals_, &allocator_));
  EXPECT_EQ(buffers1.device_ordinal(), 1);
  EXPECT_EQ(buffers0.GetDeviceAddress(0).opaque(),
            buffers1.GetDeviceAddress(0).opaque());

  // Validate stream and StreamExecutor on the acquire fast path too.
  ExecutableRunOptions null_stream_opts;
  ServiceExecutableRunOptions null_stream_service_opts(null_stream_opts);
  EXPECT_THAT(scope0
                  ->GenerateBufferAllocations(&null_stream_service_opts,
                                              no_params, &globals_, &allocator_)
                  .status(),
              StatusIs(absl::StatusCode::kInternal));

  NiceMock<se::MockStreamExecutor> other_executor;
  ON_CALL(other_executor, device_ordinal()).WillByDefault(Return(0));
  NiceMock<se::MockStream> other_stream;
  ON_CALL(other_stream, parent()).WillByDefault(Return(&other_executor));
  ExecutableRunOptions other_opts;
  other_opts.set_stream(&other_stream);
  ServiceExecutableRunOptions other_service_opts(other_opts);
  EXPECT_THAT(scope0
                  ->GenerateBufferAllocations(&other_service_opts, no_params,
                                              &globals_, &allocator_)
                  .status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("StreamExecutor mismatch")));

  EXPECT_OK(scope0->TearDown(/*live_addresses=*/{}, buffers0));
  EXPECT_OK(scope1->TearDown(/*live_addresses=*/{}, buffers1));
}

TEST_F(GpuExecutableBufferAllocatorTest,
       RejectsMisalignedPersistentBufferFromExecutor) {
  BufferAllocation persistent(/*index=*/0, /*size=*/16, kDefault);
  persistent.set_persistent(true);
  persistent.set_initial_data(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  std::vector<const BufferAllocation*> allocations = {&persistent};

  alignas(kAlignment) uint8_t backing[512]{};
  uint8_t* unaligned = backing + 64;
  EXPECT_CALL(executor_, Allocate(16, kDefault))
      .WillOnce(Return(se::DeviceAddressBase(unaligned, 16)));
  std::vector<se::DeviceAddressBase> deallocated;
  EXPECT_CALL(executor_, Deallocate(_))
      .WillOnce(
          [&](se::DeviceAddressBase* mem) { deallocated.push_back(*mem); });
  EXPECT_CALL(stream_, Memcpy(_, persistent.initial_data().data(), _)).Times(0);
  EXPECT_CALL(stream_, BlockHostUntilDone()).Times(0);

  EXPECT_THAT(Generate(allocations).status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("not aligned to 256 bytes")));
  ASSERT_EQ(deallocated.size(), 1);
  EXPECT_EQ(deallocated[0].opaque(), static_cast<void*>(unaligned));
  EXPECT_EQ(deallocated[0].size(), 16);
}

}  // namespace
}  // namespace xla::gpu

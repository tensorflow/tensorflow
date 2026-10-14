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

#include "xla/service/gpu/gpu_executable_buffer_allocator.h"

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
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::NiceMock;
using ::testing::Pair;

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
        &service_run_options_, get_parameter_buffer, &globals_, &allocator_,
        /*device_ordinal=*/0);
  }

  NiceMock<se::MockPlatform> platform_;
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

}  // namespace
}  // namespace xla::gpu

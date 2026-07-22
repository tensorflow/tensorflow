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

#include "xla/service/gpu/gpu_executable_va_remap_allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NiceMock;
using ::testing::Optional;
using ::testing::Return;

// Matches kXlaAllocatedBufferAlignBytes so reservation slices and host-backed
// allocations satisfy CheckAlignment for every allocation kind.
constexpr uint64_t kGranularity = 256;

uint64_t RoundUpTestSize(uint64_t size) {
  return ((size + kGranularity - 1) / kGranularity) * kGranularity;
}

// Host-backed storage aligned to kGranularity.
class AlignedStorage {
 public:
  explicit AlignedStorage(uint64_t size)
      : storage_(std::make_unique<uint8_t[]>(size + kGranularity)) {}

  uint8_t* data() const {
    return reinterpret_cast<uint8_t*>(
        (reinterpret_cast<uintptr_t>(storage_.get()) + kGranularity - 1) &
        ~uintptr_t{kGranularity - 1});
  }

 private:
  std::unique_ptr<uint8_t[]> storage_;
};

class TestMemoryAllocation final : public se::MemoryAllocation {
 public:
  explicit TestMemoryAllocation(uint64_t size) : storage_(size), size_(size) {}

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(storage_.data(), size_);
  }

 private:
  AlignedStorage storage_;
  uint64_t size_;
};

class TestMemoryReservation final : public se::MemoryReservation {
 public:
  TestMemoryReservation(
      uint64_t size,
      std::shared_ptr<std::vector<TestMemoryReservation*>> registry)
      : storage_(size), size_(size), registry_(std::move(registry)) {
    registry_->push_back(this);
  }

  ~TestMemoryReservation() override {
    registry_->erase(std::find(registry_->begin(), registry_->end(), this));
  }

  se::DeviceAddressBase address() const override {
    return se::DeviceAddressBase(storage_.data(), size_);
  }

  int active_mapping_count() const { return active_mapping_count_; }
  const void* last_mapped_allocation_address() const {
    return last_mapped_allocation_address_;
  }

  bool Contains(const void* addr) const {
    const uintptr_t base = reinterpret_cast<uintptr_t>(storage_.data());
    const uintptr_t ptr = reinterpret_cast<uintptr_t>(addr);
    return ptr >= base && ptr < base + size_;
  }

 private:
  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, se::MemoryAllocation& allocation) override {
    if (reservation_offset > size_ || size > size_ - reservation_offset ||
        allocation_offset > allocation.address().size() ||
        size > allocation.address().size() - allocation_offset) {
      return absl::InvalidArgumentError("mapping range is out of bounds");
    }
    last_mapped_allocation_address_ = allocation.address().opaque();
    ++active_mapping_count_;
    return absl::OkStatus();
  }

  absl::Status SetAccess(uint64_t /*reservation_offset*/,
                         size_t /*size*/) override {
    return absl::OkStatus();
  }

  absl::Status UnMap(size_t /*reservation_offset*/, size_t /*size*/) override {
    if (active_mapping_count_ == 0) {
      return absl::FailedPreconditionError("reservation is not mapped");
    }
    --active_mapping_count_;
    return absl::OkStatus();
  }

  AlignedStorage storage_;
  uint64_t size_;
  std::shared_ptr<std::vector<TestMemoryReservation*>> registry_;
  int active_mapping_count_ = 0;
  const void* last_mapped_allocation_address_ = nullptr;
};

// Host-only VMM allocator: the 4 platform hooks are backed by heap memory, so
// the full reservation/mapping state machine runs without a GPU.
class TestVmmAllocator final : public se::DeviceAddressVmmAllocator {
 public:
  static absl::StatusOr<std::unique_ptr<TestVmmAllocator>> Create(
      const se::Platform* platform, absl::Span<const DeviceConfig> devices) {
    auto allocator =
        std::unique_ptr<TestVmmAllocator>(new TestVmmAllocator(platform));
    absl::Status status = PopulateDevices(allocator.get(), devices);
    if (!status.ok()) {
      return status;
    }
    return allocator;
  }

  TestMemoryReservation* last_reservation() const { return last_reservation_; }
  const void* last_created_allocation_address() const {
    return last_created_allocation_address_;
  }
  void FailNextDeferredDeallocation() {
    fail_next_deferred_deallocation_ = true;
  }

  // Returns the live reservation whose storage contains `addr`, or null.
  TestMemoryReservation* FindReservationContaining(const void* addr) const {
    for (TestMemoryReservation* reservation : *live_reservations_) {
      if (reservation->Contains(addr)) {
        return reservation;
      }
    }
    return nullptr;
  }

 protected:
  absl::Status InitializeDeviceState(PerDeviceState& state) override {
    state.allocation_granularity = kGranularity;
    auto* timeline = new uint64_t(0);
    state.pinned_timeline = timeline;
    state.destroy_fn = [timeline] { delete timeline; };
    return absl::OkStatus();
  }

  absl::StatusOr<std::unique_ptr<se::MemoryAllocation>> CreateAllocation(
      se::StreamExecutor* /*executor*/, uint64_t size) override {
    std::unique_ptr<se::MemoryAllocation> allocation =
        std::make_unique<TestMemoryAllocation>(RoundUpTestSize(size));
    last_created_allocation_address_ = allocation->address().opaque();
    return allocation;
  }

  absl::StatusOr<std::unique_ptr<se::MemoryReservation>> CreateReservation(
      se::StreamExecutor* /*executor*/, uint64_t size) override {
    auto reservation = std::make_unique<TestMemoryReservation>(
        RoundUpTestSize(size), live_reservations_);
    last_reservation_ = reservation.get();
    return reservation;
  }

  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override {
    if (fail_next_deferred_deallocation_) {
      fail_next_deferred_deallocation_ = false;
      return absl::InternalError("injected deferred deallocation failure");
    }
    __atomic_store_n(state.pinned_timeline, seqno, __ATOMIC_RELEASE);
    return absl::OkStatus();
  }

 private:
  explicit TestVmmAllocator(const se::Platform* platform)
      : DeviceAddressVmmAllocator(platform) {}

  TestMemoryReservation* last_reservation_ = nullptr;
  const void* last_created_allocation_address_ = nullptr;
  bool fail_next_deferred_deallocation_ = false;
  std::shared_ptr<std::vector<TestMemoryReservation*>> live_reservations_ =
      std::make_shared<std::vector<TestMemoryReservation*>>();
};

class GpuExecutableVaRemapAllocatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ON_CALL(executor_, device_ordinal()).WillByDefault(Return(0));
    ON_CALL(stream_, parent()).WillByDefault(Return(&executor_));
    run_options_.set_stream(&stream_);
    service_run_options_ = ServiceExecutableRunOptions(run_options_);
  }

  absl::StatusOr<std::unique_ptr<TestVmmAllocator>> CreateAllocator() {
    return TestVmmAllocator::Create(&platform_, {{&executor_, &stream_}});
  }

  // Observations from one GenerateBufferAllocations + Execute round trip for
  // allocation 0.
  struct ExecutionResult {
    bool va_remap_enabled = false;
    // Persistent allocation indices seen by `execute` (nullopt while the
    // SKIP_PROFILED profile is still running).
    std::optional<std::vector<BufferAllocation::Index>> persistent;
    se::DeviceAddressBase address_during_execute;
    se::DeviceAddressBase owning_address_after_execute;
  };

  absl::StatusOr<ExecutionResult> RunExecution(
      GpuExecutableVaRemapAllocator& allocator, TestVmmAllocator* vmm_allocator,
      GpuExecutableBufferAllocator::ParameterBufferResolver
          get_parameter_buffer,
      absl::Span<const BufferAllocation* const> allocations_to_tear_down = {}) {
    GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;
    ExecutionResult result;
    ASSIGN_OR_RETURN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        allocator.CreateExecutionScope(&service_run_options_, vmm_allocator,
                                       /*device_ordinal=*/0));
    result.va_remap_enabled = scope->va_remap_enabled();
    ASSIGN_OR_RETURN(BufferAllocations buffer_allocations,
                     scope->GenerateBufferAllocations(
                         &service_run_options_, get_parameter_buffer, &globals,
                         vmm_allocator, /*device_ordinal=*/0));
    RETURN_IF_ERROR(scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations& execution_buffers,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices) {
          result.address_during_execute = execution_buffers.GetDeviceAddress(0);
          if (persistent_alloc_indices.has_value()) {
            result.persistent.emplace(persistent_alloc_indices->begin(),
                                      persistent_alloc_indices->end());
          }
          return absl::OkStatus();
        }));
    result.owning_address_after_execute =
        buffer_allocations.GetDeviceAddress(0);
    if (!allocations_to_tear_down.empty()) {
      std::set<se::DeviceAddressBase> no_live_addresses;
      RETURN_IF_ERROR(buffer_allocations.TearDown(no_live_addresses,
                                                  allocations_to_tear_down));
    }
    return result;
  }

  NiceMock<se::MockPlatform> platform_;
  NiceMock<se::MockStreamExecutor> executor_;
  NiceMock<se::MockStream> stream_;
  ExecutableRunOptions run_options_;
  ServiceExecutableRunOptions service_run_options_;
};

TEST_F(GpuExecutableVaRemapAllocatorTest,
       RemapsParameterBufferToStableReservationAddress) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  param_alloc.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(DebugOptions::SKIP_TEMP);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddVaRemappedAllocationForTesting(0);
  EXPECT_EQ(allocator.command_buffer_allocation_count(), 1);

  // Caller-owned parameter buffer. It must be an exact allocator address so
  // it can be aliased into the reservation with Map().
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  const void* param_allocation_address =
      vmm_allocator->last_created_allocation_address();
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> replacement_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  const void* replacement_allocation_address =
      vmm_allocator->last_created_allocation_address();

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  const void* reservation_address = nullptr;
  for (int run = 0; run < 2; ++run) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        allocator.CreateExecutionScope(&service_run_options_,
                                       vmm_allocator.get(),
                                       /*device_ordinal=*/0));
    EXPECT_TRUE(scope->va_remap_enabled());

    ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                         scope->GenerateBufferAllocations(
                             &service_run_options_, get_parameter_buffer,
                             &globals, vmm_allocator.get(),
                             /*device_ordinal=*/0));

    // GenerateBufferAllocations and any output handling before execution keep
    // the owning table at external, deallocatable addresses. The second run
    // simulates copy protection replacing the parameter entry with an output
    // buffer after allocation generation.
    se::DeviceAddressBase external_address = param_buffer.cref();
    const void* external_allocation_address = param_allocation_address;
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              external_address.opaque());
    if (run == 1) {
      external_address = replacement_buffer.cref();
      external_allocation_address = replacement_allocation_address;
      buffer_allocations.GetMutableDeviceAddress(0) = external_address;
    }
    ASSERT_NE(vmm_allocator->last_reservation(), nullptr);

    bool executed = false;
    absl::Status execute_status = scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations& execution_buffers,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices) {
          executed = true;
          se::DeviceAddressBase mapped = execution_buffers.GetDeviceAddress(0);
          EXPECT_NE(mapped.opaque(), external_address.opaque());
          if (run == 0) {
            reservation_address = mapped.opaque();
          }
          EXPECT_EQ(mapped.opaque(), reservation_address);
          EXPECT_EQ(mapped.opaque(),
                    vmm_allocator->last_reservation()->address().opaque());
          EXPECT_EQ(vmm_allocator->last_reservation()
                        ->last_mapped_allocation_address(),
                    external_allocation_address);
          EXPECT_TRUE(persistent_alloc_indices.has_value());
          EXPECT_THAT(*persistent_alloc_indices, ElementsAre(0));
          return run == 0 ? absl::OkStatus()
                          : absl::InternalError("execution failed");
        });
    EXPECT_TRUE(executed);
    EXPECT_EQ(execute_status.ok(), run == 0);

    // The owning table is never rewritten, including when execution fails.
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              external_address.opaque());
  }

  // All reservation-address aliases are released once deferred operations
  // complete.
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  EXPECT_EQ(vmm_allocator->last_reservation()->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       MappingFailureReleasesPreviouslyInstalledAliases) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation valid_param_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  valid_param_alloc.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  BufferAllocation null_param_alloc(/*index=*/1, kBufferSize, /*color=*/0);
  null_param_alloc.set_entry_computation_parameter(
      /*parameter_number=*/1, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  std::vector<const BufferAllocation*> allocations = {&valid_param_alloc,
                                                      &null_param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(DebugOptions::SKIP_TEMP);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddVaRemappedAllocationForTesting(0);
  allocator.AddVaRemappedAllocationForTesting(1);

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> valid_param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  const void* valid_param_allocation_address =
      vmm_allocator->last_created_allocation_address();

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    if (allocation.parameter_number() == 0) {
      return GpuExecutableBufferAllocator::ParameterBuffer{
          valid_param_buffer.cref(), allocation.parameter_number()};
    }
    return GpuExecutableBufferAllocator::ParameterBuffer{
        se::DeviceAddressBase(), allocation.parameter_number(),
        /*allow_null_buffer=*/true};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
      allocator.CreateExecutionScope(&service_run_options_, vmm_allocator.get(),
                                     /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                       scope->GenerateBufferAllocations(
                           &service_run_options_, get_parameter_buffer,
                           &globals, vmm_allocator.get(),
                           /*device_ordinal=*/0));
  EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
            valid_param_buffer.cref().opaque());
  EXPECT_TRUE(buffer_allocations.GetDeviceAddress(1).is_null());

  bool executed = false;
  absl::Status status = scope->ExecuteWithBufferAllocations(
      buffer_allocations, /*device_ordinal=*/0,
      [&](const BufferAllocations&,
          std::optional<absl::Span<const BufferAllocation::Index>>) {
        executed = true;
        return absl::OkStatus();
      });
  EXPECT_FALSE(status.ok());
  EXPECT_FALSE(executed);

  // The first alias was installed before mapping the null parameter failed;
  // it must still be released before ExecuteWithBufferAllocations returns.
  ASSERT_NE(vmm_allocator->last_reservation(), nullptr);
  EXPECT_EQ(vmm_allocator->last_reservation()->last_mapped_allocation_address(),
            valid_param_allocation_address);
  EXPECT_EQ(vmm_allocator->last_reservation()->active_mapping_count(), 1);
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  EXPECT_EQ(vmm_allocator->last_reservation()->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       ExecutionScopeRetriesFailedAliasRelease) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  param_alloc.set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(DebugOptions::SKIP_TEMP);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddVaRemappedAllocationForTesting(0);

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
      allocator.CreateExecutionScope(&service_run_options_, vmm_allocator.get(),
                                     /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                       scope->GenerateBufferAllocations(
                           &service_run_options_, get_parameter_buffer,
                           &globals, vmm_allocator.get(),
                           /*device_ordinal=*/0));

  vmm_allocator->FailNextDeferredDeallocation();
  absl::Status status = scope->ExecuteWithBufferAllocations(
      buffer_allocations, /*device_ordinal=*/0,
      [](const BufferAllocations&,
         std::optional<absl::Span<const BufferAllocation::Index>>) {
        return absl::OkStatus();
      });
  EXPECT_FALSE(status.ok());
  ASSERT_NE(vmm_allocator->last_reservation(), nullptr);
  EXPECT_EQ(vmm_allocator->last_reservation()->active_mapping_count(), 1);
  EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
            param_buffer.cref().opaque());

  // ReleaseStepAliases retains the failed alias. Destroying the execution
  // scope retries UnMap after the one-shot injected failure.
  scope.reset();
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  EXPECT_EQ(vmm_allocator->last_reservation()->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       RemapsOutputBufferUsingExecutionOnlyAllocationTable) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  // Exercise a logical allocation size that is not VMM-granularity aligned.
  constexpr int64_t kBufferSize = 1000;
  BufferAllocation output_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  output_alloc.set_maybe_live_out(true);
  std::vector<const BufferAllocation*> allocations = {&output_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(DebugOptions::SKIP_TEMP);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {250}),
                                          &debug_options, &thunk_executor);
  allocator.AddVaRemappedAllocationForTesting(0);

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return absl::InternalError("no parameters in this test");
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  const void* reservation_address = nullptr;
  TestMemoryReservation* va_reservation = nullptr;
  for (int run = 0; run < 2; ++run) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        allocator.CreateExecutionScope(&service_run_options_,
                                       vmm_allocator.get(),
                                       /*device_ordinal=*/0));
    ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                         scope->GenerateBufferAllocations(
                             &service_run_options_, get_parameter_buffer,
                             &globals, vmm_allocator.get(),
                             /*device_ordinal=*/0));

    // GenerateBufferAllocations leaves an allocator-owned address in the
    // owning table, so callers can expose and eventually deallocate it.
    const se::DeviceAddressBase external =
        buffer_allocations.GetDeviceAddress(0);
    EXPECT_EQ(external.size(), kBufferSize);
    se::MemoryAllocation* raw_allocation =
        vmm_allocator->GetRawAllocation(/*device_ordinal=*/0, external);
    ASSERT_NE(raw_allocation, nullptr);

    bool executed = false;
    ASSERT_OK(scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations& execution_buffers,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices) {
          executed = true;
          const se::DeviceAddressBase mapped =
              execution_buffers.GetDeviceAddress(0);
          EXPECT_NE(mapped.opaque(), external.opaque());
          if (run == 0) {
            reservation_address = mapped.opaque();
            va_reservation =
                vmm_allocator->FindReservationContaining(mapped.opaque());
            if (va_reservation == nullptr) {
              return absl::InternalError(
                  "execution address is outside the VA reservation");
            }
            EXPECT_EQ(mapped.opaque(), va_reservation->address().opaque());
          }
          EXPECT_EQ(mapped.opaque(), reservation_address);
          EXPECT_EQ(va_reservation->last_mapped_allocation_address(),
                    raw_allocation->address().opaque());
          EXPECT_TRUE(persistent_alloc_indices.has_value());
          EXPECT_THAT(*persistent_alloc_indices, ElementsAre(0));
          return absl::OkStatus();
        }));
    EXPECT_TRUE(executed);

    // ExecuteWithBufferAllocations remaps only its local copy.
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              external.opaque());
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).size(), kBufferSize);

    // The external buffer escapes to the caller; release it like a result
    // consumer would.
    ASSERT_OK(vmm_allocator->Deallocate(/*device_ordinal=*/0, external));
  }

  // All reservation-address aliases are released once deferred operations
  // complete.
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  ASSERT_NE(va_reservation, nullptr);
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

BufferAllocation MakeParameterAllocation(BufferAllocation::Index index,
                                         int64_t size,
                                         int64_t parameter_number) {
  BufferAllocation allocation(index, size, /*color=*/0);
  allocation.set_entry_computation_parameter(
      parameter_number, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  return allocation;
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledRemapsStableParameterAfterProfiling) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  // The parameter keeps the same VMM-backed address on every execution, so
  // the profile selects it.
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };

  const void* reservation_address = nullptr;
  for (int run = 0; run < 5; ++run) {
    ASSERT_OK_AND_ASSIGN(
        ExecutionResult result,
        RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
    EXPECT_TRUE(result.va_remap_enabled);
    if (run < 3) {
      // Profiling executions pass std::nullopt and do not remap.
      EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
      EXPECT_EQ(result.address_during_execute.opaque(),
                param_buffer.cref().opaque());
    } else {
      // After the transition the parameter is aliased at a stable
      // reservation address and passed as persistent.
      EXPECT_THAT(result.persistent, Optional(ElementsAre(0))) << "run " << run;
      EXPECT_NE(result.address_during_execute.opaque(),
                param_buffer.cref().opaque());
      if (reservation_address == nullptr) {
        reservation_address = result.address_during_execute.opaque();
      }
      EXPECT_EQ(result.address_during_execute.opaque(), reservation_address);
      EXPECT_EQ(result.owning_address_after_execute.opaque(),
                param_buffer.cref().opaque());
    }
  }

  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  TestMemoryReservation* va_reservation =
      vmm_allocator->FindReservationContaining(reservation_address);
  ASSERT_NE(va_reservation, nullptr);
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledDoesNotCountFailedExecution) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
      allocator.CreateExecutionScope(&service_run_options_, vmm_allocator.get(),
                                     /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                       scope->GenerateBufferAllocations(
                           &service_run_options_, get_parameter_buffer,
                           &globals, vmm_allocator.get(),
                           /*device_ordinal=*/0));
  EXPECT_FALSE(
      scope
          ->ExecuteWithBufferAllocations(
              buffer_allocations, /*device_ordinal=*/0,
              [](const BufferAllocations&,
                 std::optional<absl::Span<const BufferAllocation::Index>>) {
                return absl::InternalError("execution failed");
              })
          .ok());
  // Execution scopes serialize remapping for an executable/executor pair.
  // Release the failed scope before starting another execution.
  scope.reset();

  // Three successful observations are still required after the failure.
  for (int run = 0; run < 3; ++run) {
    ASSERT_OK_AND_ASSIGN(
        ExecutionResult result,
        RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
    EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
  }
  ASSERT_OK_AND_ASSIGN(
      ExecutionResult active_result,
      RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
  EXPECT_THAT(active_result.persistent, Optional(ElementsAre(0)));
  EXPECT_NE(active_result.address_during_execute.opaque(),
            param_buffer.cref().opaque());
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledAutomaticallyRemapsTempWithoutProfileCandidates) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation temp_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  std::vector<const BufferAllocation*> allocations = {&temp_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  // Simulate automatic constructor classification with an empty test thunk
  // sequence.
  allocator.AddVaRemappedAllocationForTesting(0);

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return absl::InternalError("no parameters in this test");
  };
  const void* remapped_temp_address = nullptr;
  TestMemoryReservation* va_reservation = nullptr;

  for (int run = 0; run < 5; ++run) {
    ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                         RunExecution(allocator, vmm_allocator.get(),
                                      get_parameter_buffer, allocations));
    EXPECT_TRUE(result.va_remap_enabled) << "run " << run;
    EXPECT_EQ(result.address_during_execute.opaque(),
              result.owning_address_after_execute.opaque());
    if (run < 3) {
      EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
      continue;
    }

    EXPECT_THAT(result.persistent, Optional(ElementsAre(0))) << "run " << run;
    if (remapped_temp_address == nullptr) {
      remapped_temp_address = result.address_during_execute.opaque();
      va_reservation =
          vmm_allocator->FindReservationContaining(remapped_temp_address);
      ASSERT_NE(va_reservation, nullptr);
    }
    EXPECT_EQ(result.address_during_execute.opaque(), remapped_temp_address);
  }

  ASSERT_NE(va_reservation, nullptr);
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(
      /*device_ordinal=*/0));
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledAutomaticallyRemapsTempWhenParameterIsUnstable) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation temp_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/1, kBufferSize,
                              /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&temp_alloc,
                                                      &param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  // Simulate constructor classification with an empty test thunk sequence:
  // the temp is automatic, while the parameter is profiled.
  allocator.AddVaRemappedAllocationForTesting(0);
  allocator.AddProfileCandidateAllocationForTesting(1);

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_a,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_b,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  int run = 0;
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        (run % 2 == 0) ? param_a.cref() : param_b.cref(),
        allocation.parameter_number()};
  };

  const void* remapped_temp_address = nullptr;
  TestMemoryReservation* va_reservation = nullptr;
  for (run = 0; run < 5; ++run) {
    ASSERT_OK_AND_ASSIGN(ExecutionResult result,
                         RunExecution(allocator, vmm_allocator.get(),
                                      get_parameter_buffer, allocations));
    EXPECT_TRUE(result.va_remap_enabled) << "run " << run;
    EXPECT_EQ(result.address_during_execute.opaque(),
              result.owning_address_after_execute.opaque());
    if (run < 3) {
      EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
      continue;
    }

    // The unstable parameter is rejected, but the automatic temp keeps the
    // remapping active and is the only persistent allocation.
    EXPECT_THAT(result.persistent, Optional(ElementsAre(0))) << "run " << run;
    if (remapped_temp_address == nullptr) {
      remapped_temp_address = result.address_during_execute.opaque();
      va_reservation =
          vmm_allocator->FindReservationContaining(remapped_temp_address);
      ASSERT_NE(va_reservation, nullptr);
    }
    EXPECT_EQ(result.address_during_execute.opaque(), remapped_temp_address);
  }

  ASSERT_NE(va_reservation, nullptr);
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(
      /*device_ordinal=*/0));
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledRemapsStableOutputAfterProfiling) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  // Exercise an output whose logical size is not VMM-granularity aligned.
  constexpr int64_t kBufferSize = 1000;
  BufferAllocation output_alloc(/*index=*/0, kBufferSize, /*color=*/0);
  output_alloc.set_maybe_live_out(true);
  std::vector<const BufferAllocation*> allocations = {&output_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {250}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return absl::InternalError("no parameters in this test");
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  const void* profiled_external_address = nullptr;
  const void* reservation_address = nullptr;
  TestMemoryReservation* va_reservation = nullptr;
  for (int run = 0; run < 5; ++run) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        allocator.CreateExecutionScope(&service_run_options_,
                                       vmm_allocator.get(),
                                       /*device_ordinal=*/0));
    EXPECT_TRUE(scope->va_remap_enabled());
    ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                         scope->GenerateBufferAllocations(
                             &service_run_options_, get_parameter_buffer,
                             &globals, vmm_allocator.get(),
                             /*device_ordinal=*/0));

    const se::DeviceAddressBase external =
        buffer_allocations.GetDeviceAddress(0);
    EXPECT_EQ(external.size(), kBufferSize);
    se::MemoryAllocation* raw_allocation =
        vmm_allocator->GetRawAllocation(/*device_ordinal=*/0, external);
    ASSERT_NE(raw_allocation, nullptr);
    if (run < 3) {
      if (profiled_external_address == nullptr) {
        profiled_external_address = external.opaque();
      }
      EXPECT_EQ(external.opaque(), profiled_external_address) << "run " << run;
    }

    ASSERT_OK(scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations& execution_buffers,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices) {
          const se::DeviceAddressBase execution_address =
              execution_buffers.GetDeviceAddress(0);
          if (run < 3) {
            EXPECT_EQ(execution_address.opaque(), external.opaque());
            EXPECT_FALSE(persistent_alloc_indices.has_value());
          } else {
            EXPECT_NE(execution_address.opaque(), external.opaque());
            if (reservation_address == nullptr) {
              reservation_address = execution_address.opaque();
              va_reservation = vmm_allocator->FindReservationContaining(
                  execution_address.opaque());
            }
            EXPECT_EQ(execution_address.opaque(), reservation_address);
            if (va_reservation == nullptr) {
              return absl::InternalError(
                  "execution address is outside the VA reservation");
            }
            EXPECT_EQ(va_reservation->last_mapped_allocation_address(),
                      raw_allocation->address().opaque());
            EXPECT_THAT(persistent_alloc_indices, Optional(ElementsAre(0)));
          }
          return absl::OkStatus();
        }));

    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              external.opaque());
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).size(), kBufferSize);
    ASSERT_OK(vmm_allocator->Deallocate(/*device_ordinal=*/0, external));
  }

  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  ASSERT_NE(va_reservation, nullptr);
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledDisablesRemappingWhenParameterAddressUnstable) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  // The parameter address changes between profiling executions.
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> buffer_a,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> buffer_b,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  int run = 0;
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        (run % 2 == 0) ? buffer_a.cref() : buffer_b.cref(),
        allocation.parameter_number()};
  };

  for (run = 0; run < 3; ++run) {
    ASSERT_OK_AND_ASSIGN(
        ExecutionResult result,
        RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
    EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
  }

  // The profile selected nothing: executions fall back to the base scope and
  // pass only the (empty) constant set as persistent.
  ASSERT_OK_AND_ASSIGN(
      ExecutionResult result,
      RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
  EXPECT_FALSE(result.va_remap_enabled);
  EXPECT_THAT(result.persistent, Optional(IsEmpty()));
  EXPECT_EQ(result.address_during_execute.opaque(), buffer_b.cref().opaque());
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledSkipsParameterNotBackedByVmmAllocator) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  // Stable address, but not an address owned by the VMM allocator, so it
  // cannot be aliased with Map().
  AlignedStorage host_storage(kBufferSize);
  se::DeviceAddressBase host_buffer(host_storage.data(), kBufferSize);
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        host_buffer, allocation.parameter_number()};
  };

  for (int run = 0; run < 3; ++run) {
    ASSERT_OK_AND_ASSIGN(
        ExecutionResult result,
        RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
    EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
  }

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult result,
      RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
  EXPECT_FALSE(result.va_remap_enabled);
  EXPECT_THAT(result.persistent, Optional(IsEmpty()));
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledDropsParametersSharingABuffer) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc0 =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  BufferAllocation param_alloc1 =
      MakeParameterAllocation(/*index=*/1, kBufferSize, /*parameter_number=*/1);
  std::vector<const BufferAllocation*> allocations = {&param_alloc0,
                                                      &param_alloc1};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);
  allocator.AddProfileCandidateAllocationForTesting(1);

  // Both parameters resolve to the same buffer: Map() supports only one
  // reservation alias per allocator address, so both must be dropped.
  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> shared_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        shared_buffer.cref(), allocation.parameter_number()};
  };

  for (int run = 0; run < 3; ++run) {
    ASSERT_OK_AND_ASSIGN(
        ExecutionResult result,
        RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
    EXPECT_FALSE(result.persistent.has_value()) << "run " << run;
  }

  ASSERT_OK_AND_ASSIGN(
      ExecutionResult result,
      RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer));
  EXPECT_FALSE(result.va_remap_enabled);
  EXPECT_THAT(result.persistent, Optional(IsEmpty()));
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledProfilesFinalCopyProtectedAddress) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  const void* profiled_address = nullptr;
  for (int run = 0; run < 3; ++run) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
        allocator.CreateExecutionScope(&service_run_options_,
                                       vmm_allocator.get(),
                                       /*device_ordinal=*/0));
    ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                         scope->GenerateBufferAllocations(
                             &service_run_options_, get_parameter_buffer,
                             &globals, vmm_allocator.get(),
                             /*device_ordinal=*/0));
    ASSERT_OK_AND_ASSIGN(
        se::DeviceAddressBase copy_protected,
        scope->AllocateCopyProtectedOutputBuffer(
            &service_run_options_, buffer_allocations, /*index=*/{},
            param_alloc, /*device_ordinal=*/0, vmm_allocator.get(),
            [](absl::Status status) { return status; }));
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              copy_protected.opaque());
    if (profiled_address == nullptr) {
      profiled_address = copy_protected.opaque();
    }
    EXPECT_EQ(copy_protected.opaque(), profiled_address) << "run " << run;

    ASSERT_OK(scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations& execution_buffers,
            std::optional<absl::Span<const BufferAllocation::Index>>
                persistent_alloc_indices) {
          EXPECT_EQ(execution_buffers.GetDeviceAddress(0).opaque(),
                    copy_protected.opaque());
          EXPECT_FALSE(persistent_alloc_indices.has_value());
          return absl::OkStatus();
        }));
    EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
              copy_protected.opaque());
    ASSERT_OK(vmm_allocator->Deallocate(/*device_ordinal=*/0, copy_protected));
  }

  // The stable finalized copy-protected address was profiled, so the next
  // execution transitions to active remapping.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
      allocator.CreateExecutionScope(&service_run_options_, vmm_allocator.get(),
                                     /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                       scope->GenerateBufferAllocations(
                           &service_run_options_, get_parameter_buffer,
                           &globals, vmm_allocator.get(),
                           /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(
      se::DeviceAddressBase copy_protected,
      scope->AllocateCopyProtectedOutputBuffer(
          &service_run_options_, buffer_allocations, /*index=*/{}, param_alloc,
          /*device_ordinal=*/0, vmm_allocator.get(),
          [](absl::Status status) { return status; }));
  se::MemoryAllocation* raw_allocation =
      vmm_allocator->GetRawAllocation(/*device_ordinal=*/0, copy_protected);
  ASSERT_NE(raw_allocation, nullptr);

  TestMemoryReservation* va_reservation = nullptr;
  ASSERT_OK(scope->ExecuteWithBufferAllocations(
      buffer_allocations, /*device_ordinal=*/0,
      [&](const BufferAllocations& execution_buffers,
          std::optional<absl::Span<const BufferAllocation::Index>>
              persistent_alloc_indices) {
        const se::DeviceAddressBase mapped =
            execution_buffers.GetDeviceAddress(0);
        EXPECT_NE(mapped.opaque(), copy_protected.opaque());
        va_reservation =
            vmm_allocator->FindReservationContaining(mapped.opaque());
        if (va_reservation == nullptr) {
          return absl::InternalError(
              "execution address is outside the VA reservation");
        }
        EXPECT_EQ(va_reservation->last_mapped_allocation_address(),
                  raw_allocation->address().opaque());
        EXPECT_THAT(persistent_alloc_indices, Optional(ElementsAre(0)));
        return absl::OkStatus();
      }));
  EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
            copy_protected.opaque());
  ASSERT_OK(vmm_allocator->Deallocate(/*device_ordinal=*/0, copy_protected));
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  ASSERT_NE(va_reservation, nullptr);
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

TEST_F(GpuExecutableVaRemapAllocatorTest,
       SkipProfiledRemapsCopyProtectedReplacementAfterTransition) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<TestVmmAllocator> vmm_allocator,
                       CreateAllocator());

  constexpr int64_t kBufferSize = 1024;
  BufferAllocation param_alloc =
      MakeParameterAllocation(/*index=*/0, kBufferSize, /*parameter_number=*/0);
  std::vector<const BufferAllocation*> allocations = {&param_alloc};

  ThunkExecutor thunk_executor{ThunkSequence{}};
  DebugOptions debug_options;
  debug_options.set_xla_gpu_command_buffer_update_mode(
      DebugOptions::SKIP_PROFILED);
  GpuExecutableVaRemapAllocator allocator("test", allocations,
                                          ShapeUtil::MakeShape(F32, {256}),
                                          &debug_options, &thunk_executor);
  allocator.AddProfileCandidateAllocationForTesting(0);

  ASSERT_OK_AND_ASSIGN(
      se::ScopedDeviceAddress<uint8_t> param_buffer,
      vmm_allocator->Allocate(/*device_ordinal=*/0, kBufferSize,
                              /*retry_on_failure=*/true, /*memory_space=*/0));
  auto get_parameter_buffer = [&](const BufferAllocation& allocation)
      -> absl::StatusOr<GpuExecutableBufferAllocator::ParameterBuffer> {
    return GpuExecutableBufferAllocator::ParameterBuffer{
        param_buffer.cref(), allocation.parameter_number()};
  };
  GpuExecutableBufferAllocator::BufferAllocToDeviceMemoryMap globals;

  // Profile the stable parameter address.
  for (int run = 0; run < 3; ++run) {
    ASSERT_OK(RunExecution(allocator, vmm_allocator.get(), get_parameter_buffer)
                  .status());
  }

  // The next scope transitions to active remapping. Copy protection replaces
  // the owning address before execution, so the execution-only copy must map
  // that finalized address rather than the profiled parameter address.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope> scope,
      allocator.CreateExecutionScope(&service_run_options_, vmm_allocator.get(),
                                     /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(BufferAllocations buffer_allocations,
                       scope->GenerateBufferAllocations(
                           &service_run_options_, get_parameter_buffer,
                           &globals, vmm_allocator.get(),
                           /*device_ordinal=*/0));
  ASSERT_OK_AND_ASSIGN(
      se::DeviceAddressBase copy_protected,
      scope->AllocateCopyProtectedOutputBuffer(
          &service_run_options_, buffer_allocations, /*index=*/{}, param_alloc,
          /*device_ordinal=*/0, vmm_allocator.get(),
          [](absl::Status status) { return status; }));
  EXPECT_NE(copy_protected.opaque(), param_buffer.cref().opaque());
  EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
            copy_protected.opaque());
  se::MemoryAllocation* raw_allocation =
      vmm_allocator->GetRawAllocation(/*device_ordinal=*/0, copy_protected);
  ASSERT_NE(raw_allocation, nullptr);

  TestMemoryReservation* va_reservation = nullptr;
  ASSERT_OK(scope->ExecuteWithBufferAllocations(
      buffer_allocations, /*device_ordinal=*/0,
      [&](const BufferAllocations& execution_buffers,
          std::optional<absl::Span<const BufferAllocation::Index>>
              persistent_alloc_indices) {
        const se::DeviceAddressBase mapped =
            execution_buffers.GetDeviceAddress(0);
        EXPECT_NE(mapped.opaque(), copy_protected.opaque());
        va_reservation =
            vmm_allocator->FindReservationContaining(mapped.opaque());
        if (va_reservation == nullptr) {
          return absl::InternalError(
              "execution address is outside the VA reservation");
        }
        EXPECT_EQ(va_reservation->last_mapped_allocation_address(),
                  raw_allocation->address().opaque());
        EXPECT_THAT(persistent_alloc_indices, Optional(ElementsAre(0)));
        return absl::OkStatus();
      }));

  // Only the execution copy is remapped; the owning table retains the
  // copy-protected external address for result handling and deallocation.
  EXPECT_EQ(buffer_allocations.GetDeviceAddress(0).opaque(),
            copy_protected.opaque());
  ASSERT_OK(vmm_allocator->Deallocate(/*device_ordinal=*/0, copy_protected));
  ASSERT_OK(vmm_allocator->SynchronizePendingOperations(/*device_ordinal=*/0));
  ASSERT_NE(va_reservation, nullptr);
  EXPECT_EQ(va_reservation->active_mapping_count(), 0);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

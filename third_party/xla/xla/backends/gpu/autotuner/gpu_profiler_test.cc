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

#include "xla/backends/gpu/autotuner/gpu_profiler.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/redzone_buffers.h"
#include "xla/backends/gpu/gpu_compiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;

constexpr absl::string_view kGemmBackendConfig =
    R"(backend_config={"gemm_backend_config":{"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["1"]}}})";

class MockExecutable : public Executable {
 public:
  explicit MockExecutable(std::shared_ptr<HloModule> module, int duration_ns,
                          bool should_fail = false,
                          bool write_past_allocated_buffer = false)
      : MockExecutable(std::move(module), std::vector<int>{duration_ns},
                       should_fail, write_past_allocated_buffer) {}

  // The i-th profiled run reports the duration
  // `durations_ns[i % durations_ns.size()]`.
  MockExecutable(std::shared_ptr<HloModule> module,
                 std::vector<int> durations_ns, bool should_fail = false,
                 bool write_past_allocated_buffer = false)
      : Executable(std::move(module)),
        durations_ns_(std::move(durations_ns)),
        should_fail_(should_fail),
        write_past_allocated_buffer_(write_past_allocated_buffer) {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    if (should_fail_) {
      return absl::InternalError("MockExecutable failed as requested.");
    }
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    if (profile != nullptr) {
      profile->set_compute_time_ns(
          durations_ns_[num_profiled_runs_ % durations_ns_.size()]);
      ++num_profiled_runs_;
    }
    if (write_past_allocated_buffer_) {
      ABSL_RETURN_IF_ERROR(WriteOutOfBounds(*run_options));
    }
    const Shape& result_shape =
        module().entry_computation()->root_instruction()->shape();
    return ExecutionOutput(result_shape, result_shape,
                           run_options->run_options().allocator(),
                           run_options->run_options().device_ordinal());
  }

  int num_profiled_runs() const { return num_profiled_runs_; }

 private:
  // Simulates a kernel that writes past the end of an allocated buffer:
  // allocates a buffer through the run's allocator and then writes a few
  // bytes past the end of it. When the allocator handed to us is a
  // redzone-wrapping allocator (as GpuProfiler::Profile uses during its
  // warm-up run when ProfileOptions.redzone_padding_bytes > 0), this lands
  // in the mapped post-redzone instead of faulting, so it can be detected by
  // CheckRedzones() instead of crashing the process.
  absl::Status WriteOutOfBounds(
      const ServiceExecutableRunOptions& run_options) {
    constexpr int64_t kBufferBytes = 1024;
    constexpr int64_t kOverrunBytes = 64;
    se::DeviceAddressAllocator* allocator =
        run_options.run_options().allocator();
    ABSL_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> buffer,
        allocator->Allocate(run_options.run_options().device_ordinal(),
                            kBufferBytes));
    se::DeviceAddressBase oob_region(
        static_cast<char*>(buffer->opaque()) + kBufferBytes, kOverrunBytes);
    return run_options.run_options().stream()->MemZero(&oob_region,
                                                       kOverrunBytes);
  }

  std::vector<int> durations_ns_;
  int num_profiled_runs_ = 0;
  bool should_fail_;
  bool write_past_allocated_buffer_;
};

// Forwards to another allocator, records allocations, and optionally enforces a
// live-memory cap or asynchronous-deallocation semantics.
class RecordingAllocator : public se::DeviceAddressAllocator {
 public:
  struct AllocationRecord {
    uint64_t size;
    bool retry_on_failure;
  };

  explicit RecordingAllocator(
      se::DeviceAddressAllocator* wrapped,
      bool allows_async_deallocation = false,
      uint64_t max_live_bytes = std::numeric_limits<uint64_t>::max())
      : se::DeviceAddressAllocator(wrapped->platform()),
        wrapped_(wrapped),
        allows_async_deallocation_(allows_async_deallocation),
        max_live_bytes_(max_live_bytes) {}

  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    allocations_.push_back({size, retry_on_failure});
    if (size > max_live_bytes_ - live_bytes_) {
      return absl::ResourceExhaustedError("Simulated allocator OOM.");
    }
    ABSL_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> address,
                     wrapped_->Allocate(device_ordinal, size, retry_on_failure,
                                        memory_space));
    live_bytes_ += address->size();
    peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
    return se::ScopedDeviceAddress<uint8_t>(address.Release(), device_ordinal,
                                            this);
  }

  bool AllowsAsynchronousDeallocation() const override {
    return allows_async_deallocation_;
  }

  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return wrapped_->GetStream(device_ordinal);
  }

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase mem) override {
    if (!mem.is_null()) {
      if (allows_async_deallocation_ &&
          !wrapped_->AllowsAsynchronousDeallocation()) {
        ABSL_ASSIGN_OR_RETURN(se::Stream * stream, GetStream(device_ordinal));
        ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
      }
      live_bytes_ -= mem.size();
    }
    return wrapped_->Deallocate(device_ordinal, mem);
  }

  const std::vector<AllocationRecord>& allocations() const {
    return allocations_;
  }
  uint64_t live_bytes() const { return live_bytes_; }
  uint64_t peak_live_bytes() const { return peak_live_bytes_; }

 private:
  se::DeviceAddressAllocator* wrapped_;
  bool allows_async_deallocation_;
  uint64_t max_live_bytes_;
  uint64_t live_bytes_ = 0;
  uint64_t peak_live_bytes_ = 0;
  std::vector<AllocationRecord> allocations_;
};

int CountCacheFlushAllocations(const RecordingAllocator& allocator,
                               const se::StreamExecutor* stream_exec) {
  const uint64_t min_flush_bytes = std::max<uint64_t>(
      2 * stream_exec->GetDeviceDescription().l2_cache_size(), 64ULL << 20);
  int count = 0;
  for (const RecordingAllocator::AllocationRecord& alloc :
       allocator.allocations()) {
    if (alloc.size >= min_flush_bytes) {
      ++count;
      EXPECT_FALSE(alloc.retry_on_failure);
    }
  }
  return count;
}

absl::StatusOr<ScopedShapedBuffer> CreateTestBuffer(
    se::DeviceAddressAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  ABSL_ASSIGN_OR_RETURN(auto* transfer_manager,
                   TransferManager::GetForPlatform(stream_exec->GetPlatform()));
  ABSL_ASSIGN_OR_RETURN(ScopedShapedBuffer output,
                   transfer_manager->AllocateScopedShapedBuffer(
                       test_shape, allocator, stream_exec->device_ordinal()));
  Literal literal = LiteralUtil::CreateR0<int32_t>(value);
  ABSL_RETURN_IF_ERROR(
      transfer_manager->TransferLiteralToDevice(stream, literal, output));
  return output;
}

absl::StatusOr<ScopedShapedBuffer> CreateTupleTestBuffer(
    se::DeviceAddressAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value1, int32_t value2) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  Shape test_shape_tuple = ShapeUtil::MakeTupleShape({test_shape, test_shape});
  ABSL_ASSIGN_OR_RETURN(auto* transfer_manager,
                   TransferManager::GetForPlatform(stream_exec->GetPlatform()));
  ABSL_ASSIGN_OR_RETURN(
      ScopedShapedBuffer output,
      transfer_manager->AllocateScopedShapedBuffer(
          test_shape_tuple, allocator, stream_exec->device_ordinal()));
  Literal literal1 = LiteralUtil::CreateR0<int32_t>(value1);
  Literal literal2 = LiteralUtil::CreateR0<int32_t>(value2);
  Literal tuple_literal = LiteralUtil::MakeTuple({&literal1, &literal2});
  ABSL_RETURN_IF_ERROR(
      transfer_manager->TransferLiteralToDevice(stream, tuple_literal, output));
  return output;
}

class GpuProfilerTest : public HloHardwareIndependentTestBase {
 public:
  GpuProfilerTest() {
    se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
    std::vector<se::StreamExecutor*> executors =
        PlatformUtil::GetStreamExecutors(platform).value();
    stream_exec_ = executors[0];
    allocator_ =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            stream_exec_);
  }

  absl::StatusOr<int64_t> GetScratchBytes(absl::string_view hlo_text) {
    ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                     PlatformUtil::GetDefaultPlatform());
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                     Compiler::GetForPlatform(platform->id()));
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo_text));
    module->mutable_config()
        .mutable_debug_options()
        .clear_xla_gpu_enable_command_buffer();
    ABSL_ASSIGN_OR_RETURN(auto gpu_executable,
                     compiler->RunBackend(std::move(module), stream_exec_,
                                          GpuCompiler::CompileOptions()));
    auto profiler =
        GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> buffers,
                     profiler->CreateInputBuffers(gpu_executable.get()));
    ABSL_ASSIGN_OR_RETURN(ProfileResult profile,
                     profiler->Profile(gpu_executable.get(), *buffers));
    return profile.scratch_bytes;
  }

  // Returns a kRaggedNonContracting XTile fusion, as emitted by
  // GemmRewriter::HandleRaggedDot, whose group_sizes operand has element type
  // `gs_type`: 130 ragged rows split over 4 groups.
  static std::string RaggedDotHlo(absl::string_view gs_type) {
    return absl::Substitute(R"(
      HloModule module

      fusion1 {
        p0 = bf16[130,32] parameter(0)
        p1 = bf16[4,32,16] parameter(1)
        p2 = $0[4] parameter(2)
        ROOT rd = bf16[130,16] ragged-dot(p0, p1, p2),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
      }

      ENTRY main {
        p0 = bf16[130,32] parameter(0)
        p1 = bf16[4,32,16] parameter(1)
        p2 = $0[4] parameter(2)
        ROOT fusion = bf16[130,16] fusion(p0, p1, p2), kind=kCustom,
          calls=fusion1,
          backend_config={"fusion_backend_config": {kind: "__triton"}}
      })",
                            gs_type);
  }

  // Creates input buffers for the ragged-dot fusion in `hlo` and returns its
  // group_sizes buffer read back from the device. Fails if initializing that
  // buffer wrote outside of it.
  absl::StatusOr<Literal> ReadInitializedGroupSizes(absl::string_view hlo) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<HloModule> module,
                     ParseAndReturnVerifiedModule(hlo));
    const HloInstruction* fusion =
        module->entry_computation()->root_instruction();
    MockExecutable mock_executable(module, /*duration_ns=*/1000);

    // Redzones turn a write past the end of the group_sizes buffer into a
    // detectable mismatch rather than silent corruption of whatever device
    // memory follows it.
    ProfileOptions options;
    options.redzone_padding_bytes = 1024;
    std::unique_ptr<GpuProfiler> profiler =
        GpuProfiler::Create(stream_exec_, options, allocator_.get());
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<InputBuffers> buffers,
                     profiler->CreateInputBuffers(&mock_executable, fusion));
    const RedzoneBuffers& rz_buffers =
        absl::down_cast<const GpuInputBuffers&>(*buffers).redzone_buffers;

    ABSL_ASSIGN_OR_RETURN(se::RedzoneAllocator::RedzoneCheckStatus rz_check,
                     rz_buffers.RedzoneAllocator().CheckRedzones());
    if (!rz_check.ok()) {
      return absl::InternalError(
          absl::StrCat("group_sizes initialization wrote out of bounds: ",
                       rz_check.RedzoneFailureMsg()));
    }

    ABSL_ASSIGN_OR_RETURN(Literal group_sizes,
                     Literal::Make(fusion->operand(2)->shape()));
    ABSL_ASSIGN_OR_RETURN(se::Stream * stream,
                     allocator_->GetStream(stream_exec_->device_ordinal()));
    ABSL_RETURN_IF_ERROR(stream->Memcpy(group_sizes.untyped_data(),
                                   rz_buffers.input_buffers()[2],
                                   group_sizes.size_bytes()));
    ABSL_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    return group_sizes;
  }

  se::StreamExecutor* stream_exec_;
  std::unique_ptr<se::DeviceAddressAllocator> allocator_;
};

TEST_F(GpuProfilerTest, CreateInputBuffersAndProfile) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
  EXPECT_EQ(profile.output_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {}));
  EXPECT_EQ(profile.scratch_bytes, 0);
  EXPECT_EQ(mock_executable.num_profiled_runs(), 1);
}

TEST_F(GpuProfilerTest, InitializesRaggedDotGroupSizesForEveryIndexType) {
  // The emitter sign-extends group_sizes from any of S8, S16, S32 and S64, so
  // the buffer must be filled with elements of exactly that width: filling a
  // narrower buffer with wider elements runs past its end.
  for (absl::string_view gs_type : {"s8", "s16", "s32", "s64"}) {
    SCOPED_TRACE(gs_type);
    ASSERT_OK_AND_ASSIGN(Literal group_sizes,
                         ReadInitializedGroupSizes(RaggedDotHlo(gs_type)));
    ASSERT_OK_AND_ASSIGN(Literal widened, group_sizes.Convert(S64));
    // 130 rows over 4 groups: the first 130 % 4 == 2 groups get an extra row,
    // so the group sizes sum to exactly 130.
    EXPECT_EQ(widened, LiteralUtil::CreateR1<int64_t>({33, 33, 32, 32}));
  }
}

TEST_F(GpuProfilerTest, RejectsRaggedDotGroupSizesOfUnexpectedType) {
  EXPECT_THAT(ReadInitializedGroupSizes(RaggedDotHlo("u32")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr("Unexpected group_sizes type")));
}

TEST_F(GpuProfilerTest, InitializesRaggedDotGroupSizesPerBatchElement) {
  // A batched kRaggedNonContracting dot takes one [G] row of group sizes per
  // batch element, so the whole [B, G] buffer has to be filled.
  constexpr absl::string_view kBatchedRaggedDotHlo = R"(
    HloModule module

    fusion1 {
      p0 = bf16[2,130,32] parameter(0)
      p1 = bf16[4,2,32,16] parameter(1)
      p2 = s32[2,4] parameter(2)
      ROOT rd = bf16[2,130,16] ragged-dot(p0, p1, p2),
        lhs_batch_dims={0}, rhs_batch_dims={1},
        lhs_contracting_dims={2}, rhs_contracting_dims={2},
        lhs_ragged_dims={1}, rhs_group_dims={0}
    }

    ENTRY main {
      p0 = bf16[2,130,32] parameter(0)
      p1 = bf16[4,2,32,16] parameter(1)
      p2 = s32[2,4] parameter(2)
      ROOT fusion = bf16[2,130,16] fusion(p0, p1, p2), kind=kCustom,
        calls=fusion1,
        backend_config={"fusion_backend_config": {kind: "__triton"}}
    })";
  ASSERT_OK_AND_ASSIGN(Literal group_sizes,
                       ReadInitializedGroupSizes(kBatchedRaggedDotHlo));
  EXPECT_EQ(group_sizes, LiteralUtil::CreateR2<int32_t>(
                             {{33, 33, 32, 32}, {33, 33, 32, 32}}));
}

TEST_F(GpuProfilerTest, LeavesRaggedDotGroupSizesAloneWhenThereAreNoGroups) {
  // Nothing to balance across zero groups, and nothing to divide by.
  constexpr absl::string_view kNoGroupsRaggedDotHlo = R"(
    HloModule module

    fusion1 {
      p0 = bf16[128,32] parameter(0)
      p1 = bf16[0,32,16] parameter(1)
      p2 = s32[0] parameter(2)
      ROOT rd = bf16[128,16] ragged-dot(p0, p1, p2),
        lhs_contracting_dims={1}, rhs_contracting_dims={1},
        lhs_ragged_dims={0}, rhs_group_dims={0}
    }

    ENTRY main {
      p0 = bf16[128,32] parameter(0)
      p1 = bf16[0,32,16] parameter(1)
      p2 = s32[0] parameter(2)
      ROOT fusion = bf16[128,16] fusion(p0, p1, p2), kind=kCustom,
        calls=fusion1,
        backend_config={"fusion_backend_config": {kind: "__triton"}}
    })";
  ASSERT_OK_AND_ASSIGN(Literal group_sizes,
                       ReadInitializedGroupSizes(kNoGroupsRaggedDotHlo));
  EXPECT_EQ(group_sizes.element_count(), 0);
}

class GpuProfilerSingleRunEarlyStopTest
    : public GpuProfilerTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(GpuProfilerSingleRunEarlyStopTest,
       StopsAfterFirstRunAndFlushesCacheOnce) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  const int duration_ns = GetParam();
  MockExecutable mock_executable(module, duration_ns);
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true);
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 3;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(mock_executable.num_profiled_runs(), 1);
  EXPECT_EQ(CountCacheFlushAllocations(allocator, stream_exec_), 1);
  EXPECT_EQ(profile.duration, absl::Nanoseconds(duration_ns));
  EXPECT_EQ(profile.output_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {}));
}

INSTANTIATE_TEST_SUITE_P(GpuProfilerSingleRunEarlyStopTest,
                         GpuProfilerSingleRunEarlyStopTest,
                         ::testing::Values(500'000, 0, -1));

TEST_F(GpuProfilerTest,
       CompetitiveMarginEarlyStoppingRetainsContendersAndStopsDominated) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true);
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 3;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);

  // Candidate 1 (first candidate on `buffers`, <500 us): runs all 3 passes and,
  // once validated, establishes `best_duration_ns = 20,000 ns`. At
  // `best = 20,000 ns` and `N = 3`, the 3-sigma contender upper bound is
  // ~25,533 ns.
  MockExecutable leader(module, std::vector<int>{22'000, 19'000, 19'000});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&leader));
  ASSERT_OK_AND_ASSIGN(ProfileResult leader_profile,
                       profiler->Profile(&leader, *buffers));
  EXPECT_EQ(leader.num_profiled_runs(), 3);
  EXPECT_EQ(leader_profile.duration, absl::Nanoseconds(20'000));
  EXPECT_EQ(CountCacheFlushAllocations(allocator, stream_exec_), 3);
  profiler->RecordValidatedCandidate(*buffers, leader_profile.duration);

  // Candidate 2 (`first_run = 26,000 ns > 25,533 ns`): outside the competitive
  // margin, so it stops after `run == 0` and performs only 1 cache flush.
  MockExecutable dominated(module, std::vector<int>{26'000, 18'000, 18'000});
  ASSERT_OK_AND_ASSIGN(ProfileResult dominated_profile,
                       profiler->Profile(&dominated, *buffers));
  EXPECT_EQ(dominated.num_profiled_runs(), 1);
  EXPECT_EQ(dominated_profile.duration, absl::Nanoseconds(26'000));
  EXPECT_EQ(CountCacheFlushAllocations(allocator, stream_exec_), 4);

  // Candidate 3 (`first_run = 24,500 ns <= 25,533 ns` due to a first-run idle
  // spike, steady state 17,300 ns): retained within the competitive margin,
  // runs all 3 passes, and beats `leader` with mean 19,700 ns.
  MockExecutable contender(module, std::vector<int>{24'500, 17'300, 17'300});
  ASSERT_OK_AND_ASSIGN(ProfileResult contender_profile,
                       profiler->Profile(&contender, *buffers));
  EXPECT_EQ(contender.num_profiled_runs(), 3);
  EXPECT_EQ(contender_profile.duration, absl::Nanoseconds(19'700));
  EXPECT_EQ(CountCacheFlushAllocations(allocator, stream_exec_), 7);
  profiler->RecordValidatedCandidate(*buffers, contender_profile.duration);

  // A new `InputBuffers` instance resets the competitive baseline for the next
  // HLO instruction.
  MockExecutable next_instr(module, std::vector<int>{40'000, 38'000, 39'000});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> next_buffers,
                       profiler->CreateInputBuffers(&next_instr));
  ASSERT_OK_AND_ASSIGN(ProfileResult next_profile,
                       profiler->Profile(&next_instr, *next_buffers));
  EXPECT_EQ(next_instr.num_profiled_runs(), 3);
  EXPECT_EQ(next_profile.duration, absl::Nanoseconds(39'000));
  EXPECT_EQ(CountCacheFlushAllocations(allocator, stream_exec_), 10);
}

TEST_F(GpuProfilerTest, ZeroOrFailedCandidatesDoNotPoisonCompetitiveBaseline) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true);
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 3;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);

  MockExecutable zero_exec(module, 0);
  MockExecutable failing_exec(module, 1'000, /*should_fail=*/true);
  MockExecutable valid_exec(module, std::vector<int>{30'000, 27'000, 27'000});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&zero_exec));
  ASSERT_OK_AND_ASSIGN(ProfileResult zero_profile,
                       profiler->Profile(&zero_exec, *buffers));
  profiler->RecordValidatedCandidate(*buffers, zero_profile.duration);
  EXPECT_THAT(profiler->Profile(&failing_exec, *buffers),
              StatusIs(absl::StatusCode::kInternal));

  ASSERT_OK_AND_ASSIGN(ProfileResult valid_profile,
                       profiler->Profile(&valid_exec, *buffers));
  EXPECT_EQ(valid_exec.num_profiled_runs(), 3);
  EXPECT_EQ(valid_profile.duration, absl::Nanoseconds(28'000));
}

TEST_F(GpuProfilerTest, UnvalidatedCandidatesDoNotLowerCompetitiveBaseline) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true);
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 3;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);

  // A fast candidate that later fails validation (or is excluded from
  // selection) is profiled but never reported via `RecordValidatedCandidate`.
  MockExecutable invalid_fast(module, std::vector<int>{10'000, 10'000, 10'000});
  MockExecutable valid(module, std::vector<int>{20'000, 20'000, 20'000});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&invalid_fast));
  ASSERT_OK(profiler->Profile(&invalid_fast, *buffers));
  EXPECT_EQ(invalid_fast.num_profiled_runs(), 3);

  // Without a validated baseline, `valid` (20,000 ns, well above the
  // ~15,300 ns contender bound around 10,000 ns) still runs all passes.
  ASSERT_OK_AND_ASSIGN(ProfileResult valid_profile,
                       profiler->Profile(&valid, *buffers));
  EXPECT_EQ(valid.num_profiled_runs(), 3);
  EXPECT_EQ(valid_profile.duration, absl::Nanoseconds(20'000));
}

TEST_F(GpuProfilerTest, CompetitivePruningRequiresActiveCacheFlushing) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  ProfileOptions options;
  options.flush_caches = false;
  options.num_timed_runs = 3;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_NE(profiler, nullptr);

  MockExecutable leader(module, std::vector<int>{20'000, 20'000, 20'000});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&leader));
  ASSERT_OK_AND_ASSIGN(ProfileResult leader_profile,
                       profiler->Profile(&leader, *buffers));
  profiler->RecordValidatedCandidate(*buffers, leader_profile.duration);

  // Without L2 flushing, `run == 0` is cold relative to L2-hot runs `1..N-1`,
  // so a candidate outside the competitive margin on `run == 0` (26,000 ns >
  // ~25,533 ns) still runs all passes and can win on steady-state timing.
  MockExecutable cold_first_run(module,
                                std::vector<int>{26'000, 15'500, 15'500});
  ASSERT_OK_AND_ASSIGN(ProfileResult cold_profile,
                       profiler->Profile(&cold_first_run, *buffers));
  EXPECT_EQ(cold_first_run.num_profiled_runs(), 3);
  EXPECT_EQ(cold_profile.duration, absl::Nanoseconds(19'000));

  // The >= 500 us cap still applies without L2 flushing.
  MockExecutable long_exec(module, std::vector<int>{600'000, 1, 1});
  ASSERT_OK_AND_ASSIGN(ProfileResult long_profile,
                       profiler->Profile(&long_exec, *buffers));
  EXPECT_EQ(long_exec.num_profiled_runs(), 1);
  EXPECT_EQ(long_profile.duration, absl::Nanoseconds(600'000));
}

TEST_F(GpuProfilerTest, DisablingAdaptiveEarlyStoppingRunsAllTimedPasses) {
  constexpr absl::string_view kHloModule = R"hlo(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true);
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 3;
  options.adaptive_early_stopping = false;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);

  MockExecutable long_exec(module, std::vector<int>{600'000, 580'000, 590'000});
  MockExecutable fast_exec(module, std::vector<int>{10'000, 10'000, 10'000});
  MockExecutable slow_exec(module, std::vector<int>{100'000, 95'000, 96'000});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&long_exec));
  ASSERT_OK_AND_ASSIGN(ProfileResult long_profile,
                       profiler->Profile(&long_exec, *buffers));
  EXPECT_EQ(long_exec.num_profiled_runs(), 3);
  EXPECT_EQ(long_profile.duration, absl::Nanoseconds(590'000));

  ASSERT_OK_AND_ASSIGN(ProfileResult fast_profile,
                       profiler->Profile(&fast_exec, *buffers));
  // With adaptive early stopping disabled, this must not establish a
  // competitive baseline that would otherwise prune `slow_exec`.
  profiler->RecordValidatedCandidate(*buffers, fast_profile.duration);
  ASSERT_OK_AND_ASSIGN(ProfileResult slow_profile,
                       profiler->Profile(&slow_exec, *buffers));
  EXPECT_EQ(slow_exec.num_profiled_runs(), 3);
  EXPECT_EQ(slow_profile.duration, absl::Nanoseconds(97'000));
  EXPECT_EQ(CountCacheFlushAllocations(allocator, stream_exec_), 9);
}

TEST_F(GpuProfilerTest, RejectsCandidateThatWritesPastAllocatedBuffer) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/1000,
                                 /*should_fail=*/false,
                                 /*write_past_allocated_buffer=*/true);

  ProfileOptions options;
  options.redzone_padding_bytes = 1024;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  EXPECT_THAT(profiler->Profile(&mock_executable, *buffers),
              StatusIs(absl::StatusCode::kInternal,
                       ::testing::HasSubstr("Redzone mismatch")));
}

TEST_F(GpuProfilerTest, FailingExecutablesReturnStatus) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/0,
                                 /*should_fail=*/true);

  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  EXPECT_THAT(profiler->Profile(&mock_executable, *buffers),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(GpuProfilerTest, ReportsMeanDurationOfTimedRuns) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, std::vector<int>{1000, 2000, 6000});
  ProfileOptions options;
  options.num_timed_runs = 3;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(3000));
  EXPECT_EQ(mock_executable.num_profiled_runs(), 3);
  EXPECT_EQ(profile.output_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {}));
}

TEST_F(GpuProfilerTest, DoesNotAllocateCacheFlushBufferByDefault) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  RecordingAllocator allocator(allocator_.get());
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), &allocator);
  ASSERT_NE(profiler, nullptr);
  EXPECT_THAT(allocator.allocations(), ::testing::IsEmpty());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
  const uint64_t min_flush_bytes = std::max<uint64_t>(
      2 * stream_exec_->GetDeviceDescription().l2_cache_size(), 64ULL << 20);
  EXPECT_LT(allocator.peak_live_bytes(), min_flush_bytes);
}

TEST_F(GpuProfilerTest, FlushesCachesBeforeTimedRuns) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, std::vector<int>{1000, 3000});
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/false);
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 2;
  options.redzone_padding_bytes = 1024;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);
  EXPECT_THAT(allocator.allocations(), ::testing::IsEmpty());

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(2000));
  EXPECT_EQ(mock_executable.num_profiled_runs(), 2);
  EXPECT_OK(profiler->CheckInputBuffers(*buffers));

  const uint64_t min_flush_bytes = std::max<uint64_t>(
      2 * stream_exec_->GetDeviceDescription().l2_cache_size(), 64ULL << 20);
  int flush_alloc_count = 0;
  for (const auto& alloc : allocator.allocations()) {
    if (alloc.size >= min_flush_bytes) {
      ++flush_alloc_count;
      EXPECT_FALSE(alloc.retry_on_failure);
    }
  }
  // Synchronous allocators allocate the cache-flush buffer lazily once per
  // instruction group and reuse it across timed runs.
  EXPECT_EQ(flush_alloc_count, 1);
}

TEST_F(GpuProfilerTest,
       ReleasesCacheFlushBufferBetweenRunsWithAsyncDeallocation) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = s32[] parameter(0)
      ROOT copy = s32[] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, std::vector<int>{1000, 3000});
  const uint64_t min_flush_bytes = std::max<uint64_t>(
      2 * stream_exec_->GetDeviceDescription().l2_cache_size(), 64ULL << 20);
  // Budget fits input redzones (16 MiB) + warm-up output redzones (16 MiB) =
  // 32 MiB, and fits input redzones (16 MiB) + transient cache-flush scratch,
  // but CANNOT hold input redzones + warm-up redzones + cache-flush scratch
  // simultaneously (32 MiB + min_flush_bytes > max_live_bytes).
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true,
                               /*max_live_bytes=*/min_flush_bytes + (20 << 20));
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 2;
  options.redzone_padding_bytes = 8 << 20;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(2000));
  EXPECT_EQ(mock_executable.num_profiled_runs(), 2);

  int flush_alloc_count = 0;
  for (const auto& alloc : allocator.allocations()) {
    if (alloc.size >= min_flush_bytes) {
      ++flush_alloc_count;
      EXPECT_FALSE(alloc.retry_on_failure);
    }
  }
  EXPECT_EQ(flush_alloc_count, 2);
  // After Profile() returns, only input buffers + committed output remain live.
  EXPECT_LT(allocator.live_bytes(), min_flush_bytes);
}

TEST_F(GpuProfilerTest,
       FallsBackToWarmCacheWhenPoolTooSmallWithoutStarvingInputs) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      p0 = s32[] parameter(0)
      ROOT copy = s32[] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, std::vector<int>{1000, 3000});
  const uint64_t min_flush_bytes = std::max<uint64_t>(
      2 * stream_exec_->GetDeviceDescription().l2_cache_size(), 64ULL << 20);
  // Pool is slightly larger than min_flush_bytes (e.g. 128 MiB BFCAllocator on
  // H100 where min_flush_bytes = 100 MiB), so an eager persistent allocation in
  // Create() would starve the 16 MiB input redzones in CreateInputBuffers().
  RecordingAllocator allocator(allocator_.get(),
                               /*allows_async_deallocation=*/true,
                               /*max_live_bytes=*/min_flush_bytes + (4 << 20));
  ProfileOptions options;
  options.flush_caches = true;
  options.num_timed_runs = 2;
  options.redzone_padding_bytes = 8 << 20;
  auto profiler = GpuProfiler::Create(stream_exec_, options, &allocator);
  ASSERT_NE(profiler, nullptr);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                       profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(2000));
  EXPECT_EQ(mock_executable.num_profiled_runs(), 2);

  int flush_attempt_count = 0;
  for (const auto& alloc : allocator.allocations()) {
    if (alloc.size >= min_flush_bytes) {
      ++flush_attempt_count;
      EXPECT_FALSE(alloc.retry_on_failure);
    }
  }
  EXPECT_EQ(flush_attempt_count, 1);
}

class GpuProfilerTestWithRedzonePadding
    : public GpuProfilerTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(GpuProfilerTestWithRedzonePadding, CheckInputBuffers) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  ProfileOptions options;
  options.redzone_padding_bytes = GetParam();
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                       profiler->CreateInputBuffers(&mock_executable));
  EXPECT_OK(profiler->CheckInputBuffers(*buffers));
}

INSTANTIATE_TEST_SUITE_P(GpuProfilerTestWithRedzonePadding,
                         GpuProfilerTestWithRedzonePadding,
                         ::testing::Values(0, 1024));

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/1));
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/1));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/1));
  ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                       CreateTestBuffer(allocator.get(), stream_exec_,
                                        stream.get(), /*value=*/2));
  auto status = profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0);
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Mismatch count:"));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWithTupleShapeAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer reference,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWithTupleShapeAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer reference,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output_error_in_first_element,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/0, /*value2=*/2));
  ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output_error_in_second_element,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/3));
  EXPECT_THAT(profiler->CheckOutputBuffer(output_error_in_first_element,
                                          reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(profiler->CheckOutputBuffer(output_error_in_second_element,
                                          reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(GpuProfilerTest, CheckScratchBytesArePopulated) {
  constexpr int64_t kScratchBytes = 26738688;
  std::string hlo_text = absl::Substitute(R"hlo(
    HloModule gemm_fusion_dot.1
    ENTRY %entry_computation (lhs: bf16[3072,512], rhs: bf16[3840,3072]) -> bf16[512,3840] {
      %lhs = bf16[3072,512]{1,0} parameter(0)
      %rhs = bf16[3840,3072]{1,0} parameter(1)
      %custom-call.1 = (bf16[512,3840]{0,1}, s8[$1]{0}) custom-call(%lhs, %rhs), custom_call_target="__cublas$$lt$$matmul", $0
      ROOT %get-tuple-element = bf16[512,3840]{0,1} get-tuple-element(%custom-call.1), index=0
    }
  )hlo",
                                          kGemmBackendConfig, kScratchBytes);
  EXPECT_THAT(GetScratchBytes(hlo_text), IsOkAndHolds(kScratchBytes));
}

TEST_F(GpuProfilerTest, CheckScratchBytesAreDeDuplicated) {
  constexpr int64_t kScratchBytes = 26738688;
  std::string hlo_text = absl::Substitute(R"hlo(
    HloModule gemm_fusion_dot.2
    ENTRY %entry_computation (lhs: bf16[3072,512], rhs: bf16[3840,3072]) -> (bf16[512,3840], bf16[512,3840]) {
      %lhs = bf16[3072,512]{1,0} parameter(0)
      %rhs = bf16[3840,3072]{1,0} parameter(1)
      %custom-call.1 = (bf16[512,3840]{0,1}, s8[$1]{0}) custom-call(%lhs, %rhs), custom_call_target="__cublas$$lt$$matmul", $0
      %val1 = bf16[512,3840]{0,1} get-tuple-element(%custom-call.1), index=0

      %custom-call.2 = (bf16[512,3840]{0,1}, s8[$1]{0}) custom-call(%lhs, %rhs), custom_call_target="__cublas$$lt$$matmul", $0
      %val2 = bf16[512,3840]{0,1} get-tuple-element(%custom-call.2), index=0

      ROOT %tuple = (bf16[512,3840]{0,1}, bf16[512,3840]{0,1}) tuple(%val1, %val2)
    }
  )hlo",
                                          kGemmBackendConfig, kScratchBytes);

  EXPECT_THAT(GetScratchBytes(hlo_text), IsOkAndHolds(kScratchBytes));
}

}  // namespace

}  // namespace gpu
}  // namespace xla

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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

using absl_testing::StatusIs;

class MockExecutable : public Executable {
 public:
  explicit MockExecutable(std::shared_ptr<HloModule> module, int duration_ns,
                          bool should_fail = false)
      : Executable(module),
        duration_ns_(duration_ns),
        should_fail_(should_fail) {}
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    if (should_fail_) {
      return absl::InternalError("MockExecutable failed as requested.");
    }
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    if (profile != nullptr) {
      profile->set_compute_time_ns(duration_ns_);
    }
    const Shape& result_shape =
        module().entry_computation()->root_instruction()->shape();
    return ExecutionOutput(result_shape, result_shape,
                           run_options->run_options().allocator(),
                           run_options->run_options().device_ordinal());
  }

 private:
  int duration_ns_;
  bool should_fail_;
};

absl::StatusOr<ScopedShapedBuffer> CreateTestBuffer(
    se::DeviceAddressAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  TF_ASSIGN_OR_RETURN(auto* transfer_manager, TransferManager::GetForPlatform(
                                                  stream_exec->GetPlatform()));
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer output,
      transfer_manager->AllocateScopedShapedBuffer(
          test_shape, allocator, stream_exec->device_ordinal()));
  Literal literal = LiteralUtil::CreateR0<int32_t>(value);
  TF_RETURN_IF_ERROR(
      transfer_manager->TransferLiteralToDevice(stream, literal, output));
  return output;
}

absl::StatusOr<ScopedShapedBuffer> CreateTupleTestBuffer(
    se::DeviceAddressAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value1, int32_t value2) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  Shape test_shape_tuple = ShapeUtil::MakeTupleShape({test_shape, test_shape});
  TF_ASSIGN_OR_RETURN(auto* transfer_manager, TransferManager::GetForPlatform(
                                                  stream_exec->GetPlatform()));
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer output,
      transfer_manager->AllocateScopedShapedBuffer(
          test_shape_tuple, allocator, stream_exec->device_ordinal()));
  Literal literal1 = LiteralUtil::CreateR0<int32_t>(value1);
  Literal literal2 = LiteralUtil::CreateR0<int32_t>(value2);
  Literal tuple_literal = LiteralUtil::MakeTuple({&literal1, &literal2});
  TF_RETURN_IF_ERROR(
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
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(&mock_executable));
  TF_ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                          profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
  EXPECT_EQ(profile.output_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {}));
  EXPECT_EQ(profile.scratch_bytes, 0);
}

TEST_F(GpuProfilerTest, ProfileWithTupleOutput) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = (s32[], s32[]) tuple(s32[] constant(1), s32[] constant(2))
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(&mock_executable));
  TF_ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                          profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.output_buffer->on_device_shape(),
            ShapeUtil::MakeShape(S32, {}));
}

TEST_F(GpuProfilerTest, FailingExecutablesReturnStatus) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, /*duration_ns=*/0,
                                 /*should_fail=*/true);

  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(&mock_executable));
  EXPECT_THAT(profiler->Profile(&mock_executable, *buffers),
              StatusIs(absl::StatusCode::kInternal));
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
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  ProfileOptions options;
  options.redzone_padding_bytes = GetParam();
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(&mock_executable));
  TF_EXPECT_OK(profiler->CheckInputBuffers(*buffers));
}

INSTANTIATE_TEST_SUITE_P(GpuProfilerTestWithRedzonePadding,
                         GpuProfilerTestWithRedzonePadding,
                         ::testing::Values(0, 1024));

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/1));
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/1));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/1));
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/2));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWithTupleShapeAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer reference,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWithTupleShapeAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options, allocator_.get());

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_exec_);
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer reference,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/1, /*value2=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedShapedBuffer output_error_in_first_element,
      CreateTupleTestBuffer(allocator.get(), stream_exec_, stream.get(),
                            /*value1=*/0, /*value2=*/2));
  TF_ASSERT_OK_AND_ASSIGN(
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

TEST_F(GpuProfilerTest, CheckScratchBytesArePopulatedUsingBufferAssignment) {
  constexpr absl::string_view kHloModule = R"(
HloModule gemm_fusion_dot.1, is_scheduled=true, entry_computation_layout={(bf16[32,120,6,512]{3,2,1,0}, f32[3072,512]{1,0})->bf16[3840,512]{1,0}}, frontend_attributes={fingerprint_before_lhs="40f912baf5b53a4f75b1ba9b3442042f"}

%wrapped_convert_computation (param_0: f32[3072,512]) -> bf16[3072,512] {
  %param_0 = f32[3072,512]{1,0} parameter(0)
  ROOT %convert.1 = bf16[3072,512]{1,0} convert(%param_0)
}

ENTRY %entry_computation (transpose.562: bf16[32,120,6,512], Arg_1.2: f32[3072,512]) -> bf16[3840,512] {
  %Arg_1.2 = f32[3072,512]{1,0} parameter(1)
  %transpose.562 = bf16[32,120,6,512]{3,2,1,0} parameter(0)
  %bitcast.0 = bf16[1,32,120,6,512]{4,3,2,1,0} bitcast(%transpose.562)
  %bitcast.1 = bf16[3840,3072]{1,0} bitcast(%bitcast.0)
  %wrapped_convert = bf16[3072,512]{1,0} fusion(%Arg_1.2), kind=kLoop, calls=%wrapped_convert_computation
  %custom-call.1 = (bf16[512,3840]{0,1}, s8[26738688]{0}) custom-call(%wrapped_convert, %bitcast.1), custom_call_target="__cublas$gemm", backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"gemm_backend_config":{"alpha_real":1,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["0"],"rhs_contracting_dimensions":["1"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"alpha_imag":0,"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","lhs_stride":"1572864","rhs_stride":"11796480","grad_x":false,"grad_y":false,"damax_output":false},"force_earliest_schedule":false,"reification_cost":[]}
  %get-tuple-element = bf16[512,3840]{0,1} get-tuple-element(%custom-call.1), index=0
  ROOT %bitcast.2 = bf16[3840,512]{1,0} bitcast(%get-tuple-element)
})";
  NVPTXCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(auto gpu_executable,
                          compiler.RunBackend(std::move(module), stream_exec_,
                                              GpuCompiler::CompileOptions()));
  auto profiler =
      GpuProfiler::Create(stream_exec_, ProfileOptions(), allocator_.get());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(gpu_executable.get()));
  TF_ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                          profiler->Profile(gpu_executable.get(), *buffers));
  EXPECT_EQ(profile.scratch_bytes, 26738688);
}

}  // namespace

}  // namespace gpu
}  // namespace xla

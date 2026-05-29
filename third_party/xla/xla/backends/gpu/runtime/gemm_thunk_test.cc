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

#include "xla/backends/gpu/runtime/gemm_thunk.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

namespace xla::gpu {

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Returns true if the GPU supports CUDA graph tracing (requires CUDA 12.3+).
static bool SupportsCudaGraphTracing(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) return false;
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         stream_executor::SemanticVersion(12, 3, 0);
}

namespace {

TEST(GemmThunkTest, ProtoRoundTrip) {
  constexpr absl::string_view kProtoText = R"pb(
    thunk_info { profile_annotation: "gemm_thunk_test_profile" }
    gemm_thunk {
      gemm_config {
        lhs_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 2
          num_cols: 3
          batch_size: 1
          leading_dim_stride: 3
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        rhs_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 3
          num_cols: 4
          batch_size: 1
          leading_dim_stride: 4
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        c_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 2
          num_cols: 4
          batch_size: 1
          leading_dim_stride: 4
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        output_layout {
          order: ORDER_ROW_MAJOR
          num_rows: 2
          num_cols: 4
          batch_size: 1
          leading_dim_stride: 4
          transpose: BLAS_NO_TRANSPOSE
          dtype: F32
        }
        alpha_real: 1.0
        alpha_imag: 0.0
        beta: 0.5
        precision_algorithm: ALG_UNSET
        compute_type: BLAS_COMPUTATION_TYPE_F32
      }
      lhs_buffer { offset: 10 size: 24 buffer_allocation_index: 0 }
      rhs_buffer { offset: 20 size: 48 buffer_allocation_index: 1 }
      output_buffer { offset: 30 size: 32 buffer_allocation_index: 2 }
      workspace { offset: 40 size: 1024 buffer_allocation_index: 3 }
      deterministic: true
    }
  )pb";

  ThunkProto original_thunk_proto = ParseTextProtoOrDie<ThunkProto>(kProtoText);

  std::vector<BufferAllocation> buffer_allocations;
  buffer_allocations.emplace_back(/*index=*/0, /*size=*/100, /*color=*/10);
  buffer_allocations.emplace_back(/*index=*/1, /*size=*/200, /*color=*/11);
  buffer_allocations.emplace_back(/*index=*/2, /*size=*/150, /*color=*/12);
  buffer_allocations.emplace_back(/*index=*/3, /*size=*/2048, /*color=*/13);

  const GemmThunkProto& original_gemm_thunk_proto =
      original_thunk_proto.gemm_thunk();

  TF_ASSERT_OK_AND_ASSIGN(
      Thunk::ThunkInfo thunk_info_from_proto,
      Thunk::ThunkInfo::FromProto(original_thunk_proto.thunk_info()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GemmThunk> gemm_thunk,
      GemmThunk::FromProto(thunk_info_from_proto, original_gemm_thunk_proto,
                           buffer_allocations));
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto round_tripped_thunk_proto,
                          gemm_thunk->ToProto());
  EXPECT_THAT(round_tripped_thunk_proto, EqualsProto(original_thunk_proto));
}
// ===========================================================================
// Command buffer tests (Record)
// ===========================================================================

TEST(GemmThunkTest, RecordCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!SupportsCudaGraphTracing(executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto trace_stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 3;
  int64_t workspace_length = 1024 * 1024;

  // lhs = [1, 2, 3, 4 / 5, 6, 7, 8], rhs = all-ones [4x3]
  se::DeviceAddress<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1.0f);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(workspace_length / sizeof(float));
  TF_ASSERT_OK(stream->MemZero(&workspace, workspace_length));

  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/3, workspace_length, /*color=*/0);

  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0,
                                          workspace_length);

  TF_ASSERT_OK_AND_ASSIGN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          /*scale_mode=*/se::gpu::ScaleMode::kNone,
          executor->GetDeviceDescription().gpu_compute_capability()));

  GemmThunk thunk(Thunk::ThunkInfo{}, std::move(config), slice_lhs, slice_rhs,
                  slice_out, slice_workspace, /*deterministic=*/false);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(),
      /*command_buffer_trace_stream=*/trace_stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(params, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Expected: lhs [2x4] * rhs [4x3, all-ones] = [[10,10,10],[26,26,26]]
  std::vector<float> dst(6, 0.0f);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));
  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));
}

TEST(GemmThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();

  if (!SupportsCudaGraphTracing(executor)) {
    GTEST_SKIP() << "CUDA graph tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto trace_stream, executor->CreateStream());

  int64_t lhs_length = sizeof(float) * 2 * 4;
  int64_t rhs_length = sizeof(float) * 4 * 3;
  int64_t out_length = sizeof(float) * 2 * 3;
  int64_t workspace_length = 1024 * 1024;

  se::DeviceAddress<float> lhs = executor->AllocateArray<float>(2 * 4);
  std::vector<float> lhs_arr{1, 2, 3, 4, 5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&lhs, lhs_arr.data(), lhs_length));

  se::DeviceAddress<float> rhs = executor->AllocateArray<float>(4 * 3);
  std::vector<float> rhs_arr(12, 1.0f);
  TF_ASSERT_OK(stream->Memcpy(&rhs, rhs_arr.data(), rhs_length));

  se::DeviceAddress<float> out = executor->AllocateArray<float>(2 * 3);
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  se::DeviceAddress<float> workspace =
      executor->AllocateArray<float>(workspace_length / sizeof(float));
  TF_ASSERT_OK(stream->MemZero(&workspace, workspace_length));

  BufferAllocation alloc_lhs(/*index=*/0, lhs_length, /*color=*/0);
  BufferAllocation alloc_rhs(/*index=*/1, rhs_length, /*color=*/0);
  BufferAllocation alloc_out(/*index=*/2, out_length, /*color=*/0);
  BufferAllocation alloc_workspace(/*index=*/3, workspace_length, /*color=*/0);

  BufferAllocation::Slice slice_lhs(&alloc_lhs, 0, lhs_length);
  BufferAllocation::Slice slice_rhs(&alloc_rhs, 0, rhs_length);
  BufferAllocation::Slice slice_out(&alloc_out, 0, out_length);
  BufferAllocation::Slice slice_workspace(&alloc_workspace, 0,
                                          workspace_length);

  TF_ASSERT_OK_AND_ASSIGN(
      GemmConfig config,
      GemmConfig::For(
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 4}), {}, {1},
          ShapeUtil::MakeShape(PrimitiveType::F32, {4, 3}), {}, {0},
          ShapeUtil::MakeShape(PrimitiveType::F32, {2, 3}), 1.0, 0.0, 0.0,
          PrecisionConfig::ALG_UNSET, std::nullopt,
          se::blas::kDefaultComputePrecision, false, false,
          /*scale_mode=*/se::gpu::ScaleMode::kNone,
          executor->GetDeviceDescription().gpu_compute_capability()));

  GemmThunk thunk(Thunk::ThunkInfo{}, std::move(config), slice_lhs, slice_rhs,
                  slice_out, slice_workspace, /*deterministic=*/false);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({lhs, rhs, out, workspace}, 0, &allocator);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(
      thunk.Initialize({executor, source, &allocations, stream.get()}));

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(),
      /*command_buffer_trace_stream=*/trace_stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  // First recording: RecordCreate.
  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(params, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<float> dst(6, 0.0f);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));
  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));

  // Transition to update state; zero output to confirm re-execution.
  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK(stream->MemZero(&out, out_length));

  // Second recording: RecordUpdate with same buffers → cache hit, same cmd.
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params, record_params, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused (cache hit)
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::fill(dst.begin(), dst.end(), 0.0f);
  TF_ASSERT_OK(stream->Memcpy(dst.data(), out, out_length));
  ASSERT_EQ(dst, std::vector<float>({10, 10, 10, 26, 26, 26}));
}

}  // namespace
}  // namespace xla::gpu

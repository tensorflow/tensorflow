/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"  // IWYU pragma: keep - cudnn frontend headers are not hermetic
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_interface.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_properties.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_utils.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::gpu {

using MemoryAccess = BufferUse::MemoryAccess;
using KernelArgsPacking = se::MultiKernelLoaderSpec::KernelArgsPacking;

namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Give a short aliases to execution threads.
constexpr auto s0 = ExecutionStreamId(0);

// Give a short alias to synchronization mode.
static constexpr auto serialize =
    CommandBufferCmdExecutor::SynchronizationMode::kSerialize;

}  // namespace

TEST(CommandBufferThunkTest, CuDnnCmd) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());
  se::dnn::DnnSupport& dnn_support = *stream_executor->AsDnn();

  if (dnn_support.GetVersion().value_or(se::dnn::VersionInfo{0, 0, 0}) <
      se::dnn::VersionInfo(9, 7, 0)) {
    GTEST_SKIP() << "Requires cuDNN 9.7.0 or later.";
  }

  constexpr int kDimSize = 32;
  constexpr int kTotalElements = kDimSize * kDimSize;

  se::gpu::CudnnGraph graph([]() {
    cudnn_frontend::graph::Graph graph;
    graph.set_compute_data_type(cudnn_frontend::DataType_t::INT32);
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> lhs =
        graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                         .set_dim({1, kDimSize, kDimSize})
                         .set_stride({kDimSize * kDimSize, kDimSize, 1})
                         .set_data_type(cudnn_frontend::DataType_t::INT8)
                         .set_uid(1));
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> rhs =
        graph.tensor_like(lhs);
    rhs->set_uid(2);
    graph.matmul(lhs, rhs, cudnn_frontend::graph::Matmul_attributes())
        ->set_output(true)
        .set_data_type(cudnn_frontend::DataType_t::INT32)
        .set_uid(3);
    return graph;
  }());
  int64_t workspace_size = graph.Graph().get_workspace_size();
  TF_ASSERT_OK(graph.Prepare(dnn_support, se::NumericOptions{}));
  TF_ASSERT_OK(graph.Build(dnn_support, /*plan_id=*/std::nullopt));
  EXPECT_THAT(graph.SupportsExplicitCommandBufferConstruction(),
              tsl::testing::IsOkAndHolds(true));

  std::vector<BufferAllocation::Slice> args;
  BufferAllocation alloc_input(/*index=*/0, kTotalElements, /*color=*/0);
  BufferAllocation alloc_output(/*index=*/1, kTotalElements * sizeof(int32_t),
                                /*color=*/0);

  BufferAllocation::Slice slice_input(&alloc_input, 0, kTotalElements);
  BufferAllocation::Slice slice_output(&alloc_output, 0,
                                       kTotalElements * sizeof(int32_t));

  args.reserve(4);
  args.push_back(slice_input);  // multiplying the input by itself
  args.push_back(slice_input);
  args.push_back(slice_output);

  if (workspace_size > 0) {
    BufferAllocation alloc_workspace(
        /*index=*/2, workspace_size, /*color=*/0);
    BufferAllocation::Slice slice_workspace(&alloc_workspace, 0,
                                            workspace_size);
    args.push_back(slice_workspace);
  }

  auto dnn_graph = std::make_unique<se::gpu::CudnnGraph>(std::move(graph));
  CommandBufferCmdSequence commands;
  commands.Emplace<CuDnnCmd>(
      s0, args, std::make_shared<se::dnn::LazyDnnGraph>(std::move(dnn_graph)));
  TF_ASSERT_OK_AND_ASSIGN(
      CommandBufferCmdExecutor executor,
      CommandBufferCmdExecutor::Create(std::move(commands), serialize));

  // Construct a thunk with command sequence.
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo());

  std::vector<se::DeviceMemoryBase> operands;
  operands.reserve(3);

  se::DeviceMemory<int8_t> input =
      stream_executor->AllocateArray<int8_t>(kTotalElements);
  TF_ASSERT_OK(stream->MemZero(&input, input.size()));

  se::DeviceMemory<int32_t> output0 =
      stream_executor->AllocateArray<int32_t>(kTotalElements);
  TF_ASSERT_OK(stream->Memset32(&output0, 123, output0.size()));

  operands.push_back(input);  // multiplying the input by itself
  operands.push_back(output0);

  se::DeviceMemoryBase workspace;
  if (workspace_size > 0) {
    workspace = stream_executor->Allocate(workspace_size);
    operands.push_back(workspace);
  }

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  BufferAllocations allocations(operands, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr);

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  TF_ASSERT_OK(thunk.Initialize(
      {stream_executor, source, &allocations, stream.get(), stream.get()}));

  // Execute command buffer thunk and verify that it executed a GEMM.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy output0 data back to host.
  std::vector<int32_t> dst(kTotalElements, 1);
  TF_ASSERT_OK(
      stream->Memcpy(dst.data(), output0, kTotalElements * sizeof(int32_t)));

  ASSERT_EQ(dst, std::vector<int32_t>(kTotalElements, 0));

  // Prepare buffer allocation for updating command buffer.
  se::DeviceMemory<int32_t> output1 =
      stream_executor->AllocateArray<int32_t>(kTotalElements);
  TF_ASSERT_OK(stream->Memset32(&output1, 456, output1.size()));

  // Update buffer allocation
  operands[1] = output1;
  allocations = BufferAllocations(operands, 0, &allocator);
  // Thunk execution should automatically update underlying command
  // buffer.
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  // Copy output1 data back to host.
  std::fill(dst.begin(), dst.end(), 1);
  TF_ASSERT_OK(
      stream->Memcpy(dst.data(), output1, kTotalElements * sizeof(int32_t)));

  ASSERT_EQ(dst, std::vector<int32_t>(kTotalElements, 0));
}
}  // namespace xla::gpu

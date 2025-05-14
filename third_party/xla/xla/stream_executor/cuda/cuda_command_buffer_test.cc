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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"  // IWYU pragma: keep - cudnn frontend headers are not hermetic
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_interface.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_properties.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_utils.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::cuda {
namespace {

using ::testing::Each;
using ::tsl::testing::IsOkAndHolds;

static Platform* CudaPlatform() {
  auto name = absl::AsciiStrToUpper(
      xla::PlatformUtil::CanonicalPlatformName("cuda").value());
  return PlatformManager::PlatformWithName(name).value();
}

static constexpr auto primary = CommandBuffer::Mode::kPrimary;  // NOLINT

TEST(CudaCommandBufferTest, CuDnnExplicitConstructionAndUpdateWork) {
  Platform* platform = CudaPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor->CreateStream());
  dnn::DnnSupport& dnn_support = *executor->AsDnn();

  if (dnn_support.GetVersion().value_or(dnn::VersionInfo{0, 0, 0}) <
      dnn::VersionInfo(9, 7, 0)) {
    GTEST_SKIP() << "Requires cuDNN 9.7.0 or later.";
  }

  if (executor->GetDeviceDescription().cuda_compute_capability() <
      CudaComputeCapability::Ampere()) {
    GTEST_SKIP() << "Requires at least an Ampere GPU.";
  }

  constexpr int kDimSize = 32;
  constexpr int kTotalElements = kDimSize * kDimSize;

  stream_executor::gpu::CudnnGraph graph([]() {
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
  TF_ASSERT_OK(graph.Prepare(dnn_support, NumericOptions{}));
  TF_ASSERT_OK(graph.Build(dnn_support, /*plan_id=*/std::nullopt));
  EXPECT_THAT(graph.SupportsExplicitCommandBufferConstruction(),
              IsOkAndHolds(true));

  DeviceMemory<int8_t> input = executor->AllocateArray<int8_t>(kTotalElements);
  TF_ASSERT_OK(stream->MemZero(&input, input.size()));
  DeviceMemory<int32_t> output0 =
      executor->AllocateArray<int32_t>(kTotalElements);
  DeviceMemoryBase workspace;
  std::vector<DeviceMemoryBase> operands;
  operands.reserve(4);
  operands.push_back(input);  // multiplying the input by itself
  operands.push_back(input);
  operands.push_back(output0);
  if (graph.Graph().get_workspace_size() > 0) {
    workspace = executor->Allocate(graph.Graph().get_workspace_size());
    operands.push_back(workspace);
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> cmd_buffer,
                          executor->CreateCommandBuffer(primary));
  TF_ASSERT_OK_AND_ASSIGN(
      auto* dnn_command,
      cmd_buffer->CreateDnnGraphCommand(
          graph, *stream, absl::Span<DeviceMemoryBase>(operands), {}));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  std::vector<int32_t> host_buffer(output0.ElementCount());

  // Initialize and check the output before execution.
  TF_ASSERT_OK(stream->Memset32(&output0, 123, output0.size()));
  TF_ASSERT_OK(stream->Memcpy(host_buffer.data(), output0, output0.size()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(123));

  // Run the computation.
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Check the output after execution.
  TF_ASSERT_OK(stream->Memcpy(host_buffer.data(), output0, output0.size()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(0));

  // Swap the output buffer.
  DeviceMemory<int32_t> output1 =
      executor->AllocateArray<int32_t>(kTotalElements);
  operands[2] = output1;
  executor->Deallocate(&output0);

  // Initialize and check the output before execution.
  TF_ASSERT_OK(stream->Memset32(&output1, 456, output1.size()));
  TF_ASSERT_OK(stream->Memcpy(host_buffer.data(), output1, output1.size()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(456));

  // Update the command buffer to write into the new output buffer.
  TF_ASSERT_OK(cmd_buffer->Update());
  TF_ASSERT_OK(cmd_buffer->UpdateDnnGraphCommand(
      dnn_command, graph, *stream, absl::Span<DeviceMemoryBase>(operands)));
  TF_ASSERT_OK(cmd_buffer->Finalize());

  // Run the computation.
  TF_ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Check the output after execution.
  TF_ASSERT_OK(stream->Memcpy(host_buffer.data(), output1, output1.size()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(0));
}

}  // namespace
}  // namespace stream_executor::cuda

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

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"  // IWYU pragma: keep - cudnn frontend headers are not hermetic
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_interface.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend/graph_properties.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend_utils.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_dnn.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::cuda {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Each;

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

  if (!executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastAmpere()) {
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
  ASSERT_OK(graph.Prepare(&dnn_support, executor->GetDeviceDescription(),
                          EngineOptions{/*require_determinism=*/false,
                                        /*allow_tf32=*/true,
                                        /*require_command_buffer=*/true}));
  ASSERT_OK(graph.Build(&dnn_support, executor->GetDeviceDescription(),
                        /*plan_id=*/std::nullopt));
  EXPECT_THAT(graph.SupportsExplicitCommandBufferConstruction(),
              IsOkAndHolds(true));

  DeviceAddress<int8_t> input = executor->AllocateArray<int8_t>(kTotalElements);
  ASSERT_OK(stream->MemZero(&input, input.size()));
  DeviceAddress<int32_t> output0 =
      executor->AllocateArray<int32_t>(kTotalElements);
  DeviceAddressBase workspace;
  std::vector<DeviceAddressBase> operands;
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
          graph, *stream, absl::Span<DeviceAddressBase>(operands), {}));
  ASSERT_OK(cmd_buffer->Finalize());

  std::vector<int32_t> host_buffer(output0.ElementCount());

  // Initialize and check the output before execution.
  ASSERT_OK(stream->Memset32(&output0, 123, output0.size()));
  ASSERT_OK(stream->Memcpy(host_buffer.data(), output0, output0.size()));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(123));

  // Run the computation.
  ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Check the output after execution.
  ASSERT_OK(stream->Memcpy(host_buffer.data(), output0, output0.size()));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(0));

  // Swap the output buffer.
  DeviceAddress<int32_t> output1 =
      executor->AllocateArray<int32_t>(kTotalElements);
  operands[2] = output1;
  executor->Deallocate(&output0);

  // Initialize and check the output before execution.
  ASSERT_OK(stream->Memset32(&output1, 456, output1.size()));
  ASSERT_OK(stream->Memcpy(host_buffer.data(), output1, output1.size()));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(456));

  // Update the command buffer to write into the new output buffer.
  ASSERT_OK(cmd_buffer->Update());
  ASSERT_OK(cmd_buffer->UpdateDnnGraphCommand(
      dnn_command, graph, *stream, absl::Span<DeviceAddressBase>(operands)));
  ASSERT_OK(cmd_buffer->Finalize());

  // Run the computation.
  ASSERT_OK(cmd_buffer->Submit(stream.get()));

  // Check the output after execution.
  ASSERT_OK(stream->Memcpy(host_buffer.data(), output1, output1.size()));
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(host_buffer, Each(0));
}

TEST(CudaCommandBufferTest, PdlKernelEdgeUsesProgrammaticDependency) {
  Platform* platform = CudaPlatform();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  if (!executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Requires at least a Hopper GPU.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());
  KernelLoaderSpec add_spec = ::stream_executor::gpu::GetAddI32TestKernelSpec(
                                  executor->GetPlatform()->id())
                                  .value();
  std::unique_ptr<Kernel> kernel = executor->LoadKernel(add_spec).value();
  kernel->set_use_pdl(true);

  DeviceAddress<int32_t> a = executor->AllocateArray<int32_t>(4);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> cmd_buffer,
                       executor->CreateCommandBuffer(primary));

  ASSERT_OK_AND_ASSIGN(
      const CommandBuffer::Command* first,
      cmd_buffer->CreateLaunch(
          ThreadDim(4, 1, 1), BlockDim(1, 1, 1), std::nullopt, *kernel,
          *stream_executor::PackKernelArgs(0, a, a, a), {}));
  std::array<const CommandBuffer::Command*, 1> dependencies = {first};
  ASSERT_OK_AND_ASSIGN(
      const CommandBuffer::Command* second,
      cmd_buffer->CreateLaunch(
          ThreadDim(4, 1, 1), BlockDim(1, 1, 1), std::nullopt, *kernel,
          *stream_executor::PackKernelArgs(0, a, a, a),
          absl::Span<const CommandBuffer::Command* const>(dependencies)));

  auto* first_kernel =
      dynamic_cast<const gpu::GpuCommandBuffer::GpuCommand*>(first);
  auto* second_kernel =
      dynamic_cast<const gpu::GpuCommandBuffer::GpuCommand*>(second);
  ASSERT_NE(first_kernel, nullptr);
  ASSERT_NE(second_kernel, nullptr);

  const CUgraphNode first_node =
      absl::bit_cast<CUgraphNode>(first_kernel->handle);
  const CUgraphNode second_node =
      absl::bit_cast<CUgraphNode>(second_kernel->handle);

  size_t num_dependencies = 0;
  ASSERT_EQ(cuGraphNodeGetDependencies_v2(second_node, nullptr, nullptr,
                                          &num_dependencies),
            CUDA_SUCCESS);
  ASSERT_EQ(num_dependencies, 1);

  CUgraphNode dep_node;
  CUgraphEdgeData edge_data;
  ASSERT_EQ(cuGraphNodeGetDependencies_v2(second_node, &dep_node, &edge_data,
                                          &num_dependencies),
            CUDA_SUCCESS);

  EXPECT_EQ(dep_node, first_node);
  EXPECT_EQ(edge_data.from_port, CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC);
  EXPECT_EQ(edge_data.type, CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC);
}

TEST(CudaCommandBufferTest, TraceDisallowsForbiddenOpsOnCaptureStream) {
  Platform* platform = CudaPlatform();
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  if (executor->GetDeviceDescription().driver_version() <
      SemanticVersion{12, 3, 0}) {
    GTEST_SKIP() << "Command buffer tracing is not supported";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor->CreateStream());

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CommandBuffer> cmd_buffer,
      TraceCommandBufferFactory::Create(
          executor,
          [&](Stream* capture_stream) -> absl::Status {
            EXPECT_THAT(capture_stream->BlockHostUntilDone(),
                        StatusIs(absl::StatusCode::kFailedPrecondition));
            EXPECT_THAT(capture_stream->RefreshStatus(),
                        StatusIs(absl::StatusCode::kFailedPrecondition));
            EXPECT_THAT(capture_stream->DoHostCallbackWithStatus(
                            []() { return absl::OkStatus(); }),
                        StatusIs(absl::StatusCode::kFailedPrecondition));
            return absl::OkStatus();
          },
          CommandBuffer::Mode::kPrimary));
}

TEST(CudaCommandBufferTest, LaunchClusterKernelWithClusterDimsSucceeds) {
  Platform* platform = CudaPlatform();
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  if (!executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Requires at least a Hopper GPU.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor->CreateStream());
  KernelLoaderSpec spec = stream_executor::gpu::GetMinimalClusterKernelSpec();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                       executor->LoadKernel(spec));
  DeviceAddress<uint8_t> dummy = executor->AllocateArray<uint8_t>(256);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CommandBuffer> cmd_buffer,
                       executor->CreateCommandBuffer(primary));
  ClusterDim cluster_dims{2, 1, 1};
  ASSERT_OK_AND_ASSIGN(
      const CommandBuffer::Command* cmd,
      cmd_buffer->CreateLaunch(ThreadDim(128, 1, 1), BlockDim(2, 1, 1),
                               cluster_dims, *kernel,
                               *stream_executor::PackKernelArgs(
                                   /*shmem_bytes=*/0, dummy),
                               {}));
  ASSERT_NE(cmd, nullptr);
  ASSERT_OK(cmd_buffer->Finalize());
  ASSERT_OK(cmd_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
}

}  // namespace
}  // namespace stream_executor::cuda

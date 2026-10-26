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

#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry_test_util.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry.h"
#include "xla/stream_executor/cuda/cuda_graph_annotation_registry.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"

namespace stream_executor::gpu {
namespace {

CUgraphNode NodeOf(const CommandBuffer::Command* command) {
  return absl::bit_cast<CUgraphNode>(
      absl::down_cast<const GpuCommandBuffer::GpuCommand*>(command)->handle);
}

absl::StatusOr<NodeKey> KeyOfNode(CUgraphNode node) {
  std::optional<CUgraph> graph =
      command_buffer_telemetry::GetContainingGraph(node);
  if (!graph.has_value()) {
    return absl::InternalError("Failed to query the node's graph");
  }
  std::optional<uint32_t> graph_id =
      command_buffer_telemetry::GetGraphId(*graph);
  std::optional<uint64_t> node_id = command_buffer_telemetry::GetToolsId(node);
  if (!graph_id.has_value() || !node_id.has_value()) {
    return absl::InternalError("Failed to query graph or node tools ID");
  }
  return NodeKey{*graph_id, *node_id};
}

}  // namespace

absl::StatusOr<NodeKey> KeyOf(const CommandBuffer::Command* command) {
  return KeyOfNode(NodeOf(command));
}

absl::StatusOr<NodeKey> EmbeddedKey(const CommandBuffer::Command* child,
                                    const CommandBuffer::Command* original) {
  CUgraphNode orig_node = NodeOf(original);
  std::optional<CUgraph> orig_graph =
      command_buffer_telemetry::GetContainingGraph(orig_node);
  if (!orig_graph.has_value()) {
    return absl::InternalError("Failed to query the original node's graph");
  }
  CUgraph embedded = nullptr;
  ABSL_RETURN_IF_ERROR(
      cuda::ToStatus(cuGraphChildGraphNodeGetGraph(NodeOf(child), &embedded)));

  const std::vector<CUgraphNode> orig_nodes =
      command_buffer_telemetry::GetGraphNodes(*orig_graph);
  const std::vector<CUgraphNode> nodes =
      command_buffer_telemetry::GetGraphNodes(embedded);
  if (nodes.empty() || nodes.size() != orig_nodes.size()) {
    return absl::InternalError(
        "Embedded child graph node count does not match original graph");
  }
  for (size_t i = 0; i < orig_nodes.size(); ++i) {
    if (orig_nodes[i] == orig_node) {
      return KeyOfNode(nodes[i]);
    }
  }
  return absl::NotFoundError("Original node not found in its containing graph");
}

std::optional<std::string> Lookup(const NodeKey& key,
                                  CudaGraphAnnotationRegistry::KeySpace space) {
  std::optional<std::string> annotation;
  CudaGraphAnnotationRegistry::Default().VisitNodeAnnotation(
      key.graph_id, key.node_id,
      [&](absl::string_view value) { annotation.emplace(value); }, space);
  return annotation;
}

bool PollUntil(absl::FunctionRef<bool()> done) {
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!done()) {
    if (absl::Now() > deadline) {
      return false;
    }
    absl::SleepFor(absl::Milliseconds(10));
  }
  return true;
}

void CudaCommandBufferTelemetryTest::SetUp() {
  ASSERT_OK_AND_ASSIGN(std::string platform_name,
                       xla::PlatformUtil::CanonicalPlatformName("cuda"));
  ASSERT_OK_AND_ASSIGN(
      Platform * platform,
      PlatformManager::PlatformWithName(absl::AsciiStrToUpper(platform_name)));
  ASSERT_OK_AND_ASSIGN(executor_, platform->ExecutorForDevice(0));
  const DeviceDescription& desc = executor_->GetDeviceDescription();
  if (CudaGraphAnnotationRegistry::SelectTelemetryStrategy(
          desc.compile_time_toolkit_version(), desc.driver_version(),
          desc.runtime_version()) ==
      CudaGraphAnnotationRegistry::TelemetryStrategy::kDisabled) {
    GTEST_SKIP() << "Command buffer telemetry needs CUDA 13.2+ (toolkit, "
                    "runtime and driver)";
  }
  // Entries of earlier tests may still be waiting for their cleanups. Their
  // tokens are never reissued, so late cleanups can't affect this test.
  registry().ClearForTesting();

  ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream());
  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec spec,
                       GetAddI32TestKernelSpec(executor_->GetPlatform()->id()));
  ASSERT_OK_AND_ASSIGN(kernel_, executor_->LoadKernel(spec));
  a_ = executor_->AllocateArray<int32_t>(4);
  b_ = executor_->AllocateArray<int32_t>(4);
}

void CudaCommandBufferTelemetryTest::TearDown() {
  for (DeviceAddress<int32_t>* buffer : {&a_, &b_}) {
    if (!buffer->is_null()) {
      executor_->Deallocate(buffer);
    }
  }
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
CudaCommandBufferTelemetryTest::NewPrimary() {
  return executor_->CreateCommandBuffer(CommandBuffer::Mode::kPrimary);
}

absl::StatusOr<std::unique_ptr<CommandBuffer>>
CudaCommandBufferTelemetryTest::NewNested() {
  return executor_->CreateCommandBuffer(CommandBuffer::Mode::kNested);
}

absl::StatusOr<const CommandBuffer::Command*>
CudaCommandBufferTelemetryTest::Launch(
    CommandBuffer& command_buffer,
    absl::Span<const CommandBuffer::Command* const> dependencies) {
  return command_buffer.CreateLaunch(ThreadDim(4, 1, 1), BlockDim(1, 1, 1),
                                     /*cluster_dims=*/std::nullopt, *kernel_,
                                     *PackKernelArgs(0, a_, b_, a_),
                                     dependencies);
}

absl::Status CudaCommandBufferTelemetryTest::UpdateChild(
    CommandBuffer& primary, const CommandBuffer::Command* child,
    const CommandBuffer& nested) {
  ABSL_RETURN_IF_ERROR(primary.Update());
  ABSL_RETURN_IF_ERROR(primary.UpdateChildCommand(child, nested));
  return primary.Finalize();
}

}  // namespace stream_executor::gpu

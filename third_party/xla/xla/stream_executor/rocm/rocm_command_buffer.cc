/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<hipGraph_t> CreateGraph() {
  VLOG(2) << "Create new HIP graph";
  hipGraph_t graph;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphCreate(&graph, /*flags=*/0),
                              "Failed to create HIP graph"));
  VLOG(2) << "Created HIP graph " << graph;
  return graph;
}

hipDeviceptr_t AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<hipDeviceptr_t>(mem.opaque());
}

struct BitPatternToString {
  std::string operator()(uint8_t pattern) {
    return absl::StrCat("u8:", pattern);
  }
  std::string operator()(uint16_t pattern) {
    return absl::StrCat("u16:", pattern);
  }
  std::string operator()(uint32_t pattern) {
    return absl::StrCat("u32:", pattern);
  }
};

// Broadcasts a pattern value of 1/2/4 bytes to a 4 byte value.
struct BitPatternToValue {
  std::pair<unsigned, unsigned> operator()(uint8_t pattern) {
    unsigned value = pattern;
    return {(value << 24) | (value << 16) | (value << 8) | value,
            /*element_size=*/1};
  }
  std::pair<unsigned, unsigned> operator()(uint16_t pattern) {
    unsigned value = pattern;
    return {(value << 16) | value, /*element_size=*/2};
  }
  std::pair<unsigned, unsigned> operator()(uint32_t pattern) {
    return {pattern, /*element_size=*/4};
  }
};

// Takes a list of GpuGraphNodeInfo instances and converts them to a list of
// hipGraphNode_t handles.
std::vector<hipGraphNode_t> AsNodeHandles(
    absl::Span<const GpuCommandBuffer::GpuGraphNodeInfo* const> nodes) {
  std::vector<hipGraphNode_t> handles;
  handles.reserve(nodes.size());
  for (const GpuCommandBuffer::GpuGraphNodeInfo* node : nodes) {
    handles.push_back(node->handle);
  }
  return handles;
}
}  // namespace

absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> RocmCommandBuffer::Create(
    Mode mode, GpuExecutor* parent) {
  TF_ASSIGN_OR_RETURN(hipGraph_t graph, CreateGraph());
  return std::unique_ptr<RocmCommandBuffer>(
      new RocmCommandBuffer(mode, parent, graph,
                            /*is_owned_graph=*/true));
}

std::unique_ptr<GpuCommandBuffer> RocmCommandBuffer::CreateNestedCommandBuffer(
    hipGraph_t graph) {
  return std::unique_ptr<RocmCommandBuffer>(
      new RocmCommandBuffer(Mode::kNested, parent_, graph,
                            /*is_owned_graph=*/false));
}

absl::StatusOr<GpuCommandBuffer::SetIfConditionKernel*>
RocmCommandBuffer::GetSetIfConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetIfElseConditionKernel*>
RocmCommandBuffer::GetSetIfElseConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetCaseConditionKernel*>
RocmCommandBuffer::GetSetCaseConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetForConditionKernel*>
RocmCommandBuffer::GetSetForConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetWhileConditionKernel*>
RocmCommandBuffer::GetSetWhileConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::NoOpKernel*>
RocmCommandBuffer::GetNoOpKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<RocmCommandBuffer::GpuGraphNodeInfo*>
RocmCommandBuffer::CreateMemsetNode(const Dependencies& dependencies,
                                    DeviceMemoryBase destination,
                                    BitPattern bit_pattern,
                                    size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << parent_->gpu_context()
          << "; deps: " << dependencies.size();

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  hipMemsetParams params{
      .dst = AsDevicePtr(destination),
      .elementSize = element_size,
      .height = 1,
      .pitch = 0,  // unused if height is 1
      .value = value,
      .width = num_elements,
  };

  std::vector<hipGraphNode_t> deps = AsNodeHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddMemsetNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add memset node to a HIP graph"));

  node_storage_.push_back(std::make_unique<HipGraphNode>(node_handle, this));
  return node_storage_.back().get();
}

absl::Status RocmCommandBuffer::HipGraphNode::UpdateMemsetNode(
    DeviceMemoryBase destination, BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Set memset node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements;

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  hipMemsetParams params{
      .dst = AsDevicePtr(destination),
      .elementSize = element_size,
      .height = 1,
      .pitch = 0,  // unused if height is 1
      .value = value,
      .width = num_elements,
  };

  return ToStatus(wrap::hipGraphExecMemsetNodeSetParams(command_buffer_->exec_,
                                                        handle, &params),
                  "Failed to set memset node params");
}
}  // namespace stream_executor::gpu

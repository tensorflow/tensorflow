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

#include "xla/stream_executor/cuda/cuda_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/command_buffer_kernels.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<CUgraph> CreateGraph() {
  VLOG(2) << "Create new CUDA graph";
  CUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphCreate(&graph, /*flags=*/0),
                                    "Failed to create CUDA graph"));
  VLOG(2) << "Created CUDA graph " << graph;
  return graph;
}

CUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<CUdeviceptr>(mem.opaque());
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
// CUgraphNode handles.
std::vector<CUgraphNode> AsNodeHandles(
    absl::Span<const GpuCommandBuffer::GpuGraphNodeInfo* const> nodes) {
  std::vector<CUgraphNode> handles;
  handles.reserve(nodes.size());
  for (const GpuCommandBuffer::GpuGraphNodeInfo* node : nodes) {
    handles.push_back(node->handle);
  }
  return handles;
}
}  // namespace

absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> CudaCommandBuffer::Create(
    Mode mode, GpuExecutor* parent) {
  TF_ASSIGN_OR_RETURN(CUgraph graph, CreateGraph());
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(mode, parent, graph,
                            /*is_owned_graph=*/true));
}

absl::StatusOr<CudaCommandBuffer::SetIfConditionKernel*>
CudaCommandBuffer::GetSetIfConditionKernel() {
  if (!set_if_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetIfConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_condition_kernel_,
        SetIfConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_if_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetIfElseConditionKernel*>
CudaCommandBuffer::GetSetIfElseConditionKernel() {
  if (!set_if_else_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetIfElseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_else_condition_kernel_,
        SetIfElseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_if_else_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetCaseConditionKernel*>
CudaCommandBuffer::GetSetCaseConditionKernel() {
  if (!set_case_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetCaseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_case_condition_kernel_,
        SetCaseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_case_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetForConditionKernel*>
CudaCommandBuffer::GetSetForConditionKernel() {
  if (!set_for_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetForConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_for_condition_kernel_,
        SetForConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_for_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetWhileConditionKernel*>
CudaCommandBuffer::GetSetWhileConditionKernel() {
  if (!set_while_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetWhileConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_while_condition_kernel_,
        SetWhileConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_while_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::NoOpKernel*>
CudaCommandBuffer::GetNoOpKernel() {
  if (!noop_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetNoOpKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(noop_kernel_,
                        NoOpKernel::FactoryType::Create(parent_, spec));
  }
  return &noop_kernel_;
}

std::unique_ptr<GpuCommandBuffer> CudaCommandBuffer::CreateNestedCommandBuffer(
    CUgraph graph) {
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(Mode::kNested, parent_, graph,
                            /*is_owned_graph=*/false));
}

absl::StatusOr<CudaCommandBuffer::GpuGraphNodeInfo*>
CudaCommandBuffer::CreateMemsetNode(const Dependencies& dependencies,
                                    DeviceMemoryBase destination,
                                    BitPattern bit_pattern,
                                    size_t num_elements) {
  CudaContext* cuda_context =
      tensorflow::down_cast<CudaContext*>(parent_->gpu_context());
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context->context()
          << "; deps: " << dependencies.size();

  CUDA_MEMSET_NODE_PARAMS params;
  std::memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = AsDevicePtr(destination);
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  std::vector<CUgraphNode> deps = AsNodeHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemsetNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, cuda_context->context()),
      "Failed to add memset node to a CUDA graph"));

  node_storage_.push_back(std::make_unique<CudaGraphNode>(node_handle, this));
  return node_storage_.back().get();
}

absl::Status CudaCommandBuffer::CudaGraphNode::UpdateMemsetNode(
    DeviceMemoryBase destination, BitPattern bit_pattern, size_t num_elements) {
  CudaContext* cuda_context = tensorflow::down_cast<CudaContext*>(
      command_buffer_->parent_->gpu_context());
  VLOG(2) << "Set memset node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << std::visit(BitPatternToString(), bit_pattern)
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context->context();

  CUDA_MEMSET_NODE_PARAMS params;
  std::memset(&params, 0, sizeof(params));

  auto [value, element_size] = std::visit(BitPatternToValue(), bit_pattern);

  params.dst = AsDevicePtr(destination);
  params.elementSize = element_size;
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = value;
  params.width = num_elements;

  return cuda::ToStatus(
      cuGraphExecMemsetNodeSetParams(command_buffer_->exec_, handle, &params,
                                     cuda_context->context()),
      "Failed to set memset node params");
}

}  // namespace stream_executor::gpu

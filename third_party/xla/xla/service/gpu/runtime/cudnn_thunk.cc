/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/runtime/cudnn_thunk.h"

#include "absl/status/status.h"

namespace xla {
namespace gpu {

CuDnnThunk::CuDnnThunk(std::string serialized_graph, ThunkInfo thunk_info,
                       absl::Span<const KernelArgument> kernel_arguments)
    : Thunk(Kind::kCuDnn, std::move(thunk_info)),
      serialized_graph_(std::move(serialized_graph)) {
  args_.reserve(kernel_arguments.size());
  for (const KernelArgument& kernel_argument : kernel_arguments) {
    args_.push_back(kernel_argument.slice());
  }
}

absl::Status CuDnnThunk::Initialize(const InitializeParams& params) {
  absl::Status ret = absl::OkStatus();
  absl::call_once(once_flag_, [&] {
    auto result =
        params.stream->parent()->AsDnn()->DeserializeGraph(serialized_graph_);
    std::string().swap(serialized_graph_);
    if (result.ok()) {
      graph_ = std::move(*result);
    }
    ret = result.status();
  });
  return ret;
}

absl::Status CuDnnThunk::ExecuteOnStream(const ExecuteParams& params) {
  std::vector<se::DeviceMemoryBase> buffer_args;
  buffer_args.reserve(args_.size());
  for (const BufferAllocation::Slice& arg : args_) {
    buffer_args.push_back(params.buffer_allocations->GetDeviceAddress(arg));
  }
  return graph_->Execute(*params.stream,
                         absl::Span<se::DeviceMemoryBase>(buffer_args));
}

}  // namespace gpu
}  // namespace xla

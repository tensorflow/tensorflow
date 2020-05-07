/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/tuple_thunk.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

Status TupleThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  auto n = tuple_element_buffers_.size();
  auto tuple_data = absl::make_unique<void*[]>(n);
  for (int i = 0; i < n; ++i) {
    tuple_data[i] =
        buffer_allocations.GetDeviceAddress(tuple_element_buffers_[i]).opaque();
  }

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  SafeH2DMemcpy(se::DeviceMemory<void*>(
                    buffer_allocations.GetDeviceAddress(dest_buffer_)),
                std::move(tuple_data), n, &stream,
                params.deferred_host_callbacks);
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla

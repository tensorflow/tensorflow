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

#include "xla/backends/cpu/runtime/topk_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime_topk.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

TopKThunk::TopKThunk(Info info, BufferAllocation::Slice values,
                     BufferAllocation::Slice output,
                     BufferAllocation::Slice indices, int64_t batch_size,
                     int64_t input_size, int64_t k)
    : Thunk(Thunk::Kind::kTopK, std::move(info)),
      values_buffer_(values),
      output_buffer_(output),
      indices_buffer_(indices),
      batch_size_(batch_size),
      input_size_(input_size),
      k_(k) {}

absl::StatusOr<std::unique_ptr<TopKThunk>> TopKThunk::Create(
    Info info, BufferAllocation::Slice values, BufferAllocation::Slice output,
    BufferAllocation::Slice indices, int64_t batch_size, int64_t input_size,
    int64_t k) {
  return absl::WrapUnique(new TopKThunk(std::move(info), values, output,
                                        indices, batch_size, input_size, k));
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> TopKThunk::Execute(
    const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase values,
      params.buffer_allocations->GetDeviceAddress(values_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output,
      params.buffer_allocations->GetDeviceAddress(output_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase indices,
      params.buffer_allocations->GetDeviceAddress(indices_buffer_));

  __xla_cpu_runtime_TopKF32(batch_size_, input_size_, k_,
                            reinterpret_cast<const float*>(values.opaque()),
                            reinterpret_cast<float*>(output.opaque()),
                            reinterpret_cast<int32_t*>(indices.opaque()));
  return OkExecuteEvent();
}

}  // namespace xla::cpu

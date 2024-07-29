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
#include "xla/service/cpu/runtime/fft_thunk.h"

#include <cstdint>
#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime_fft.h"
#include "xla/service/cpu/runtime_single_threaded_fft.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

FftThunk::FftThunk(Info thunk_info, bool is_multi_thread_eigen,
                   int32_t fft_type, absl::Span<const int64_t> fft_length,
                   BufferAllocation::Slice input_buffer,
                   const Shape& input_shape,
                   BufferAllocation::Slice output_buffer,
                   const Shape& output_shape)
    : Thunk(Kind::kFft, thunk_info),
      is_multi_thread_eigen_(is_multi_thread_eigen),
      is_double_precision_(input_shape.element_type() == F64 ||
                           input_shape.element_type() == C128),
      fft_type_(fft_type),
      fft_length_(fft_length.begin(), fft_length.end()),
      input_buffer_(input_buffer),
      output_buffer_(output_buffer),
      input_shape_(input_shape),
      output_shape_(output_shape) {}

absl::StatusOr<std::unique_ptr<FftThunk>> FftThunk::Create(
    Info thunk_info, bool is_multi_thread_eigen, int32_t fft_type,
    absl::Span<const int64_t> fft_length, BufferAllocation::Slice input_buffer,
    const Shape& input_shape, BufferAllocation::Slice output_buffer,
    const Shape& output_shape) {
  return absl::WrapUnique(
      new FftThunk(thunk_info, is_multi_thread_eigen, fft_type, fft_length,
                   input_buffer, input_shape, output_buffer, output_shape));
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> FftThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(input_shape_.layout()));
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(output_shape_.layout()));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase input_data,
      params.buffer_allocations->GetDeviceAddress(input_buffer_));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output_data,
      params.buffer_allocations->GetDeviceAddress(output_buffer_));

  const int fft_rank = fft_length_.size();

  // Flatten operand batches.
  absl::InlinedVector<int64_t, 4> operand_shape_flat(fft_rank + 1);
  int64_t input_batch = 1;
  int64_t input_batch_length = output_shape_.dimensions_size() - fft_rank;
  for (int i = 0; i < input_batch_length; i++) {
    input_batch *= input_shape_.dimensions(i);
  }
  operand_shape_flat[0] = input_batch;
  for (int i = 0; i < fft_rank; ++i) {
    operand_shape_flat[i + 1] = input_shape_.dimensions(i + input_batch_length);
  }

  // Args have been computed, make the call.
  if (is_multi_thread_eigen_) {
    __xla_cpu_runtime_DuccFft(nullptr,
                              reinterpret_cast<float*>(output_data.opaque()),
                              reinterpret_cast<float*>(input_data.opaque()),
                              fft_type_, is_double_precision_, fft_rank,
                              operand_shape_flat.data(), fft_length_.data());
  } else {
    __xla_cpu_runtime_DuccSingleThreadedFft(
        nullptr, reinterpret_cast<float*>(output_data.opaque()),
        reinterpret_cast<float*>(input_data.opaque()), fft_type_,
        is_double_precision_, fft_rank, operand_shape_flat.data(),
        fft_length_.data());
  }
  return OkExecuteEvent();
}

Thunk::BufferUses FftThunk::buffer_uses() const {
  return {{input_buffer_, BufferUse::kRead},
          {output_buffer_, BufferUse::kWrite}};
}

}  // namespace xla::cpu

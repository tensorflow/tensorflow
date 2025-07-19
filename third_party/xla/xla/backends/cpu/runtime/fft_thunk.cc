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
#include "xla/backends/cpu/runtime/fft_thunk.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ducc/google/fft.h"
#include "Eigen/ThreadPool"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/layout_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

FftThunk::FftThunk(Info thunk_info, bool is_multi_thread_eigen,
                   int32_t fft_type, absl::Span<const int64_t> fft_length,
                   BufferAllocation::Slice input_buffer,
                   const Shape &input_shape,
                   BufferAllocation::Slice output_buffer,
                   const Shape &output_shape)
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
    const Shape &input_shape, BufferAllocation::Slice output_buffer,
    const Shape &output_shape) {
  return absl::WrapUnique(
      new FftThunk(thunk_info, is_multi_thread_eigen, fft_type, fft_length,
                   input_buffer, input_shape, output_buffer, output_shape));
}

static void DuccFft(const Eigen::ThreadPoolDevice *device, void *out,
                    void *operand, int32_t fft_type, int32_t double_precision,
                    int32_t fft_rank, const int64_t *input_shape,
                    const int64_t *fft_length) {
  bool forward = (fft_type == /*FFT*/ 0 || fft_type == /*RFFT*/ 2);
  bool real = (fft_type == /*RFFT*/ 2 || fft_type == /*IRFFT*/ 3);

  using Shape = std::vector<std::size_t>;
  using Stride = std::vector<std::ptrdiff_t>;

  Shape in_shape(fft_rank + 1);
  Stride in_stride(fft_rank + 1);
  Shape out_shape(fft_rank + 1);
  Stride out_stride(fft_rank + 1);
  Shape axes(fft_rank);

  in_shape[fft_rank] = input_shape[fft_rank];
  in_stride[fft_rank] = 1;
  out_shape[fft_rank] = (real && forward) ? fft_length[fft_rank - 1] / 2 + 1
                                          : fft_length[fft_rank - 1];
  out_stride[fft_rank] = 1;
  for (int i = fft_rank; i-- > 1;) {
    in_shape[i] = input_shape[i];
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
    out_shape[i] = fft_length[i - 1];
    out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    axes[i] = i + 1;
  }
  in_shape[0] = input_shape[0];
  in_stride[0] = in_stride[1] * in_shape[1];
  out_shape[0] = in_shape[0];
  out_stride[0] = out_stride[1] * out_shape[1];
  axes[0] = 1;

  // DUCC doesn't handle the case where fft_size[i] < input_size[i],
  // so manually adjust inputs if required.  If doing irfft, the limit
  // of the last axis is actually fft_size[i]/2 + 1.
  const bool is_irfft = real && !forward;
  for (int i = 0; i < fft_rank; ++i) {
    int limit = (is_irfft && (i == (fft_rank - 1))) ? fft_length[i] / 2 + 1
                                                    : fft_length[i];
    if (in_shape[axes[i]] > limit) {
      in_shape[axes[i]] = limit;
    }
  }

  double inv_scale = 1.0;
  for (int i = 0; i < fft_rank; ++i) {
    inv_scale *= out_shape[axes[i]];
  }
  double scale = forward ? 1.0 : 1.0 / inv_scale;

  Eigen::ThreadPoolInterface *thread_pool =
      device ? device->getPool() : nullptr;

  if (!real) {
    if (double_precision) {
      ducc0::google::c2c(static_cast<const std::complex<double> *>(operand),
                         in_shape, in_stride,
                         static_cast<std::complex<double> *>(out), out_shape,
                         out_stride, axes, forward, scale, thread_pool);
    } else {
      ducc0::google::c2c(
          static_cast<const std::complex<float> *>(operand), in_shape,
          in_stride, static_cast<std::complex<float> *>(out), out_shape,
          out_stride, axes, forward, static_cast<float>(scale), thread_pool);
    }
  } else if (forward) {
    if (double_precision) {
      ducc0::google::r2c(static_cast<double *>(operand), in_shape, in_stride,
                         static_cast<std::complex<double> *>(out), out_shape,
                         out_stride, axes, forward, scale, thread_pool);
    } else {
      ducc0::google::r2c(static_cast<float *>(operand), in_shape, in_stride,
                         static_cast<std::complex<float> *>(out), out_shape,
                         out_stride, axes, forward, static_cast<float>(scale),
                         thread_pool);
    }
  } else {
    if (double_precision) {
      ducc0::google::c2r(static_cast<const std::complex<double> *>(operand),
                         in_shape, in_stride, static_cast<double *>(out),
                         out_shape, out_stride, axes, forward, scale,
                         thread_pool);
    } else {
      ducc0::google::c2r(static_cast<const std::complex<float> *>(operand),
                         in_shape, in_stride, static_cast<float *>(out),
                         out_shape, out_stride, axes, forward,
                         static_cast<float>(scale), thread_pool);
    }
  }
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> FftThunk::Execute(
    const ExecuteParams &params) {
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
  int64_t input_batch_length = output_shape_.dimensions().size() - fft_rank;
  for (int i = 0; i < input_batch_length; i++) {
    input_batch *= input_shape_.dimensions(i);
  }
  operand_shape_flat[0] = input_batch;
  for (int i = 0; i < fft_rank; ++i) {
    operand_shape_flat[i + 1] = input_shape_.dimensions(i + input_batch_length);
  }

  DuccFft(is_multi_thread_eigen_ ? params.intra_op_threadpool : nullptr,
          output_data.opaque(), input_data.opaque(), fft_type_,
          is_double_precision_, fft_rank, operand_shape_flat.data(),
          fft_length_.data());

  return OkExecuteEvent();
}

Thunk::BufferUses FftThunk::buffer_uses() const {
  return {{input_buffer_, BufferUse::kRead},
          {output_buffer_, BufferUse::kWrite}};
}

}  // namespace xla::cpu

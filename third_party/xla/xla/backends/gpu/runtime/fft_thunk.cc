/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/fft_thunk.h"

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

se::fft::Type FftTypeToSeType(FftType type, bool double_precision) {
  switch (type) {
    case FftType::FFT:
      return double_precision ? se::fft::Type::kZ2ZForward
                              : se::fft::Type::kC2CForward;
    case FftType::IFFT:
      return double_precision ? se::fft::Type::kZ2ZInverse
                              : se::fft::Type::kC2CInverse;
    case FftType::IRFFT:
      return double_precision ? se::fft::Type::kZ2D : se::fft::Type::kC2R;
    case FftType::RFFT:
      return double_precision ? se::fft::Type::kD2Z : se::fft::Type::kR2C;
    default:
      LOG(FATAL) << "unsupported fft type";
  }
}

std::string FftTypeToString(se::fft::Type type) {
  switch (type) {
    case se::fft::Type::kC2CForward:
    case se::fft::Type::kZ2ZForward:
      return "FFT";
    case se::fft::Type::kC2CInverse:
    case se::fft::Type::kZ2ZInverse:
      return "IFFT";
    case se::fft::Type::kC2R:
    case se::fft::Type::kZ2D:
      return "IRFFT";
    case se::fft::Type::kR2C:
    case se::fft::Type::kD2Z:
      return "RFFT";
    default:
      LOG(FATAL) << "unknown fft type";
  }
}

absl::StatusOr<stream_executor::blas::BlasSupport*> GetBlas(
    se::Stream* stream) {
  auto blas = stream->parent()->AsBlas();
  if (blas == nullptr) {
    return absl::InternalError("Unable to get Blas support");
  }
  return blas;
}

absl::StatusOr<stream_executor::fft::FftSupport*> GetFft(se::Stream* stream) {
  auto fft = stream->parent()->AsFft();
  if (fft == nullptr) {
    return absl::InternalError("Unable to get fft support");
  }
  return fft;
}
}  // namespace

FftThunk::FftThunk(ThunkInfo thunk_info, FftType fft_type,
                   absl::Span<const int64_t> fft_length,
                   const BufferAllocation::Slice& input_buffer,
                   const BufferAllocation::Slice& output_buffer,
                   const Shape& input_shape, const Shape& output_shape)
    : Thunk(Kind::kFft, thunk_info),
      fft_type_(
          FftTypeToSeType(fft_type, input_shape.element_type() == F64 ||
                                        input_shape.element_type() == C128)),
      fft_length_(fft_length.begin(), fft_length.end()),
      input_buffer_(input_buffer),
      output_buffer_(output_buffer),
      input_shape_(input_shape),
      output_shape_(output_shape) {}

absl::Status FftThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;

  return RunFft(
      buffer_allocations.GetDeviceAddress(input_buffer_), input_shape_,
      buffer_allocations.GetDeviceAddress(output_buffer_), output_shape_,
      fft_type_, fft_length_, buffer_allocations.device_ordinal(),
      &fft_plan_cache_, params.stream, buffer_allocations.memory_allocator());
}

absl::Status RunFft(se::DeviceMemoryBase input, const Shape& input_shape,
                    se::DeviceMemoryBase output, const Shape& output_shape,
                    se::fft::Type fft_type, absl::Span<const int64_t> fft_len,
                    int device_ordinal, FftPlanCache* fft_plan_cache,
                    se::Stream* stream,
                    se::DeviceMemoryAllocator* memory_allocator) {
  VLOG(3) << "FFT type: " << FftTypeToString(fft_type);
  VLOG(3) << "Input shape: " << ShapeUtil::HumanStringWithLayout(input_shape);
  VLOG(3) << "Output shape: " << ShapeUtil::HumanStringWithLayout(output_shape);

  se::OwningScratchAllocator<2> scratch_allocator(device_ordinal,
                                                  memory_allocator);

  // Get the Fft plan for the given device ordinal.
  FftPlan* fft_plan_ptr = fft_plan_cache->GetOrCreate(device_ordinal);

  // CuFFT thread-safety requires that separate host threads not share plans;
  // protect each plan with a mutex.
  absl::MutexLock lock(&fft_plan_ptr->mu);
  std::unique_ptr<se::fft::Plan>& fft_plan = fft_plan_ptr->plan;
  TF_ASSIGN_OR_RETURN(auto fft, GetFft(stream));
  if (fft_plan == nullptr) {
    const int64_t fft_rank = fft_len.size();
    CHECK_LE(fft_rank, 3);
    int batch_size = 1;
    for (int i = 0; i < input_shape.rank() - fft_rank; ++i) {
      batch_size *= input_shape.dimensions(i);
    }
    uint64_t fft_length[3];
    uint64_t input_embed[3];
    const uint64_t input_stride = 1;
    uint64_t input_distance = 1;
    uint64_t output_embed[3];
    const uint64_t output_stride = 1;
    uint64_t output_distance = 1;

    for (int i = 0; i < fft_rank; ++i) {
      auto dim_offset = input_shape.rank() - fft_rank + i;
      fft_length[i] = static_cast<uint64_t>(fft_len[i]);
      input_embed[i] = input_shape.dimensions(dim_offset);
      input_distance *= input_shape.dimensions(dim_offset);
      output_embed[i] = output_shape.dimensions(dim_offset);
      output_distance *= output_shape.dimensions(dim_offset);
    }

    constexpr bool kInPlaceFft = false;
    fft_plan = fft->CreateBatchedPlanWithScratchAllocator(
        stream, fft_rank, fft_length, input_embed, input_stride, input_distance,
        output_embed, output_stride, output_distance, fft_type, kInPlaceFft,
        batch_size, &scratch_allocator);
    TF_RET_CHECK(fft_plan != nullptr)
        << "Failed to create cuFFT batched plan with scratch allocator";
    fft_plan_ptr->scale_factor = output_distance;
  } else {
    fft->UpdatePlanWithScratchAllocator(stream, fft_plan.get(),
                                        &scratch_allocator);
  }

  uint64_t scale_factor = fft_plan_ptr->scale_factor;

  bool launch_ok;
  switch (fft_type) {
    case se::fft::Type::kC2CForward: {
      se::DeviceMemory<complex64> input_data(input);
      se::DeviceMemory<complex64> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      break;
    }
    case se::fft::Type::kZ2ZForward: {
      se::DeviceMemory<complex128> input_data(input);
      se::DeviceMemory<complex128> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      break;
    }
    case se::fft::Type::kC2CInverse: {
      se::DeviceMemory<complex64> input_data(input);
      se::DeviceMemory<complex64> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      if (launch_ok) {
        TF_ASSIGN_OR_RETURN(auto blas, GetBlas(stream));
        launch_ok =
            blas->DoBlasScal(stream, ShapeUtil::ElementsIn(output_shape),
                             complex64(1.0f / scale_factor), &output_data, 1);
      }
      break;
    }
    case se::fft::Type::kZ2ZInverse: {
      se::DeviceMemory<complex128> input_data(input);
      se::DeviceMemory<complex128> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      if (launch_ok) {
        TF_ASSIGN_OR_RETURN(auto blas, GetBlas(stream));
        launch_ok =
            blas->DoBlasScal(stream, ShapeUtil::ElementsIn(output_shape),
                             complex128(1.0 / scale_factor), &output_data, 1);
      }
      break;
    }
    case se::fft::Type::kR2C: {
      se::DeviceMemory<float> input_data(input);
      se::DeviceMemory<complex64> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      break;
    }
    case se::fft::Type::kD2Z: {
      se::DeviceMemory<double> input_data(input);
      se::DeviceMemory<complex128> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      break;
    }
    case se::fft::Type::kC2R: {
      se::DeviceMemory<complex64> input_data(input);
      se::DeviceMemory<float> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      if (launch_ok) {
        TF_ASSIGN_OR_RETURN(auto blas, GetBlas(stream));
        launch_ok =
            blas->DoBlasScal(stream, ShapeUtil::ElementsIn(output_shape),
                             1.0f / scale_factor, &output_data, 1);
      }
      break;
    }
    case se::fft::Type::kZ2D: {
      se::DeviceMemory<complex128> input_data(input);
      se::DeviceMemory<double> output_data(output);
      launch_ok = fft->DoFft(stream, fft_plan.get(), input_data, &output_data);
      if (launch_ok) {
        TF_ASSIGN_OR_RETURN(auto blas, GetBlas(stream));
        launch_ok =
            blas->DoBlasScal(stream, ShapeUtil::ElementsIn(output_shape),
                             1.0 / scale_factor, &output_data, 1);
      }
      break;
    }
    default:
      LOG(FATAL) << "unsupported fft type";
  }
  if (launch_ok) {
    return absl::OkStatus();
  }
  return Internal("Unable to launch fft with type %s",
                  FftTypeToString(fft_type));
}

}  // namespace gpu
}  // namespace xla

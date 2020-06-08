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

#include "tensorflow/compiler/xla/service/gpu/fft_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

FftScratchAllocator::FftScratchAllocator(
    int device_ordinal, se::DeviceMemoryAllocator* memory_allocator)
    : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

int64 FftScratchAllocator::GetMemoryLimitInBytes() {
  constexpr int64 kFftScratchSize = 1LL << 32;  // 4GB by default.
  return kFftScratchSize;
}

StatusOr<se::DeviceMemory<uint8>> FftScratchAllocator::AllocateBytes(
    int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8>(buffer_addr);
}

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

string FftTypeToString(se::fft::Type type) {
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

}  // namespace

FftThunk::FftThunk(FftType fft_type, absl::Span<const int64> fft_length,
                   const BufferAllocation::Slice& input_buffer,
                   const BufferAllocation::Slice& output_buffer,
                   const Shape& input_shape, const Shape& output_shape,
                   const HloInstruction* hlo)
    : Thunk(Kind::kFft, hlo),
      fft_type_(
          FftTypeToSeType(fft_type, input_shape.element_type() == F64 ||
                                        input_shape.element_type() == C128)),
      fft_length_(fft_length.begin(), fft_length.end()),
      scale_factor_(1.0f),
      input_buffer_(input_buffer),
      output_buffer_(output_buffer),
      input_shape_(input_shape),
      output_shape_(output_shape) {}

Status FftThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  VLOG(3) << "FFT type: " << FftTypeToString(fft_type_);
  VLOG(3) << "Input shape: " << ShapeUtil::HumanStringWithLayout(input_shape_);
  VLOG(3) << "Output shape: "
          << ShapeUtil::HumanStringWithLayout(output_shape_);

  FftScratchAllocator scratch_allocator(buffer_allocations.device_ordinal(),
                                        buffer_allocations.memory_allocator());

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  if (fft_plan_ == nullptr) {
    const int64 fft_rank = fft_length_.size();
    CHECK_LE(fft_rank, 3);
    int batch_size = 1;
    for (int i = 0; i < input_shape_.dimensions_size() - fft_rank; ++i) {
      batch_size *= input_shape_.dimensions(i);
    }
    uint64 fft_length[3];
    uint64 input_embed[3];
    const uint64 input_stride = 1;
    uint64 input_distance = 1;
    uint64 output_embed[3];
    const uint64 output_stride = 1;
    uint64 output_distance = 1;

    for (int i = 0; i < fft_rank; ++i) {
      auto dim_offset = input_shape_.dimensions_size() - fft_rank + i;
      fft_length[i] = static_cast<uint64>(fft_length_[i]);
      input_embed[i] = input_shape_.dimensions(dim_offset);
      input_distance *= input_shape_.dimensions(dim_offset);
      output_embed[i] = output_shape_.dimensions(dim_offset);
      output_distance *= output_shape_.dimensions(dim_offset);
    }

    constexpr bool kInPlaceFft = false;
    fft_plan_ = stream.parent()->AsFft()->CreateBatchedPlanWithScratchAllocator(
        &stream, fft_rank, fft_length, input_embed, input_stride,
        input_distance, output_embed, output_stride, output_distance, fft_type_,
        kInPlaceFft, batch_size, &scratch_allocator);
    scale_factor_ = 1.0f / output_distance;
  } else {
    stream.parent()->AsFft()->UpdatePlanWithScratchAllocator(
        &stream, fft_plan_.get(), &scratch_allocator);
  }

  bool launch_ok;
  switch (fft_type_) {
    case se::fft::Type::kC2CForward: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kZ2ZForward: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kC2CInverse: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok = stream
                        .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                                      complex64(scale_factor_), &output_data, 1)
                        .ok();
      }
      break;
    }
    case se::fft::Type::kZ2ZInverse: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok =
            stream
                .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                              complex128(scale_factor_), &output_data, 1)
                .ok();
      }
      break;
    }
    case se::fft::Type::kR2C: {
      se::DeviceMemory<float> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex64> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kD2Z: {
      se::DeviceMemory<double> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<complex128> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      break;
    }
    case se::fft::Type::kC2R: {
      se::DeviceMemory<complex64> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<float> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok = stream
                        .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                                      scale_factor_, &output_data, 1)
                        .ok();
      }
      break;
    }
    case se::fft::Type::kZ2D: {
      se::DeviceMemory<complex128> input_data(
          buffer_allocations.GetDeviceAddress(input_buffer_));
      se::DeviceMemory<double> output_data(
          buffer_allocations.GetDeviceAddress(output_buffer_));
      launch_ok =
          stream.ThenFft(fft_plan_.get(), input_data, &output_data).ok();
      if (launch_ok) {
        launch_ok = stream
                        .ThenBlasScal(ShapeUtil::ElementsIn(output_shape_),
                                      scale_factor_, &output_data, 1)
                        .ok();
      }
      break;
    }
    default:
      LOG(FATAL) << "unsupported fft type";
  }
  if (launch_ok) {
    return Status::OK();
  }
  return InternalError("Unable to launch fft for thunk %p with type %s", this,
                       FftTypeToString(fft_type_));
}

}  // namespace gpu
}  // namespace xla

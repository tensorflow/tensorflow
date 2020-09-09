/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// CUDA-specific support for FFT functionality -- this wraps the cuFFT library
// capabilities, and is only included into CUDA implementation code -- it will
// not introduce cuda headers into other code.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_

#include "third_party/gpus/cuda/include/cufft.h"
#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace stream_executor {

class Stream;

namespace gpu {

class GpuExecutor;

// Opaque and unique indentifier for the cuFFT plugin.
extern const PluginId kCuFftPlugin;

// CUDAFftPlan uses deferred initialization. Only a single call of
// Initialize() is allowed to properly create cufft plan and set member
// variable is_initialized_ to true. Newly added interface that uses member
// variables should first check is_initialized_ to make sure that the values of
// member variables are valid.
class CUDAFftPlan : public fft::Plan {
 public:
  CUDAFftPlan()
      : parent_(nullptr),
        plan_(-1),
        fft_type_(fft::Type::kInvalid),
        scratch_(nullptr),
        scratch_size_bytes_(0),
        is_initialized_(false),
        scratch_allocator_(nullptr) {}
  ~CUDAFftPlan() override;

  // Get FFT direction in cuFFT based on FFT type.
  int GetFftDirection() const;
  cufftHandle GetPlan() const {
    if (IsInitialized()) {
      return plan_;
    } else {
      LOG(FATAL) << "Try to get cufftHandle value before initialization.";
    }
  }

  // Initialize function for batched plan
  port::Status Initialize(GpuExecutor* parent, Stream* stream, int rank,
                          uint64* elem_count, uint64* input_embed,
                          uint64 input_stride, uint64 input_distance,
                          uint64* output_embed, uint64 output_stride,
                          uint64 output_distance, fft::Type type,
                          int batch_count, ScratchAllocator* scratch_allocator);

  // Initialize function for 1d,2d, and 3d plan
  port::Status Initialize(GpuExecutor* parent, Stream* stream, int rank,
                          uint64* elem_count, fft::Type type,
                          ScratchAllocator* scratch_allocator);

  port::Status UpdateScratchAllocator(Stream *stream,
                                      ScratchAllocator *scratch_allocator);

  ScratchAllocator* GetScratchAllocator() const { return scratch_allocator_; }

 protected:
  bool IsInitialized() const { return is_initialized_; }

 private:
  GpuExecutor* parent_;
  cufftHandle plan_;
  fft::Type fft_type_;
  DeviceMemory<uint8> scratch_;
  size_t scratch_size_bytes_;
  bool is_initialized_;
  ScratchAllocator* scratch_allocator_;
};

// FFT support for CUDA platform via cuFFT library.
//
// This satisfies the platform-agnostic FftSupport interface.
//
// Note that the cuFFT handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent GpuExecutor is tied
// to. This simply happens as an artifact of creating the cuFFT handle when a
// CUDA context is active.
//
// Thread-safe. The CUDA context associated with all operations is the CUDA
// context of parent_, so all context is explicit.
class CUDAFft : public fft::FftSupport {
 public:
  explicit CUDAFft(GpuExecutor* parent) : parent_(parent) {}
  ~CUDAFft() override {}

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  GpuExecutor* parent_;

  // Two helper functions that execute dynload::cufftExec?2?.

  // This is for complex to complex FFT, when the direction is required.
  template <typename FuncT, typename InputT, typename OutputT>
  bool DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                  FuncT cufft_exec,
                                  const DeviceMemory<InputT> &input,
                                  DeviceMemory<OutputT> *output);

  // This is for complex to real or real to complex FFT, when the direction
  // is implied.
  template <typename FuncT, typename InputT, typename OutputT>
  bool DoFftInternal(Stream *stream, fft::Plan *plan, FuncT cufft_exec,
                     const DeviceMemory<InputT> &input,
                     DeviceMemory<OutputT> *output);

  SE_DISALLOW_COPY_AND_ASSIGN(CUDAFft);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_

/* Copyright 2015 The OpenXLA Authors.

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

// ROCM-specific support for FFT functionality -- this wraps the rocFFT library
// capabilities, and is only included into ROCM implementation code -- it will
// not introduce rocm headers into other code.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_FFT_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_FFT_H_

#include <cstdint>

#include "rocm/rocm_config.h"  // IWYU pragma: keep - needed for TF_ROCM_VERSION

#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/hipfft/hipfft.h"
#else
#include "rocm/include/hipfft.h"
#endif

#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor {

namespace gpu {

// ROCMFftPlan uses deferred initialization. Only a single call of
// Initialize() is allowed to properly create hipfft plan and set member
// variable is_initialized_ to true. Newly added interface that uses member
// variables should first check is_initialized_ to make sure that the values of
// member variables are valid.
class ROCMFftPlan : public fft::Plan {
 public:
  ROCMFftPlan()
      : parent_(nullptr),
        plan_(),
        fft_type_(fft::Type::kInvalid),
        scratch_(nullptr),
        scratch_size_bytes_(0),
        is_initialized_(false) {}
  ~ROCMFftPlan() override;

  // Get FFT direction in hipFFT based on FFT type.
  int GetFftDirection() const;
  hipfftHandle GetPlan() const {
    if (IsInitialized()) {
      return plan_;
    } else {
      LOG(FATAL) << "Try to get hipfftHandle value before initialization.";
    }
  }

  // Initialize function for batched plan
  absl::Status Initialize(StreamExecutor *parent, Stream *stream, int rank,
                          uint64_t *elem_count, uint64_t *input_embed,
                          uint64_t input_stride, uint64_t input_distance,
                          uint64_t *output_embed, uint64_t output_stride,
                          uint64_t output_distance, fft::Type type,
                          int batch_count, ScratchAllocator *scratch_allocator);

  // Initialize function for 1d,2d, and 3d plan
  absl::Status Initialize(StreamExecutor *parent, Stream *stream, int rank,
                          uint64_t *elem_count, fft::Type type,
                          ScratchAllocator *scratch_allocator);

  absl::Status UpdateScratchAllocator(Stream *stream,
                                      ScratchAllocator *scratch_allocator);

  ScratchAllocator *GetScratchAllocator() const { return scratch_allocator_; }

 protected:
  bool IsInitialized() const { return is_initialized_; }
  ScratchAllocator *scratch_allocator_;

 private:
  StreamExecutor *parent_;
  hipfftHandle plan_;
  fft::Type fft_type_;
  DeviceMemory<uint8_t> scratch_;
  size_t scratch_size_bytes_;
  bool is_initialized_;
};

// FFT support for ROCM platform via rocFFT library.
//
// This satisfies the platform-agnostic FftSupport interface.
//
// Note that the hipFFT handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent StreamExecutor is tied
// to. This simply happens as an artifact of creating the hipFFT handle when a
// ROCM context is active.
//
// Thread-safe. The ROCM context associated with all operations is the ROCM
// context of parent_, so all context is explicit.
class ROCMFft : public fft::FftSupport {
 public:
  explicit ROCMFft(StreamExecutor *parent) : parent_(parent) {}
  ~ROCMFft() override {}

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  StreamExecutor *parent_;

  // Two helper functions that execute dynload::hipfftExec?2?.

  // This is for complex to complex FFT, when the direction is required.
  template <typename FuncT, typename InputT, typename OutputT>
  bool DoFftWithDirectionInternal(Stream *stream, fft::Plan *plan,
                                  FuncT hipfft_exec,
                                  const DeviceMemory<InputT> &input,
                                  DeviceMemory<OutputT> *output);

  // This is for complex to real or real to complex FFT, when the direction
  // is implied.
  template <typename FuncT, typename InputT, typename OutputT>
  bool DoFftInternal(Stream *stream, fft::Plan *plan, FuncT hipfft_exec,
                     const DeviceMemory<InputT> &input,
                     DeviceMemory<OutputT> *output);

  ROCMFft(const ROCMFft &) = delete;
  void operator=(const ROCMFft &) = delete;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_FFT_H_

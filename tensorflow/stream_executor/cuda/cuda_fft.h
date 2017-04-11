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

#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "cuda/include/cufft.h"

namespace perftools {
namespace gputools {

class Stream;

namespace cuda {

class CUDAExecutor;

// Opaque and unique indentifier for the cuFFT plugin.
extern const PluginId kCuFftPlugin;

class CUDAFftPlan : public fft::Plan {
 public:
  // Constructor creating 1d FFT plan.
  CUDAFftPlan(CUDAExecutor *parent, uint64 num_x, fft::Type type);
  // Constructor creating 2d FFT plan.
  CUDAFftPlan(CUDAExecutor *parent, uint64 num_x, uint64 num_y, fft::Type type);
  // Constructor creating 3d FFT plan.
  CUDAFftPlan(CUDAExecutor *parent, uint64 num_x, uint64 num_y, uint64 num_z,
              fft::Type type);
  // Constructor creating batched FFT plan.
  CUDAFftPlan(CUDAExecutor *parent, int rank, uint64 *elem_count,
              uint64 *input_embed, uint64 input_stride, uint64 input_distance,
              uint64 *output_embed, uint64 output_stride,
              uint64 output_distance, fft::Type type, int batch_count);
  ~CUDAFftPlan() override;

  // Get FFT direction in cuFFT based on FFT type.
  int GetFftDirection() const;
  cufftHandle GetPlan() const { return plan_; }

 private:
  CUDAExecutor *parent_;
  cufftHandle plan_;
  fft::Type fft_type_;
};

// FFT support for CUDA platform via cuFFT library.
//
// This satisfies the platform-agnostic FftSupport interface.
//
// Note that the cuFFT handle that this encapsulates is implicitly tied to the
// context (and, as a result, the device) that the parent CUDAExecutor is tied
// to. This simply happens as an artifact of creating the cuFFT handle when a
// CUDA context is active.
//
// Thread-safe. The CUDA context associated with all operations is the CUDA
// context of parent_, so all context is explicit.
class CUDAFft : public fft::FftSupport {
 public:
  explicit CUDAFft(CUDAExecutor *parent) : parent_(parent) {}
  ~CUDAFft() override {}

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  CUDAExecutor *parent_;

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

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_FFT_H_

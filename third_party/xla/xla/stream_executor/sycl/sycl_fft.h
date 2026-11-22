/* Copyright 2026 The OpenXLA Authors.

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

// SYCL-specific support for FFT functionality -- this wraps the oneMKL DFT
// library capabilities. The oneMKL headers are only included from the
// implementation file so that they don't leak into other code.

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_FFT_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_FFT_H_

#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace sycl {

// FFT support for the SYCL platform via the oneMKL DFT domain.
//
// This satisfies the platform-agnostic FftSupport interface. The oneMKL
// descriptors that back the plans are committed to the SYCL queue of the
// stream the FFT runs on; a plan is re-committed transparently if it is
// later used with a different stream.
class SyclFft : public fft::FftSupport {
 public:
  explicit SyclFft(StreamExecutor* parent) : parent_(parent) {}
  ~SyclFft() override = default;

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  // Shared implementation of the forward DoFft() overloads: transforms the
  // input buffer into the output buffer using the descriptor held by the plan.
  //
  // The element types select that descriptor: float or double picks the
  // precision, and whether the input is real or complex picks the domain.
  template <typename InputT, typename OutputT>
  bool DoFftInternal(Stream* stream, fft::Plan* plan,
                     const DeviceAddress<InputT>& input,
                     DeviceAddress<OutputT>* output);

  StreamExecutor* parent_;

  SyclFft(const SyclFft&) = delete;
  void operator=(const SyclFft&) = delete;
};

// Registers the oneMKL DFT factory with the plugin registry.
void initialize_onemkl_fft();

}  // namespace sycl
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_FFT_H_

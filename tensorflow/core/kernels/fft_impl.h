/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_FFT_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_FFT_IMPL_H_

// Generic interface for N-D FFT implementation.
// Required to isolate the DUCC FFT implementation on CPU, so we can limit
// required build flags to a single module.

#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace internal {

template <typename Device>
absl::Status FftImpl(const Device& device, const Tensor& in, Tensor* out,
                     const uint64_t* fft_shape, const std::vector<size_t>& axes,
                     bool forward);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FFT_IMPL_H_

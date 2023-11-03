/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/cpu/runtime_single_threaded_fft.h"

#include <cstdint>

#include "absl/base/attributes.h"
#include "xla/service/cpu/runtime_fft.h"

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_DuccSingleThreadedFft(
    const void* /*run_options_ptr*/, void* out, void* operand, int32_t fft_type,
    int32_t double_precision, int32_t fft_rank, const int64_t* input_shape,
    const int64_t* fft_length) {
  return __xla_cpu_runtime_DuccFft(
      /*run_options_ptr=*/nullptr, out, operand, fft_type, double_precision,
      fft_rank, input_shape, fft_length);
}

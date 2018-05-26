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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_FFT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_FFT_H_

#include "tensorflow/core/platform/types.h"

extern "C" {

extern void __xla_cpu_runtime_EigenSingleThreadedFft(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, void* out,
    void* operand, tensorflow::int32 fft_type, tensorflow::int32 fft_rank,
    tensorflow::int64 input_batch, tensorflow::int64 fft_length0,
    tensorflow::int64 fft_length1, tensorflow::int64 fft_length2);

}  // extern "C"

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_SINGLE_THREADED_FFT_H_

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_
#define TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_

#include <cstdint>
#include <string>

struct TfLiteTensor;

namespace tflite {
// Records an event of `num_bytes` of memory allocated for `tensor`.
void OnTfLiteTensorAlloc(size_t num_bytes, TfLiteTensor* tensor);
// Records an event of memory deallocated for `tensor`.
void OnTfLiteTensorDealloc(TfLiteTensor* tensor);
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_

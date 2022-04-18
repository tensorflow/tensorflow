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

// Adds memory trace information for TensorFlow profiler.
// `is_allocating`: Whether memory is being allocated or deallocated.
// `allocation_bytes`: The number of bytes being allocated or deallocated.
// `tensor_id`: A unique ID for the tensor being allocated or deallocated.
//              Usually the memory address should be used.
// `name`: The name of the tensor being allocated or deallocated.
// `dims`: The dimension of the tensor in a string form.
void AddTraceMe(bool is_allocating, int64_t allocation_bytes, int64_t tensor_id,
                const std::string& name, const std::string& dims);

#endif  // TENSORFLOW_LITE_TENSORFLOW_PROFILER_LOGGER_H_

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_SET_BIG_CORE_AFFINITY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_SET_BIG_CORE_AFFINITY_H_

#include <sys/types.h>

#include <cstdint>

namespace tflite {
namespace acceleration {

// Set the CPU affinity of current process to the system's big cores.
// Returns 0 in case of success, otherwise returns the value of errno after
// the call to sched_setaffinity.
int32_t SetBigCoresAffinity();

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_SET_BIG_CORE_AFFINITY_H_

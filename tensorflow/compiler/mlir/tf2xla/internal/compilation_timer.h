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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_COMPILATION_TIMER_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_COMPILATION_TIMER_H_

#include <chrono>  // NOLINT(build/c++11)

#include "tensorflow/core/platform/profile_utils/cpu_utils.h"

// Time the execution of kernels (in CPU cycles). Meant to be used as RAII.
struct CompilationTimer {
  uint64_t start_cycles =
      tensorflow::profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64_t ElapsedCycles() {
    return tensorflow::profile_utils::CpuUtils::GetCurrentClockCycle() -
           start_cycles;
  }

  int64_t ElapsedCyclesInMilliseconds() {
    std::chrono::duration<double> duration =
        tensorflow::profile_utils::CpuUtils::ConvertClockCycleToTime(
            ElapsedCycles());

    return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
        .count();
  }
};

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_COMPILATION_TIMER_H_

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_PROFILE_UTILS_I_CPU_UTILS_HELPER_H_
#define TENSORFLOW_TSL_PLATFORM_PROFILE_UTILS_I_CPU_UTILS_HELPER_H_

#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/types.h"

namespace tsl {
namespace profile_utils {

// ICpuUtilsHelper is an interface class for cpu_utils which proxies
// the difference of profiling functions of different platforms.
// Overridden functions must be thread safe.
class ICpuUtilsHelper {
 public:
  ICpuUtilsHelper() = default;
  virtual ~ICpuUtilsHelper() = default;
  // Reset clock cycle.
  // Resetting clock cycle is recommended to prevent
  // clock cycle counters from overflowing on some platforms.
  virtual void ResetClockCycle() = 0;
  // Return current clock cycle.
  virtual uint64 GetCurrentClockCycle() = 0;
  // Enable/Disable clock cycle profile
  // You can enable / disable profile if it's supported by the platform
  virtual void EnableClockCycleProfiling() = 0;
  virtual void DisableClockCycleProfiling() = 0;
  // Return cpu frequency.
  // CAVEAT: as this method may read file and/or call system calls,
  // this call is supposed to be slow.
  virtual int64_t CalculateCpuFrequency() = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ICpuUtilsHelper);
};

}  // namespace profile_utils
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_PROFILE_UTILS_I_CPU_UTILS_HELPER_H_

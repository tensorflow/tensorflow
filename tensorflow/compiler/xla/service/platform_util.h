/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLATFORM_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLATFORM_UTIL_H_

#include <vector>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Utilities for querying platforms and devices used by XLA.
class PlatformUtil {
 public:
  // Returns the platforms present on the system and supported by XLA.
  //
  // Note that, even if a platform is present with zero devices, if we *do* have
  // compilation support for it, it will be returned in this sequence.
  static StatusOr<std::vector<perftools::gputools::Platform*>>
  GetSupportedPlatforms();

  // Convenience function which returns the default supported platform. If
  // exactly one supported platform is present, then this platform is the
  // default platform. If exactly two supported platforms are present and one
  // platform is CPU (host) then the non-CPU platform is default. This logic is
  // used because the XLA service always links in the CPU backend to run
  // ComputeConstant, so if exactly one other platform is linked in, we assume
  // the intent is to execute on that non-CPU platform. If none of these
  // conditions are met the function returns an error.
  static StatusOr<perftools::gputools::Platform*> GetDefaultPlatform();

  // Returns a vector of StreamExecutors for the given platform. The vector is
  // indexed by device ordinal (device numbering used by StreamExecutor). If an
  // element is nullptr, then the device is present by not supported by XLA.
  //
  // If the platform has no visible devices, a not-found error is returned.
  static StatusOr<std::vector<perftools::gputools::StreamExecutor*>>
  GetStreamExecutors(perftools::gputools::Platform* platform);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PlatformUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLATFORM_UTIL_H_

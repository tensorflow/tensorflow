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

#include <set>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Utilities for querying platforms and devices used by XLA.
class PlatformUtil {
 public:
  // Returns the platforms present on the system and supported by XLA.
  //
  // Note that, even if a platform is present with zero devices, if we *do* have
  // compilation support for it, it will be returned in this sequence.
  static StatusOr<std::vector<se::Platform*>> GetSupportedPlatforms();

  // Convenience function which returns the default supported platform for
  // tests. If exactly one supported platform is present, then this platform is
  // the default platform. If exactly two platforms are present and one of them
  // is the interpreter platform, then the other platform is the default
  // platform. Otherwise returns an error.
  static StatusOr<se::Platform*> GetDefaultPlatform();

  // Convenience function which returns the sole supported platform. If
  // exactly one supported platform is present, then this platform is the
  // default platform. Otherwise returns an error.
  static StatusOr<se::Platform*> GetSolePlatform();

  // Returns the platform according to the given name. Returns error if there is
  // no such platform.
  static StatusOr<se::Platform*> GetPlatform(const string& platform_name);

  // Returns exactly one platform that does not have given name. Returns error
  // if there is no such platform, or there are multiple such platforms.
  static StatusOr<se::Platform*> GetPlatformExceptFor(
      const string& platform_name);

  // Returns a vector of StreamExecutors for the given platform. The vector is
  // indexed by device ordinal (device numbering used by StreamExecutor). If an
  // element is nullptr, then the device is present by not supported by XLA.
  // Optional parameter, allowed_devices controls which of the devices on the
  // platform will have StreamExecutors constructed for. 
  //
  // If the platform has no visible devices, a not-found error is returned.
  static StatusOr<std::vector<se::StreamExecutor*>> GetStreamExecutors(
      se::Platform* platform,
      const absl::optional<std::set<int>>& allowed_devices = absl::nullopt);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PlatformUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLATFORM_UTIL_H_

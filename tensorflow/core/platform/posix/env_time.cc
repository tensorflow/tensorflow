/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <sys/time.h>
#include <time.h>

#include "tensorflow/core/platform/env_time.h"

namespace tensorflow {

namespace {

class PosixEnvTime : public EnvTime {
 public:
  PosixEnvTime() {}

  uint64 NowNanos() const override {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (static_cast<uint64>(ts.tv_sec) * kSecondsToNanos +
            static_cast<uint64>(ts.tv_nsec));
  }
};

}  // namespace

#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
EnvTime* EnvTime::Default() {
  static EnvTime* default_env_time = new PosixEnvTime;
  return default_env_time;
}
#endif

}  // namespace tensorflow

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

#ifndef XLA_TSL_PLATFORM_CLOUD_NOW_SECONDS_ENV_H_
#define XLA_TSL_PLATFORM_CLOUD_NOW_SECONDS_ENV_H_

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/mutex.h"

namespace tsl {

/// This Env wrapper lets us control the NowSeconds() return value.
class NowSecondsEnv : public EnvWrapper {
 public:
  NowSecondsEnv() : EnvWrapper(Env::Default()) {}

  /// The current (fake) timestamp.
  uint64 NowSeconds() const override {
    mutex_lock lock(mu_);
    return now_;
  }

  /// Set the current (fake) timestamp.
  void SetNowSeconds(uint64 now) {
    mutex_lock lock(mu_);
    now_ = now;
  }

  /// Guards access to now_.
  mutable mutex mu_;

  /// The NowSeconds() value that this Env will return.
  uint64 now_ = 1;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_CLOUD_NOW_SECONDS_ENV_H_

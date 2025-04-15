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

#ifndef TENSORFLOW_CORE_UTIL_FAKE_CLOCK_ENV_H_
#define TENSORFLOW_CORE_UTIL_FAKE_CLOCK_ENV_H_

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// An Env implementation with a fake clock for NowMicros().
// The clock doesn't advance on its own. It advances
// via an explicit AdvanceByMicroseconds() method. All other Env virtual methods
// pass through to a wrapped Env.
class FakeClockEnv : public EnvWrapper {
 public:
  explicit FakeClockEnv(Env* wrapped);
  ~FakeClockEnv() override = default;

  // Advance the clock by a certain number of microseconds.
  void AdvanceByMicroseconds(int64_t micros);

  // Returns the current time of FakeClockEnv in microseconds.
  uint64 NowMicros() const override;

 private:
  mutable mutex mu_;
  uint64 current_time_ TF_GUARDED_BY(mu_) = 0;

  FakeClockEnv(const FakeClockEnv&) = delete;
  void operator=(const FakeClockEnv&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_FAKE_CLOCK_ENV_H_

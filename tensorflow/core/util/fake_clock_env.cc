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

#include "tensorflow/core/util/fake_clock_env.h"

#include <string>

namespace tensorflow {

FakeClockEnv::FakeClockEnv(Env* wrapped) : EnvWrapper(wrapped) {}

void FakeClockEnv::AdvanceByMicroseconds(int64_t micros) {
  {
    mutex_lock l(mu_);
    current_time_ += micros;
  }
}

uint64 FakeClockEnv::NowMicros() const {
  {
    mutex_lock l(mu_);
    return current_time_;
  }
}

}  // namespace tensorflow

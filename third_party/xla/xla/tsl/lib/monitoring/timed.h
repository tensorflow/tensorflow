/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_LIB_MONITORING_TIMED_H_
#define XLA_TSL_LIB_MONITORING_TIMED_H_

#include "xla/tsl/platform/env_time.h"

namespace tsl {
namespace monitoring {

// Takes a Sampler, PercentileSample or Gauge cell, and post timing values
// (default in milliseconds) according to its scope lifetime.
template <typename T>
class Timed {
 public:
  explicit Timed(T* cell, double scale = 1e-6)
      : cell_(cell), scale_(scale), start_(EnvTime::NowNanos()) {}

  ~Timed() { cell_->Add(scale_ * (EnvTime::NowNanos() - start_)); }

 private:
  T* cell_ = nullptr;
  double scale_ = 1e-6;
  uint64 start_ = 0;
};

template <typename T>
Timed<T> MakeTimed(T* cell, double scale = 1e-6) {
  return Timed<T>(cell, scale);
}

}  // namespace monitoring
}  // namespace tsl

#endif  // XLA_TSL_LIB_MONITORING_TIMED_H_

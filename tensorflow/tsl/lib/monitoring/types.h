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

#ifndef TENSORFLOW_TSL_LIB_MONITORING_TYPES_H_
#define TENSORFLOW_TSL_LIB_MONITORING_TYPES_H_

#include <cmath>
#include <vector>

#include "tensorflow/tsl/platform/types.h"

namespace tsl {
namespace monitoring {

enum class UnitOfMeasure {
  kNumber,
  kTime,
  kBytes,
};

struct PercentilePoint {
  // In the [0, 100] range.
  double percentile = 0.0;
  double value = 0.0;
};

struct Percentiles {
  UnitOfMeasure unit_of_measure = UnitOfMeasure::kNumber;
  uint64 start_nstime = 0;
  uint64 end_nstime = 0;
  double min_value = NAN;
  double max_value = NAN;
  double mean = NAN;
  double stddev = NAN;
  size_t num_samples = 0;
  size_t total_samples = 0;
  long double accumulator = NAN;
  std::vector<PercentilePoint> points;
};

}  // namespace monitoring
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_MONITORING_TYPES_H_

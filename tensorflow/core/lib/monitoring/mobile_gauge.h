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

// Null implementation of the Gauge metric for mobile platforms.

#ifndef TENSORFLOW_CORE_LIB_MONITORING_MOBILE_GAUGE_H_
#define TENSORFLOW_CORE_LIB_MONITORING_MOBILE_GAUGE_H_

#if !defined(IS_MOBILE_PLATFORM) || !defined(TENSORFLOW_INCLUDED_FROM_GAUGE_H)
// If this header file were included directly, and something else included its
// non-mobile counterpart, there could be an unchecked ODR violation on the
// classes below.
#error do not include mobile_gauge.h directly; use gauge.h instead
#endif  // !defined(IS_MOBILE_PLATFORM) ||
        // !defined(TENSORFLOW_INCLUDED_FROM_GAUGE_H)

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// GaugeCell which has a null implementation.
template <typename T>
class GaugeCell {
 public:
 public:
  GaugeCell() {}
  ~GaugeCell() {}

  void Set(const T& value) {}
  T value() const { return T(); }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GaugeCell);
};

// Gauge which has a null implementation.
template <typename ValueType, int NumLabels>
class Gauge {
 public:
  ~Gauge() {}

  template <typename... MetricDefArgs>
  static Gauge* New(MetricDefArgs&&... metric_def_args) {
    static_assert(std::is_same<ValueType, int64>::value ||
                      std::is_same<ValueType, string>::value ||
                      std::is_same<ValueType, bool>::value,
                  "Gauge only allows bool, int64, and string types.");
    return new Gauge();
  }

  template <typename... Labels>
  GaugeCell<ValueType>* GetCell(const Labels&... labels) {
    return &default_gauge_cell_;
  }

  Status GetStatus() { return Status::OK(); }

 private:
  Gauge() {}

  GaugeCell<ValueType> default_gauge_cell_;

  TF_DISALLOW_COPY_AND_ASSIGN(Gauge);
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_MONITORING_MOBILE_GAUGE_H_

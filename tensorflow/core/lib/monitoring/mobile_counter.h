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

// Null implementation of the Counter metric for mobile platforms.

#ifndef TENSORFLOW_CORE_LIB_MONITORING_MOBILE_COUNTER_H_
#define TENSORFLOW_CORE_LIB_MONITORING_MOBILE_COUNTER_H_

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// CounterCell which has a null implementation.
class CounterCell {
 public:
  CounterCell() {}
  ~CounterCell() {}

  void IncrementBy(int64 step) {}
  int64 value() const { return 0; }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CounterCell);
};

// Counter which has a null implementation.
template <int NumLabels>
class Counter {
 public:
  ~Counter() {}

  template <typename... MetricDefArgs>
  static Counter* New(MetricDefArgs&&... metric_def_args) {
    return new Counter<NumLabels>();
  }

  template <typename... Labels>
  CounterCell* GetCell(const Labels&... labels) {
    return &default_counter_cell_;
  }

 private:
  Counter() {}

  CounterCell default_counter_cell_;

  TF_DISALLOW_COPY_AND_ASSIGN(Counter);
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_MONITORING_MOBILE_COUNTER_H_

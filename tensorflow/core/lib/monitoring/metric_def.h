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

#ifndef TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_
#define TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/monitoring/types.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace monitoring {

// The different metric kinds available.
//
// Gauge indicates that the metric's values are instantaneous measurements of a
// (typically) continuously varying value. Examples: a process's current heap
// size, a queue's current length, the name of the binary used by a process,
// whether a task is complete.
//
// Cumulative indicates that the metric's values represent non-negative changes
// over specified time periods. Example: the number of rpc calls to a service.
enum class MetricKind : int { kGauge = 0, kCumulative };

// The type of the metric values.
enum class ValueType : int {
  kInt64 = 0,
  kHistogram,
  kString,
  kBool,
  kPercentiles
};

// Everything in the internal namespace is implementation details. Do not depend
// on this.
namespace internal {

template <typename Value>
ValueType GetValueType();

template <>
inline ValueType GetValueType<int64_t>() {
  return ValueType::kInt64;
}

template <>
inline ValueType GetValueType<std::function<int64_t()>>() {
  return ValueType::kInt64;
}

template <>
inline ValueType GetValueType<HistogramProto>() {
  return ValueType::kHistogram;
}

template <>
inline ValueType GetValueType<Percentiles>() {
  return ValueType::kPercentiles;
}

template <>
inline ValueType GetValueType<std::string>() {
  return ValueType::kString;
}

template <>
inline ValueType GetValueType<std::function<std::string()>>() {
  return ValueType::kString;
}

template <>
inline ValueType GetValueType<bool>() {
  return ValueType::kBool;
}

template <>
inline ValueType GetValueType<std::function<bool()>>() {
  return ValueType::kBool;
}

}  // namespace internal

// Abstract base class for a metric definition.
//
// Unlike MetricDef, this class is non-templatized and allows storing and
// accessing metric definitions without the full type information.
//
// Everything except the value type of a metric is stored here. Please read
// MetricDef class comments for more details.
class AbstractMetricDef {
 public:
  MetricKind kind() const { return kind_; }

  ValueType value_type() const { return value_type_; }

  StringPiece name() const { return name_; }

  StringPiece description() const { return description_; }

  const std::vector<string>& label_descriptions() const {
    return label_descriptions_;
  }

 private:
  template <MetricKind kind, typename Value, int NumLabels>
  friend class MetricDef;

  AbstractMetricDef(const MetricKind kind, const ValueType value_type,
                    const StringPiece name, const StringPiece description,
                    const std::vector<string>& label_descriptions)
      : kind_(kind),
        value_type_(value_type),
        name_(name),
        description_(description),
        label_descriptions_(std::vector<string>(label_descriptions.begin(),
                                                label_descriptions.end())) {}

  const MetricKind kind_;
  const ValueType value_type_;
  const string name_;
  const string description_;
  const std::vector<string> label_descriptions_;
};

// Metric definition.
//
// A metric is defined by its kind, value-type, name, description and the
// description of its labels.
//
// NOTE: Name, description, and label descriptions should be logically static,
// but do not have to live for the lifetime of the MetricDef.
//
// By "logically static", we mean that they should never contain dynamic
// information, but is static for the lifetime of the MetricDef, and
// in-turn the metric; they do not need to be compile-time constants.
// This allows for e.g. prefixed metrics in a CLIF wrapped environment.
template <MetricKind metric_kind, typename Value, int NumLabels>
class MetricDef : public AbstractMetricDef {
 public:
  template <typename... LabelDesc>
  MetricDef(const StringPiece name, const StringPiece description,
            const LabelDesc&... label_descriptions)
      : AbstractMetricDef(metric_kind, internal::GetValueType<Value>(), name,
                          description, {label_descriptions...}) {
    static_assert(sizeof...(LabelDesc) == NumLabels,
                  "Mismatch between Counter<NumLabels> and number of label "
                  "descriptions.");
  }
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_

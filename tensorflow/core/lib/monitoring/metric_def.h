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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_

#include <array>
#include <vector>

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace monitoring {

// The different metric kinds available.
//
// Gauge indicates that the metric's values are instantaneous measurements of a
// (typically) continuously varying quantity. Examples: a process's current heap
// size, a queue's current length.
//
// Cumulative indicates that the metric's values represent non-negative changes
// over specified time periods. Example: the number of rpc calls to a service.
enum class MetricKind : int { kGauge = 0, kCumulative };

// The type of the metric values.
enum class ValueType : int { kInt64 = 0, kHistogram };

// Everything in the internal namespace is implementation details. Do not depend
// on this.
namespace internal {

// Ensures that the string is a compile-time string literal.
class StringLiteral {
 public:
  // We allow implicit conversions here on purpose.
  template <int N>
  StringLiteral(const char (&data)[N]) : literal_(data, N - 1) {}

  // This ctor will be called for non-literals, causing compile-time failure.
  template <typename NotStringLiteral>
  StringLiteral(const NotStringLiteral& not_string_literal) = delete;

  // Implicit conversion to StringPiece.
  operator StringPiece() const { return literal_; }

 private:
  const StringPiece literal_;
};

template <typename Value>
ValueType GetValueType();

template <>
inline ValueType GetValueType<int64>() {
  return ValueType::kInt64;
}

template <>
inline ValueType GetValueType<HistogramProto>() {
  return ValueType::kHistogram;
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

  const std::vector<StringPiece> label_descriptions() const {
    return label_descriptions_;
  }

 private:
  template <MetricKind kind, typename Value, int NumLabels>
  friend class MetricDef;

  AbstractMetricDef(
      const MetricKind kind, const ValueType value_type,
      const internal::StringLiteral name,
      const internal::StringLiteral description,
      const std::vector<internal::StringLiteral>& label_descriptions)
      : kind_(kind),
        value_type_(value_type),
        name_(name),
        description_(description),
        label_descriptions_(std::vector<StringPiece>(
            label_descriptions.begin(), label_descriptions.end())) {}

  const MetricKind kind_;
  const ValueType value_type_;
  const StringPiece name_;
  const StringPiece description_;
  const std::vector<StringPiece> label_descriptions_;
};

// Metric definition.
//
// A metric is defined by its kind, value-type, name, description and the
// description of its labels.
//
// NOTE: We allow only string literals for the name, description and label
// descriptions because these should be fixed at compile-time and shouldn't be
// dynamic.
template <MetricKind metric_kind, typename Value, int NumLabels>
class MetricDef : public AbstractMetricDef {
 public:
  template <typename... LabelDesc>
  MetricDef(const internal::StringLiteral name,
            const internal::StringLiteral description,
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

#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_METRIC_DEF_H_

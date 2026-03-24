/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_METRICS_H_
#define XLA_METRICS_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xla {

const char kGlobalLoggingScope[] = "global";

template <typename T>
using MetricsContainer =
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::vector<T>>>;

struct MetricsContainersByType {
  MetricsContainer<int64_t> int_metrics;
  MetricsContainer<double> double_metrics;
  MetricsContainer<std::string> string_metrics;
};

// Tracks the metrics for a single thread. These will be merged with the other
// metrics at the very end of the compilation.
class MetricsPerThread {
 public:
  // Set the scope for the metrics processed by the current thread.
  void SetScope(absl::string_view scope);

  // Merges the metrics from each of the threads into a single set of metrics.
  // Note that there is no guarantee on the order that the metrics are merged.
  MetricsContainersByType GetMergedMetrics();

  void AddInt(absl::string_view key, int64_t value, bool is_scoped);
  void AddDouble(absl::string_view key, double value, bool is_scoped);
  void AddString(absl::string_view key, absl::string_view value,
                 bool is_scoped);

  void Clear();

 private:
  MetricsContainersByType metrics_;
  std::string scope_ = kGlobalLoggingScope;

  friend class GlobalMetricsHolder;
};

// Tracks the metrics for the entire compilation.
class GlobalMetricsHolder {
 public:
  void LogAll();

  // When clear is called, it is assumed there is only one thread remaining
  // that could be using the metrics, i.e. the current one.
  void Clear();

  MetricsContainersByType GetMergedMetrics();

  MetricsPerThread* GetMetricsFor(std::thread::id thread_id);

  void EnableMetricFamily(absl::string_view family);

  bool IsMetricFamilyEnabled(absl::string_view family);

  ~GlobalMetricsHolder();

 private:
  absl::Mutex mu_;
  absl::flat_hash_set<std::string> enabled_metric_families_;
  absl::flat_hash_map<std::thread::id, MetricsPerThread*> metrics_;
};

GlobalMetricsHolder& GetGlobalMetricsHolder();

void XLA_SET_METRICS_SCOPE(absl::string_view scope);

// This must be called at the very beginning of the compilation.
void XLA_ENABLE_METRIC_FAMILY(absl::string_view family);
bool XLA_IS_METRIC_FAMILY_ENABLED(absl::string_view family);
bool XLA_IS_METRICS_VLOG_ON(int level);

#define XLA_ADD_SCOPED_METRIC_I64(family, key, value)               \
  if (ABSL_PREDICT_FALSE(XLA_IS_METRICS_VLOG_ON(1))) {              \
    if (ABSL_PREDICT_FALSE(XLA_IS_METRIC_FAMILY_ENABLED(family))) { \
      AddMetricI64(family, key, value, true);                       \
    }                                                               \
  }

#define XLA_ADD_SCOPED_METRIC_F64(family, key, value)               \
  if (ABSL_PREDICT_FALSE(XLA_IS_METRICS_VLOG_ON(1))) {              \
    if (ABSL_PREDICT_FALSE(XLA_IS_METRIC_FAMILY_ENABLED(family))) { \
      AddMetricF64(family, key, value, true);                       \
    }                                                               \
  }

#define XLA_ADD_SCOPED_METRIC_STR(family, key, value)               \
  if (ABSL_PREDICT_FALSE(XLA_IS_METRICS_VLOG_ON(1))) {              \
    if (ABSL_PREDICT_FALSE(XLA_IS_METRIC_FAMILY_ENABLED(family))) { \
      AddMetricStr(family, key, value, true);                       \
    }                                                               \
  }

#define XLA_ADD_METRIC_I64(family, key, value)                      \
  if (ABSL_PREDICT_FALSE(XLA_IS_METRICS_VLOG_ON(1))) {              \
    if (ABSL_PREDICT_FALSE(XLA_IS_METRIC_FAMILY_ENABLED(family))) { \
      AddMetricI64(family, key, value);                             \
    }                                                               \
  }

#define XLA_ADD_METRIC_F64(family, key, value)                      \
  if (ABSL_PREDICT_FALSE(XLA_IS_METRICS_VLOG_ON(1))) {              \
    if (ABSL_PREDICT_FALSE(XLA_IS_METRIC_FAMILY_ENABLED(family))) { \
      AddMetricF64(family, key, value);                             \
    }                                                               \
  }

#define XLA_ADD_METRIC_STR(family, key, value)                      \
  if (ABSL_PREDICT_FALSE(XLA_IS_METRICS_VLOG_ON(1))) {              \
    if (ABSL_PREDICT_FALSE(XLA_IS_METRIC_FAMILY_ENABLED(family))) { \
      AddMetricStr(family, key, value);                             \
    }                                                               \
  }

void AddMetricI64(absl::string_view family, absl::string_view key,
                  int64_t value, bool is_scoped = false);
void AddMetricF64(absl::string_view family, absl::string_view key, double value,
                  bool is_scoped = false);
void AddMetricStr(absl::string_view family, absl::string_view key,
                  absl::string_view value, bool is_scoped = false);

void LogHloSchedulingMetrics();

}  // namespace xla

#endif  // XLA_METRICS_H_

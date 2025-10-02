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

#include "xla/metrics.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xla {

GlobalMetricsHolder global_metrics_holder;

GlobalMetricsHolder& GetGlobalMetricsHolder() { return global_metrics_holder; }

GlobalMetricsHolder::~GlobalMetricsHolder() {
  for (const auto& [_, metrics_by_thread] : metrics_) {
    delete metrics_by_thread;
  }
}

void GlobalMetricsHolder::EnableMetricFamily(absl::string_view family) {
  enabled_metric_families_.insert(std::string(family));
}

bool GlobalMetricsHolder::IsMetricFamilyEnabled(absl::string_view family) {
  return enabled_metric_families_.contains(family);
}

void GlobalMetricsHolder::Clear() {
  absl::MutexLock lock(mu_);
  MetricsPerThread* current_thread_metrics = nullptr;
  for (auto metrics_by_thread : metrics_) {
    if (metrics_by_thread.first == std::this_thread::get_id()) {
      current_thread_metrics = metrics_by_thread.second;
    } else {
      delete metrics_by_thread.second;
    }
  }
  metrics_.clear();
  if (current_thread_metrics != nullptr) {
    current_thread_metrics->Clear();
    metrics_[std::this_thread::get_id()] = current_thread_metrics;
  }
  enabled_metric_families_.clear();
}

MetricsContainersByType GlobalMetricsHolder::GetMergedMetrics() {
  absl::MutexLock lock(mu_);
  MetricsContainersByType merged_metrics;
  for (const auto& [thread_id, thread_metrics] : metrics_) {
    for (const auto& [key, scope_values] :
         thread_metrics->metrics_.int_metrics) {
      auto& merged_int_metrics_by_key = merged_metrics.int_metrics[key];
      for (const auto& [scope, values] : scope_values) {
        merged_int_metrics_by_key[scope].insert(
            merged_int_metrics_by_key[scope].end(), values.begin(),
            values.end());
      }
    }
    for (const auto& [key, scope_values] :
         thread_metrics->metrics_.double_metrics) {
      auto& merged_double_metrics_by_key = merged_metrics.double_metrics[key];
      for (const auto& [scope, values] : scope_values) {
        merged_double_metrics_by_key[scope].insert(
            merged_double_metrics_by_key[scope].end(), values.begin(),
            values.end());
      }
    }

    for (const auto& [key, scope_values] :
         thread_metrics->metrics_.string_metrics) {
      auto& merged_string_metrics_by_key = merged_metrics.string_metrics[key];
      for (const auto& [scope, values] : scope_values) {
        merged_string_metrics_by_key[scope].insert(
            merged_string_metrics_by_key[scope].end(), values.begin(),
            values.end());
      }
    }
  }

  return merged_metrics;
}

void GlobalMetricsHolder::LogAll() {
  LOG(INFO) << "Dumping metrics:";

  MetricsContainersByType merged_metrics = GetMergedMetrics();

  for (const auto& [key, scope_values] : merged_metrics.int_metrics) {
    LOG(INFO) << key << " : ";
    for (const auto& [scope, values] : scope_values) {
      LOG(INFO) << "   " << scope << " : " << absl::StrJoin(values, ",");
    }
  }

  for (const auto& [key, scope_values] : merged_metrics.double_metrics) {
    LOG(INFO) << key << " : ";
    for (const auto& [scope, values] : scope_values) {
      LOG(INFO) << "   " << scope << " : " << absl::StrJoin(values, ",");
    }
  }

  for (const auto& [key, scope_values] : merged_metrics.string_metrics) {
    LOG(INFO) << key << " : ";
    for (const auto& [scope, values] : scope_values) {
      LOG(INFO) << "   " << scope << " : " << absl::StrJoin(values, ",");
    }
  }
}

bool XLA_IS_METRICS_VLOG_ON(int level) { return VLOG_IS_ON(level); }

void MetricsPerThread::Clear() {
  metrics_.int_metrics.clear();
  metrics_.double_metrics.clear();
  metrics_.string_metrics.clear();
}

MetricsPerThread* GlobalMetricsHolder::GetMetricsFor(
    std::thread::id thread_id) {
  absl::MutexLock lock(mu_);
  metrics_[thread_id] = new MetricsPerThread();
  return metrics_[thread_id];
}

void MetricsPerThread::SetScope(absl::string_view scope) { scope_ = scope; }

void MetricsPerThread::AddInt(absl::string_view key, int64_t value,
                              bool is_scoped) {
  metrics_.int_metrics[key][is_scoped ? scope_ : kGlobalLoggingScope].push_back(
      value);
}

void MetricsPerThread::AddDouble(absl::string_view key, double value,
                                 bool is_scoped) {
  metrics_.double_metrics[key][is_scoped ? scope_ : kGlobalLoggingScope]
      .push_back(value);
}

void MetricsPerThread::AddString(absl::string_view key, absl::string_view value,
                                 bool is_scoped) {
  metrics_.string_metrics[key][is_scoped ? scope_ : kGlobalLoggingScope]
      .emplace_back(value.begin(), value.end());
}

MetricsPerThread* xla_get_metrics() {
  thread_local MetricsPerThread* metrics_per_thread =
      global_metrics_holder.GetMetricsFor(std::this_thread::get_id());
  return metrics_per_thread;
}

void XLA_ENABLE_METRIC_FAMILY(absl::string_view family) {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
    auto& global_metrics = GetGlobalMetricsHolder();
    global_metrics.EnableMetricFamily(family);
  }
}

bool XLA_IS_METRIC_FAMILY_ENABLED(absl::string_view family) {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
    return GetGlobalMetricsHolder().IsMetricFamilyEnabled(family);
  }
  return false;
}

void XLA_SET_METRICS_SCOPE(absl::string_view scope) {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1))) {
    auto* metrics = xla_get_metrics();
    metrics->SetScope(scope);
  }
}

void AddMetricI64(absl::string_view family, absl::string_view key,
                  int64_t value, bool is_scoped) {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1) &&
                         XLA_IS_METRIC_FAMILY_ENABLED(family))) {
    auto* metrics = xla_get_metrics();
    metrics->AddInt(key, value, is_scoped);
  }
}

void AddMetricF64(absl::string_view family, absl::string_view key, double value,
                  bool is_scoped) {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1) &&
                         XLA_IS_METRIC_FAMILY_ENABLED(family))) {
    auto* metrics = xla_get_metrics();
    metrics->AddDouble(key, value, is_scoped);
  }
}

void AddMetricStr(absl::string_view family, absl::string_view key,
                  absl::string_view value, bool is_scoped) {
  if (ABSL_PREDICT_FALSE(VLOG_IS_ON(1) &&
                         XLA_IS_METRIC_FAMILY_ENABLED(family))) {
    auto* metrics = xla_get_metrics();
    metrics->AddString(key, value, is_scoped);
  }
}

void LogHloSchedulingMetrics() {
  if (VLOG_IS_ON(1) && XLA_IS_METRIC_FAMILY_ENABLED("hlo-scheduling")) {
    auto metrics = GetGlobalMetricsHolder().GetMergedMetrics();
    int64_t no_fragmentation_peak_memory_heap_simulator =
        metrics.int_metrics["final-peak-memory"][kGlobalLoggingScope].back();
    int64_t buffer_assignment_peak_memory =
        metrics
            .int_metrics["final-peak-memory-with-fragmentation"]
                        [kGlobalLoggingScope]
            .back();
    std::string memory_scheduler =
        metrics.string_metrics["memory-scheduler"][kGlobalLoggingScope].back();

    LOG(INFO) << "TPU memory peak (no fragmentation): "
              << no_fragmentation_peak_memory_heap_simulator;
    LOG(INFO) << "Memory scheduler used: " << memory_scheduler;
    LOG(INFO) << "TPU memory peak (rel.difference):"
              << 1.0 - static_cast<double>(
                           no_fragmentation_peak_memory_heap_simulator) /
                           buffer_assignment_peak_memory;
    LOG(INFO) << "TPU memory peak (abs.difference):"
              << buffer_assignment_peak_memory -
                     no_fragmentation_peak_memory_heap_simulator;
  }
}

}  // namespace xla

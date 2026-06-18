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

#include "xla/tsl/lib/monitoring/collection_registry.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/metric_def.h"
#include "xla/tsl/platform/env.h"

// We replace this implementation with a null implementation for mobile
// platforms.
#ifndef IS_MOBILE_PLATFORM

#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace monitoring {
namespace internal {

void Collector::CollectMetricValues(
    const CollectionRegistry::CollectionInfo& collection_info) {
  collection_info.collection_function(
      MetricCollectorGetter(this, collection_info.metric_def,
                            collection_info.registration_time_millis));
}

std::unique_ptr<CollectedMetrics> Collector::ConsumeCollectedMetrics() {
  absl::MutexLock l(mu_);
  return std::move(collected_metrics_);
}

void Collector::CollectMetricDescriptor(
    const AbstractMetricDef* const metric_def) {
  auto* const metric_descriptor = [&]() {
    absl::MutexLock l(mu_);
    return collected_metrics_->metric_descriptor_map
        .insert(std::make_pair(std::string(metric_def->name()),
                               std::make_unique<MetricDescriptor>()))
        .first->second.get();
  }();
  metric_descriptor->name = std::string(metric_def->name());
  metric_descriptor->description = std::string(metric_def->description());

  for (const absl::string_view label_name : metric_def->label_descriptions()) {
    metric_descriptor->label_names.emplace_back(label_name);
  }

  metric_descriptor->metric_kind = metric_def->kind();
  metric_descriptor->value_type = metric_def->value_type();
}

}  // namespace internal

// static
CollectionRegistry* CollectionRegistry::Default() {
  static CollectionRegistry* default_registry =
      new CollectionRegistry(Env::Default());
  return default_registry;
}

CollectionRegistry::CollectionRegistry(Env* const env) : env_(env) {}

std::unique_ptr<CollectionRegistry::RegistrationHandle>
CollectionRegistry::Register(const AbstractMetricDef* const metric_def,
                             const CollectionFunction& collection_function) {
  CHECK(collection_function)
      << "Requires collection_function to contain an implementation.";

  absl::MutexLock l(mu_);

  const auto found_it = registry_.find(metric_def->name());
  if (found_it != registry_.end()) {
    LOG(WARNING)
        << "Trying to register 2 metrics with the same name: "
        << metric_def->name()
        << ". The old value will be erased in order to register a new one. "
           "Please check if you link the metric more than once, or "
           "if the name is already used by other metrics.";
    // Erase the old value and insert the new value to registry.
    registry_.erase(found_it);
  }
  registry_.insert(
      {metric_def->name(),
       {metric_def, collection_function, env_->NowMicros() / 1000}});

  return std::make_unique<RegistrationHandle>(this, metric_def);
}

void CollectionRegistry::Unregister(const AbstractMetricDef* const metric_def) {
  absl::MutexLock l(mu_);
  registry_.erase(metric_def->name());
}

std::unique_ptr<CollectedMetrics> CollectionRegistry::CollectMetrics(
    const CollectMetricsOptions& options) const {
  internal::Collector collector(env_->NowMicros() / 1000);

  absl::MutexLock l(mu_);
  // Though `registry_` doesn't itself have a deterministic iteration order,
  // `CollectMetricDescriptor` inserts each metric into an ordered container,
  // implicitly sorting the metrics in the process. This loop's result is
  // therefore deterministic.
  for (const auto& [unused_name, collection_info] : registry_) {  // NOLINT
    if (options.collect_metric_descriptors) {
      collector.CollectMetricDescriptor(collection_info.metric_def);
    }

    collector.CollectMetricValues(collection_info);
  }
  return collector.ConsumeCollectedMetrics();
}

}  // namespace monitoring
}  // namespace tsl

#endif  // IS_MOBILE_PLATFORM

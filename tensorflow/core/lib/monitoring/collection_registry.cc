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

#include "tensorflow/core/lib/monitoring/collection_registry.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace monitoring {
namespace internal {

void Collector::CollectMetricValues(
    const CollectionRegistry::CollectionInfo& info) {
  info.collection_function(MetricCollectorGetter(
      this, info.metric_def, info.registration_time_millis));
}

std::unique_ptr<CollectedMetrics> Collector::ConsumeCollectedMetrics() {
  mutex_lock l(mu_);
  return std::move(collected_metrics_);
}

void Collector::CollectMetricDescriptor(
    const AbstractMetricDef* const metric_def) {
  auto* const metric_descriptor = [&]() {
    mutex_lock l(mu_);
    return collected_metrics_->metric_descriptor_map
        .insert(std::make_pair(
            metric_def->name().ToString(),
            std::unique_ptr<MetricDescriptor>(new MetricDescriptor())))
        .first->second.get();
  }();
  metric_descriptor->name = metric_def->name().ToString();
  metric_descriptor->description = metric_def->description().ToString();

  for (const StringPiece label_name : metric_def->label_descriptions()) {
    metric_descriptor->label_names.push_back(label_name.ToString());
  }

  // Only cumulative int64 counter is implemented at the moment.
  metric_descriptor->metric_kind = MetricKind::kCumulative;
  metric_descriptor->value_type = ValueType::kInt64;
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

  mutex_lock l(mu_);

  const auto found_it = registry_.find(metric_def->name());
  if (found_it != registry_.end()) {
    LOG(FATAL) << "Cannot register 2 metrics with the same name: "
               << metric_def->name();
  }
  registry_.insert(
      {metric_def->name(),
       {metric_def, collection_function, env_->NowMicros() / 1000}});

  return std::unique_ptr<RegistrationHandle>(
      new RegistrationHandle(this, metric_def));
}

void CollectionRegistry::Unregister(const AbstractMetricDef* const metric_def) {
  mutex_lock l(mu_);
  registry_.erase(metric_def->name());
}

std::unique_ptr<CollectedMetrics> CollectionRegistry::CollectMetrics(
    const CollectMetricsOptions& options) const {
  internal::Collector collector(env_->NowMicros() / 1000);

  mutex_lock l(mu_);
  for (const auto& registration : registry_) {
    if (options.collect_metric_descriptors) {
      collector.CollectMetricDescriptor(registration.second.metric_def);
    }

    collector.CollectMetricValues(registration.second /* collection_info */);
  }
  return collector.ConsumeCollectedMetrics();
}

}  // namespace monitoring
}  // namespace tensorflow

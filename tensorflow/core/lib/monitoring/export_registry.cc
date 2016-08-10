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

#include "tensorflow/core/lib/monitoring/export_registry.h"

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace monitoring {

ExportRegistry* ExportRegistry::Default() {
  static ExportRegistry* default_registry = new ExportRegistry();
  return default_registry;
}

std::unique_ptr<ExportRegistry::RegistrationHandle> ExportRegistry::Register(
    const AbstractMetricDef* const metric_def) {
  mutex_lock l(mu_);

  LOG(INFO) << "Here." << registry_.size();
  const auto found_it = registry_.find(metric_def->name());
  if (found_it != registry_.end()) {
    LOG(INFO) << "Here2";
    LOG(FATAL) << "Cannot register 2 metrics with the same name: "
               << metric_def->name();
  }
  LOG(INFO) << "Here3";
  registry_.insert({metric_def->name(), metric_def});
  LOG(INFO) << "Here4." << registry_.size();

  return std::unique_ptr<RegistrationHandle>(
      new RegistrationHandle(this, metric_def));
}

void ExportRegistry::Unregister(const AbstractMetricDef* const metric_def) {
  mutex_lock l(mu_);
  registry_.erase(metric_def->name());
}

}  // namespace monitoring
}  // namespace tensorflow

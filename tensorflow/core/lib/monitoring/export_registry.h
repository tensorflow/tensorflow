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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_EXPORT_REGISTRY_H_
#define THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_EXPORT_REGISTRY_H_

#include <map>
#include <memory>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/monitoring/metric_def.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"

namespace tensorflow {
namespace monitoring {

// An export registry for metrics.
//
// Metrics are registered here so that their state can be exported later using
// an exporter.
//
// This class is thread-safe.
class ExportRegistry {
 public:
  ~ExportRegistry() = default;

  // Returns the default registry for the process.
  //
  // This registry belongs to this library and should never be deleted.
  static ExportRegistry* Default();

  // Registers the metric and returns a Registration object. The destruction of
  // the registration object would cause the metric to be unregistered from this
  // registry.
  //
  // IMPORTANT: Delete the handle before the metric-def is deleted.
  class RegistrationHandle;
  std::unique_ptr<RegistrationHandle> Register(
      const AbstractMetricDef* metric_def)
      LOCKS_EXCLUDED(mu_) TF_MUST_USE_RESULT;

 private:
  ExportRegistry() = default;

  // Unregisters the metric from this registry. This is private because the
  // public interface provides a Registration handle which automatically calls
  // this upon destruction.
  void Unregister(const AbstractMetricDef* metric_def) LOCKS_EXCLUDED(mu_);

  mutable mutex mu_;
  std::map<StringPiece, const AbstractMetricDef*> registry_ GUARDED_BY(mu_);
};

////
// Implementation details follow. API readers may skip.
////

class ExportRegistry::RegistrationHandle {
 public:
  RegistrationHandle(ExportRegistry* const export_registry,
                     const AbstractMetricDef* const metric_def)
      : export_registry_(export_registry), metric_def_(metric_def) {}

  ~RegistrationHandle() { export_registry_->Unregister(metric_def_); }

 private:
  ExportRegistry* const export_registry_;
  const AbstractMetricDef* const metric_def_;
};

}  // namespace monitoring
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_LIB_MONITORING_EXPORT_REGISTRY_H_

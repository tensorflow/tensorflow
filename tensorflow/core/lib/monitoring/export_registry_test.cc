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

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace monitoring {
namespace {

TEST(ExportRegistryTest, RegistrationUnregistration) {
  auto* export_registry = ExportRegistry::Default();
  const MetricDef<MetricKind::CUMULATIVE, int64, 0> metric_def0(
      "/tensorflow/metric0", "An example metric with no labels.");
  const MetricDef<MetricKind::GAUGE, double, 1> metric_def1(
      "/tensorflow/metric1", "An example metric with one label.", "LabelName");

  {
    // Enclosed in a scope so that we unregister before the stack variables
    // above are destroyed.

    std::unique_ptr<ExportRegistry::RegistrationHandle> handle0 =
        export_registry->Register(&metric_def0);
    std::unique_ptr<ExportRegistry::RegistrationHandle> handle1 =
        export_registry->Register(&metric_def1);

    handle0.reset();

    // Able to register again because it was unregistered earlier.
    handle0 = export_registry->Register(&metric_def0);
  }
}

TEST(ExportRegistryDeathTest, DuplicateRegistration) {
  auto* export_registry = ExportRegistry::Default();
  const MetricDef<MetricKind::CUMULATIVE, int64, 0> metric_def(
      "/tensorflow/metric", "An example metric with no labels.");

  auto handle = export_registry->Register(&metric_def);
  EXPECT_DEATH(
      { auto duplicate_handle = export_registry->Register(&metric_def); },
      "/tensorflow/metric");
}

}  // namespace
}  // namespace monitoring
}  // namespace tensorflow

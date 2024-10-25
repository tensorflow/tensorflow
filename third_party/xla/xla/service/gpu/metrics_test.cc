/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/metrics.h"

#include <memory>
#include <string>
#include <vector>

#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

TEST(MetricsTest, RecordsGpuCompilerStacktrace) {
  const std::string kGpuCompilerStacktraceMetricName =
      "/xla/service/gpu/compiler_stacktrace_count";

  RecordGpuCompilerStacktrace();

  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);

  EXPECT_TRUE(metrics->point_set_map.find(kGpuCompilerStacktraceMetricName) !=
              metrics->point_set_map.end());
  EXPECT_EQ(
      metrics->point_set_map[kGpuCompilerStacktraceMetricName]->points.size(),
      1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

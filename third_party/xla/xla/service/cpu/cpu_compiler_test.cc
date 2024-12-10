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
#include <memory>
#include <string>
#include <utility>

#include "xla/pjrt/pjrt_client.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

std::unique_ptr<HloRunnerInterface> CreateHloRunner() {
  if (!ShouldUsePjRt()) {
    return std::make_unique<HloRunner>(
        PlatformUtil::GetDefaultPlatform().value());
  }

  PjRtClientTestFactoryRegistry& pjrt_registry =
      GetGlobalPjRtClientTestFactory();
  std::unique_ptr<PjRtClient> client = pjrt_registry.Get()().value();
  PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFn
      device_shape_representation_fn =
          pjrt_registry.GetDeviceShapeRepresentationFn(client.get());
  PjRtClientTestFactoryRegistry::DeviceShapeSizeFn device_shape_size_fn =
      pjrt_registry.GetDeviceShapeSizeFn(client.get());
  return std::make_unique<HloRunnerPjRt>(
      std::move(client), [](const Shape& host_shape) { return host_shape; },
      device_shape_size_fn);
}

class CpuCompilerTest : public HloRunnerAgnosticTestBase {
 public:
  CpuCompilerTest()
      : HloRunnerAgnosticTestBase(CreateHloRunner(), CreateHloRunner()) {}
};

TEST_F(CpuCompilerTest, RecordsStreamzStackTrace) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY main {
      p = f32[10]{0} parameter(0)
      ROOT neg = f32[10]{0} negate(p)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));

  const std::string kCpuCompilerStacktraceMetricName =
      "/xla/service/cpu/compiler_stacktrace_count";

  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);
  EXPECT_TRUE(metrics->point_set_map.find(kCpuCompilerStacktraceMetricName) !=
              metrics->point_set_map.end());

  // Since Streamz is recorded every call, we expect at least one point.
  // All other callers may increment the counter as well.
  EXPECT_GT(
      metrics->point_set_map[kCpuCompilerStacktraceMetricName]->points.size(),
      0);
}

}  // namespace
}  // namespace cpu
}  // namespace xla

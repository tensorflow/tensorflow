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

#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/monitoring/collected_metrics.h"
#include "xla/tsl/lib/monitoring/collection_registry.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

using CpuCompilerTest = HloPjRtTestBase;

constexpr absl::string_view kCpuCompilerStacktraceMetricName =
    "/xla/service/cpu/compiler_stacktrace_count";

TEST_F(CpuCompilerTest, RecordsStreamzStackTrace) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "Streamz is not supported in OSS.";
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
    HloModule test
    ENTRY main {
      p = f32[10]{0} parameter(0)
      ROOT neg = f32[10]{0} negate(p)
    }
  )"));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));

  tsl::monitoring::CollectionRegistry::CollectMetricsOptions options;
  std::unique_ptr<tsl::monitoring::CollectedMetrics> metrics =
      tsl::monitoring::CollectionRegistry::Default()->CollectMetrics(options);

  const auto it = metrics->point_set_map.find(
      std::string(kCpuCompilerStacktraceMetricName));
  ASSERT_TRUE(it != metrics->point_set_map.end());

  // Since Streamz is recorded every call, we expect at least one point.
  // All other callers may increment the counter as well.
  EXPECT_GT(it->second->points.size(), 0);
}

TEST_F(CpuCompilerTest, CompilationWithLargeConstants) {
  absl::string_view module_string = R"(
HloModule module

ENTRY main {
  a = f32[1000,1000]{1,0} parameter(0)
  b = f32[1000,1000]{1,0} constant({...})
  a_plus_b = f32[1000,1000]{1,0} add(a, b)
  c = f32[1000,1000]{1,0} constant({...})
  ROOT result = f32[1000,1000]{1,0} add(a_plus_b, c)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(module_string));

  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

}  // namespace
}  // namespace cpu
}  // namespace xla

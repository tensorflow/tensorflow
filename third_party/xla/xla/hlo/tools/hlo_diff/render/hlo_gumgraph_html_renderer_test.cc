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

#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_html_renderer.h"

#include <cstdint>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/render/op_metric_getter.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

// A mock OpMetricGetter for testing.
class MockOpMetricGetter : public OpMetricGetter {
 public:
  MOCK_METHOD(absl::StatusOr<uint64_t>, GetOpTimePs, (absl::string_view),
              (const, override));
};

TEST(HloGumgraphHtmlRendererTest, RenderHtml) {
  DiffResult diff_result;
  DiffSummary diff_summary;
  std::ostringstream out;
  RenderHtml(diff_result, diff_summary, out);
  EXPECT_THAT(out.str(), HasSubstr("<style>"));
  EXPECT_THAT(out.str(), HasSubstr("<script>"));
}

TEST(HloGumgraphHtmlRendererTest, RenderHtmlWithOpMetrics) {
  DiffResult diff_result;
  DiffSummary diff_summary;
  std::ostringstream out;
  MockOpMetricGetter op_metrics;
  RenderHtml(diff_result, diff_summary, nullptr, &op_metrics, &op_metrics, out);
  EXPECT_THAT(out.str(), HasSubstr("Profile Metrics Diff"));
}

TEST(HloGumgraphHtmlRendererTest, RenderHtmlWithoutOpMetrics) {
  DiffResult diff_result;
  DiffSummary diff_summary;
  std::ostringstream out;
  RenderHtml(diff_result, diff_summary, nullptr, nullptr, nullptr, out);
  EXPECT_THAT(out.str(), Not(HasSubstr("Profile Metrics Diff")));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla

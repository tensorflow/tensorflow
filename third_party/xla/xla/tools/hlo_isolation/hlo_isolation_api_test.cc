/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tools/hlo_isolation/hlo_isolation_api.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tools/hlo_dump/hlo_dump_utils.h"
#include "xla/tools/hlo_isolation/hlo_inf_nan_intent_analyzer.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace hlo_isolation {
namespace {

TEST(HloIsolationApiTest, LiteralContainsInfOrNan) {
  Literal f32_normal = LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f});
  EXPECT_FALSE(LiteralContainsInfOrNan(f32_normal));

  Literal f32_inf = LiteralUtil::CreateR1<float>(
      {1.0f, std::numeric_limits<float>::infinity(), 3.0f});
  EXPECT_TRUE(LiteralContainsInfOrNan(f32_inf));

  Literal f32_neg_inf = LiteralUtil::CreateR1<float>(
      {-std::numeric_limits<float>::infinity(), 2.0f, 3.0f});
  EXPECT_TRUE(LiteralContainsInfOrNan(f32_neg_inf));

  Literal f32_nan = LiteralUtil::CreateR1<float>(
      {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f});
  EXPECT_TRUE(LiteralContainsInfOrNan(f32_nan));

  Literal s32_normal = LiteralUtil::CreateR1<int32_t>({1, 2, 3});
  EXPECT_FALSE(LiteralContainsInfOrNan(s32_normal));

  Literal f32_2d_inf = LiteralUtil::CreateR2<float>(
      {{1.0f, 2.0f}, {std::numeric_limits<float>::infinity(), 4.0f}});
  EXPECT_TRUE(LiteralContainsInfOrNan(f32_2d_inf));

  Literal tuple_with_inf =
      LiteralUtil::MakeTuple({&f32_normal, &f32_inf, &s32_normal});
  EXPECT_TRUE(LiteralContainsInfOrNan(tuple_with_inf));

  Literal tuple_no_inf = LiteralUtil::MakeTuple({&f32_normal, &s32_normal});
  EXPECT_FALSE(LiteralContainsInfOrNan(tuple_no_inf));
}

TEST(HloIsolationApiTest, ModuleContainsConstantInfOrNan) {
  const absl::string_view hlo_no_inf = R"hlo(
HloModule module_no_inf
ENTRY main {
  p0 = f32[10] parameter(0)
  c0 = f32[] constant(1.0)
  b0 = f32[10] broadcast(c0), dimensions={}
  ROOT add = f32[10] add(p0, b0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_1,
                       xla::ParseAndReturnUnverifiedModule(hlo_no_inf));
  EXPECT_FALSE(ModuleContainsConstantInfOrNan(*module_1));

  const absl::string_view hlo_with_neg_inf = R"hlo(
HloModule module_with_neg_inf
ENTRY main {
  p0 = pred[10] parameter(0)
  c_inf = f32[] constant(-inf)
  b_inf = f32[10] broadcast(c_inf), dimensions={}
  p1 = f32[10] parameter(1)
  ROOT sel = f32[10] select(p0, b_inf, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_2,
                       xla::ParseAndReturnUnverifiedModule(hlo_with_neg_inf));
  EXPECT_TRUE(ModuleContainsConstantInfOrNan(*module_2));

  const absl::string_view hlo_with_nan = R"hlo(
HloModule module_with_nan
ENTRY main {
  c_nan = f32[] constant(nan)
  ROOT b_nan = f32[10] broadcast(c_nan), dimensions={}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_3,
                       xla::ParseAndReturnUnverifiedModule(hlo_with_nan));
  EXPECT_TRUE(ModuleContainsConstantInfOrNan(*module_3));
}

TEST(HloIsolationApiTest, RunIsolationTestRespectsRejectUnconstrainedOps) {
  const absl::string_view kHlo = R"hlo(
HloModule masked_reduction_with_sqrt
%max_reducer (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum = f32[] maximum(%x, %y)
}

ENTRY main {
  mask = pred[10] parameter(0)
  data = f32[10] parameter(1)
  data_sqrt = f32[10] sqrt(data)
  c_neg_inf = f32[] constant(-inf)
  b_neg_inf = f32[10] broadcast(c_neg_inf), dimensions={}
  sel = f32[10] select(mask, data_sqrt, b_neg_inf)
  ROOT r = f32[] reduce(sel, c_neg_inf), dimensions={0}, to_apply=%max_reducer
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  ModuleIsolationOptions options;
  options.run_module_fn =
      [](std::unique_ptr<HloModule> /*m*/, HloRunnerInterface* /*r*/,
         absl::Span<const Literal> /*i*/,
         const RunModuleOptions& /*run_opts*/) -> absl::StatusOr<Literal> {
    return LiteralUtil::CreateR0<float>(0.0f);
  };

  options.reject_unconstrained_ops = true;
  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result_rejected,
      RunIsolationTestOnModule(*module, nullptr, nullptr, options));
  EXPECT_FALSE(result_rejected.is_intentional_inf_nan());

  options.reject_unconstrained_ops = false;
  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result_allowed,
      RunIsolationTestOnModule(*module, nullptr, nullptr, options));
  EXPECT_TRUE(result_allowed.is_intentional_inf_nan());
}

TEST(HloIsolationApiTest, ParseMismatchLineWithCoordinates) {
  std::string line =
      "actual 1.25, expected 2.5, index {1, 3, 5}, rel error 0.5, abs error "
      "1.25";
  auto mismatch_or = ParseMismatchLine(line);
  ASSERT_OK(mismatch_or);
  const auto& mismatch = *mismatch_or;
  EXPECT_DOUBLE_EQ(mismatch.actual(), 1.25);
  EXPECT_DOUBLE_EQ(mismatch.expected(), 2.5);
  EXPECT_DOUBLE_EQ(mismatch.rel_error(), 0.5);
  ASSERT_EQ(mismatch.top_mismatch_index_size(), 3);
  EXPECT_EQ(mismatch.top_mismatch_index(0), 1);
  EXPECT_EQ(mismatch.top_mismatch_index(1), 3);
  EXPECT_EQ(mismatch.top_mismatch_index(2), 5);
}

TEST(HloIsolationApiTest, ExtractMismatchDetailsPopulatesBoundingBox) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_extract
ENTRY main {
  p0 = f32[4, 5] parameter(0)
  p1 = f32[4, 5] parameter(1)
  ROOT add = f32[4, 5] add(p0, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(hlo_string));

  HloIsolationTestResult result;
  NumericCheck* check = result.add_numeric_checks();
  check->set_name("TPU_VS_INTERPRETER");
  NumericMismatch* mismatch = check->add_top_mismatches();
  mismatch->set_actual(1.0);
  mismatch->set_expected(2.0);
  mismatch->set_rel_error(0.5);
  mismatch->add_tensor_dimensions(4);
  mismatch->add_tensor_dimensions(5);
  mismatch->add_mismatch_box_min(1);
  mismatch->add_mismatch_box_min(2);
  mismatch->add_mismatch_box_max(3);
  mismatch->add_mismatch_box_max(4);
  mismatch->add_top_mismatch_index(2);
  mismatch->add_top_mismatch_index(3);
  mismatch->set_mismatch_count(7);
  mismatch->set_total_elements(20);

  std::vector<numerics::debug_info::MismatchDetails> details =
      ExtractMismatchDetails(*module, result);
  ASSERT_EQ(details.size(), 1);
  EXPECT_EQ(details[0].target_instruction_name, "add");
  EXPECT_DOUBLE_EQ(details[0].actual, 1.0);
  EXPECT_DOUBLE_EQ(details[0].expected, 2.0);
  EXPECT_DOUBLE_EQ(details[0].rel_error, 0.5);
  ASSERT_TRUE(details[0].bounding_box.has_value());
  const auto& bbox = *details[0].bounding_box;
  EXPECT_EQ(bbox.tensor_shape, (std::vector<int64_t>{4, 5}));
  EXPECT_EQ(bbox.box_min, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(bbox.box_max, (std::vector<int64_t>{3, 4}));
  ASSERT_EQ(bbox.top_mismatch_coords.size(), 1);
  EXPECT_EQ(bbox.top_mismatch_coords[0], (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(bbox.mismatch_count, 7);
  EXPECT_EQ(bbox.total_elements, 20);
}

TEST(HloIsolationApiTest, ExtractMismatchDetailsPopulatesScalarBoundingBox) {
  const absl::string_view hlo_string = R"hlo(
HloModule test_scalar
ENTRY main {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(hlo_string));

  HloIsolationTestResult result;
  NumericCheck* check = result.add_numeric_checks();
  check->set_name("TPU_VS_INTERPRETER");
  NumericMismatch* mismatch = check->add_top_mismatches();
  mismatch->set_actual(3.0);
  mismatch->set_expected(4.0);
  mismatch->set_rel_error(0.25);
  mismatch->set_mismatch_count(1);
  mismatch->set_total_elements(1);

  std::vector<numerics::debug_info::MismatchDetails> details =
      ExtractMismatchDetails(*module, result);
  ASSERT_EQ(details.size(), 1);
  EXPECT_EQ(details[0].target_instruction_name, "add");
  EXPECT_DOUBLE_EQ(details[0].actual, 3.0);
  EXPECT_DOUBLE_EQ(details[0].expected, 4.0);
  EXPECT_DOUBLE_EQ(details[0].rel_error, 0.25);
}

}  // namespace
}  // namespace hlo_isolation
}  // namespace xla

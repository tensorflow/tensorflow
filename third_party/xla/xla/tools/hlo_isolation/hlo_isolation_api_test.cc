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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tools/hlo_dump/hlo_dump_utils.h"
#include "xla/tools/hlo_isolation/hlo_inf_nan_intent_analyzer.h"
#include "xla/tools/hlo_isolation/hlo_isolation.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

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
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
         absl::Span<const Literal> i,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
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

TEST(HloIsolationApiTest, DefusedReferenceOutOfMemoryIsSkippedNotFailed) {
  const absl::string_view kHlo = R"hlo(
HloModule m
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  ModuleIsolationOptions options;
  options.run_module_fn =
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
         absl::Span<const Literal> i,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (absl::StrContains(m->name(), "defused")) {
      return absl::ResourceExhaustedError("Ran out of memory on HBM");
    }
    return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
  };

  // Without a reference runner the defused module is the only reference, so an
  // unrunnable defused module leaves nothing to compare against. That is
  // inconclusive, not a failure.
  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult test_result,
      RunIsolationTestOnModule(*module, nullptr, nullptr, options));
  EXPECT_EQ(test_result.state(), State::SKIPPED);
  EXPECT_EQ(test_result.reason(), "DEFUSED_REFERENCE_UNAVAILABLE");
}

TEST(HloIsolationApiTest, MismatchIsRetriedWithoutExcessPrecision) {
  const absl::string_view kHlo = R"hlo(
HloModule m
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_allow_excess_precision(true);

  ModuleIsolationOptions options;
  options.retry_without_excess_precision = true;
  // Models a fusion that holds an intermediate in a wider type than the HLO
  // declares: the fused and defused runs only agree once excess precision is
  // disabled.
  options.run_module_fn =
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
         absl::Span<const Literal> i,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (m->config().debug_options().xla_allow_excess_precision() &&
        absl::StrContains(m->name(), "defused")) {
      return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
    }
    return LiteralUtil::CreateR1<float>({1.0f, 1.0f, 1.0f, 1.0f});
  };

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*module, nullptr, nullptr, options));
  EXPECT_EQ(result.state(), State::SUCCESS);
  EXPECT_EQ(result.reason(), "STAGE_1B_NO_EXCESS_PRECISION_SUCCESS");
}

// A stand-in for a reference runner. `run_module_fn` is stubbed out in these
// tests, and RunIsolationTestOnModule only ever tests this pointer for null and
// forwards it, so it is never dereferenced.
HloRunnerInterface* FakeReferenceRunner() {
  static int token = 0;
  return reinterpret_cast<HloRunnerInterface*>(&token);
}

TEST(HloIsolationApiTest, DefusedReferenceOutOfMemoryIsSkippedNotEscalated) {
  const absl::string_view kHlo = R"hlo(
HloModule m
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  ModuleIsolationOptions options;
  options.run_module_fn =
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
         absl::Span<const Literal> i,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (absl::StrContains(m->name(), "defused")) {
      return absl::ResourceExhaustedError("Ran out of memory on HBM");
    }
    return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
  };

  // If the defused module does not fit in device memory, the interpreter --
  // which materializes the same intermediates on the host, far more slowly --
  // will not fit either, and trying it just times the test out. The module is
  // skipped even though a reference runner is available.
  ASSERT_OK_AND_ASSIGN(HloIsolationTestResult result,
                       RunIsolationTestOnModule(
                           *module, nullptr, FakeReferenceRunner(), options));
  EXPECT_EQ(result.state(), State::SKIPPED);
  EXPECT_EQ(result.reason(), "DEFUSED_REFERENCE_UNAVAILABLE");
}

TEST(HloIsolationApiTest, MismatchIsReportedExactlyOnce) {
  const absl::string_view kHlo = R"hlo(
HloModule m
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_allow_excess_precision(true);

  ModuleIsolationOptions options;
  options.retry_without_excess_precision = true;
  // Mismatches on every check, including the no-excess-precision retry.
  options.run_module_fn =
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
         absl::Span<const Literal> i,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (absl::StrContains(m->name(), "defused")) {
      return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
    }
    return LiteralUtil::CreateR1<float>({1.0f, 1.0f, 1.0f, 1.0f});
  };
  int mismatch_reports = 0;
  options.on_mismatch_fn =
      [&mismatch_reports](const HloModule& m, const Literal& test_output,
                          const Literal& ref_output,
                          const absl::Status& status) { ++mismatch_reports; };

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*module, nullptr, nullptr, options));
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  // Two checks mismatched (TPU_VS_DEFUSED_TPU and the no-excess-precision
  // retry), but the module is only reported once, at the final verdict.
  EXPECT_EQ(mismatch_reports, 1);
  EXPECT_EQ(result.numeric_checks_size(), 2);
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

TEST(HloIsolationApiTest, DefaultModuleSizeEstimateHandlesTupleResults) {
  const absl::string_view kHlo = R"hlo(
HloModule m
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT t = (f32[4], f32[4]) tuple(p0, p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  ModuleIsolationOptions options;
  options.max_module_size_bytes = 1;
  options.run_module_fn =
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
         absl::Span<const Literal> i,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (absl::StrContains(m->name(), "defused")) {
      return absl::ResourceExhaustedError("Ran out of memory on HBM");
    }
    Literal element = LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
    return LiteralUtil::MakeTuple({&element, &element});
  };

  // Estimating the size of a tuple result used to abort; the module is now
  // only too large for the reference run.
  ASSERT_OK_AND_ASSIGN(HloIsolationTestResult result,
                       RunIsolationTestOnModule(
                           *module, nullptr, FakeReferenceRunner(), options));
  EXPECT_EQ(result.state(), State::SKIPPED);
  EXPECT_EQ(result.reason(), "DEFUSED_REFERENCE_UNAVAILABLE");
}

TEST(HloIsolationApiTest, SubnormalFlushToZeroTrappedUnderDefaultTolerance) {
  const absl::string_view kHlo = R"hlo(
HloModule subnormal_flush
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT copy = f32[4] copy(p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(
      LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f, subnormal_f32, 1.0f}));

  ModuleIsolationOptions options;
  // Both fused and defused device runs flush the subnormal to 0.0f, while the
  // interpreter reference preserves denorm_min(). On a bit preserving op such
  // as `copy`, even under the default tolerance (abs=0.01, rel=0.1), the
  // subnormal flush to zero must fail. The normal elements differ by rounding
  // noise that the tolerance accepts and must not be counted as mismatches.
  options.run_module_fn =
      [subnormal_f32](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      Literal ref = LiteralUtil::CreateR1<float>(
          {subnormal_f32, 2.0f, subnormal_f32, 2.0f});
      if (run_opts.expected_literals != nullptr) {
        (*run_opts.expected_literals)["copy"] =
            std::make_shared<Literal>(ref.Clone());
      }
      return ref;
    }
    Literal flushed = LiteralUtil::CreateR1<float>({0.0f, 2.001f, 0.0f, 2.0f});
    for (const auto& cb : run_opts.hlo_output_callbacks) {
      auto ptr = std::make_shared<Literal>(flushed.Clone());
      cb.callback(0, 0, {ptr});
    }
    return flushed;
  };
  int mismatch_reports = 0;
  std::string reported_message;
  options.on_mismatch_fn = [&](const HloModule& m, const Literal& test_output,
                               const Literal& ref_output,
                               const absl::Status& status) {
    ++mismatch_reports;
    reported_message = std::string(status.message());
  };

  ::testing::TestPartResultArray failures;
  HloIsolationTestResult result;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS,
        &failures);
    ASSERT_OK_AND_ASSIGN(
        result, RunIsolationTestOnModule(
                    *module, nullptr, FakeReferenceRunner(), options, inputs));
  }
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  EXPECT_EQ(mismatch_reports, 1);
  EXPECT_TRUE(
      absl::StrContains(reported_message, "SUBNORMAL FLUSH-TO-ZERO DETECTED"));
  EXPECT_EQ(failures.size(), 1);

  // The interpreter check and the fusion debugger check both record only the
  // two flushed elements, not the rounding noise on the normal elements. The
  // defused device run flushed the same elements, so that check passes.
  int flush_checks = 0;
  for (const NumericCheck& check : result.numeric_checks()) {
    if (check.name() == "TPU_VS_DEFUSED_TPU") {
      EXPECT_EQ(check.top_mismatches_size(), 0);
      continue;
    }
    ++flush_checks;
    ASSERT_EQ(check.top_mismatches_size(), 1) << check.name();
    const NumericMismatch& mismatch = check.top_mismatches(0);
    EXPECT_DOUBLE_EQ(mismatch.expected(), subnormal_f32) << check.name();
    EXPECT_DOUBLE_EQ(mismatch.actual(), 0.0) << check.name();
    EXPECT_EQ(mismatch.mismatch_count(), 2) << check.name();
    EXPECT_EQ(mismatch.total_elements(), 4) << check.name();
    EXPECT_DOUBLE_EQ(mismatch.percentage_of_elems_exceeding_both_errors(), 50.0)
        << check.name();
    ASSERT_EQ(mismatch.top_mismatch_index_size(), 1) << check.name();
    EXPECT_EQ(mismatch.top_mismatch_index(0), 0) << check.name();
    ASSERT_EQ(mismatch.mismatch_box_min_size(), 1) << check.name();
    EXPECT_EQ(mismatch.mismatch_box_min(0), 0) << check.name();
    EXPECT_EQ(mismatch.mismatch_box_max(0), 2) << check.name();
  }
  EXPECT_EQ(flush_checks, 2);
}

TEST(HloIsolationApiTest, SubnormalInputsStillSucceedWhenReferenceIsSkipped) {
  const absl::string_view kHlo = R"hlo(
HloModule subnormal_inputs_no_reference
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT copy = f32[4] copy(p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(LiteralUtil::CreateR1<float>(
      {subnormal_f32, subnormal_f32, subnormal_f32, subnormal_f32}));

  ModuleIsolationOptions options;
  // The size gate removes the reference runner, so a matching defused run is
  // the only evidence available and the module must be accepted as before.
  options.max_module_size_bytes = 1;
  options.estimate_module_size_fn = [](const HloModule&) { return 2; };
  int reference_runs = 0;
  options.run_module_fn =
      [&reference_runs](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      ++reference_runs;
    }
    return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
  };
  int mismatch_reports = 0;
  options.on_mismatch_fn = [&](const HloModule&, const Literal&, const Literal&,
                               const absl::Status&) { ++mismatch_reports; };

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*module, nullptr, FakeReferenceRunner(), options,
                               inputs));
  EXPECT_EQ(result.state(), State::SUCCESS);
  EXPECT_EQ(result.reason(), "STAGE_1_DEFUSED_TPU_SUCCESS");
  EXPECT_EQ(reference_runs, 0);
  EXPECT_EQ(mismatch_reports, 0);
}

TEST(HloIsolationApiTest, SubnormalFlushRespectsNonDefaultLayouts) {
  const absl::string_view kHlo = R"hlo(
HloModule subnormal_layout_test
ENTRY main {
  p0 = f32[2,2]{0,1} parameter(0)
  ROOT copy = f32[2,2]{0,1} copy(p0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  Literal input = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0f, subnormal_f32}, {1.0f, 1.0f}}, LayoutUtil::MakeLayout({0, 1}));
  std::vector<Literal> inputs;
  inputs.push_back(std::move(input));

  ModuleIsolationOptions options;
  options.run_module_fn =
      [subnormal_f32](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      return LiteralUtil::CreateR2WithLayout<float>(
          {{2.0f, subnormal_f32 * 2.0f}, {2.0f, 2.0f}},
          LayoutUtil::MakeLayout({0, 1}));
    }
    return LiteralUtil::CreateR2WithLayout<float>(
        {{2.0f, 0.0f}, {2.0f, 2.0f}}, LayoutUtil::MakeLayout({1, 0}));
  };
  options.on_mismatch_fn = [](const HloModule&, const Literal&, const Literal&,
                              const absl::Status&) {};

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*module, nullptr, FakeReferenceRunner(), options,
                               inputs));
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");

  bool found_interp_check = false;
  for (const NumericCheck& check : result.numeric_checks()) {
    if (check.name() != "TPU_VS_INTERPRETER") {
      continue;
    }
    found_interp_check = true;
    ASSERT_EQ(check.top_mismatches_size(), 1);
    const NumericMismatch& mismatch = check.top_mismatches(0);
    EXPECT_EQ(mismatch.mismatch_count(), 1);
    EXPECT_EQ(mismatch.total_elements(), 4);
    ASSERT_EQ(mismatch.top_mismatch_index_size(), 2);
    EXPECT_EQ(mismatch.top_mismatch_index(0), 0);
    EXPECT_EQ(mismatch.top_mismatch_index(1), 1);
    ASSERT_EQ(mismatch.mismatch_box_min_size(), 2);
    EXPECT_EQ(mismatch.mismatch_box_min(0), 0);
    EXPECT_EQ(mismatch.mismatch_box_min(1), 1);
    EXPECT_EQ(mismatch.mismatch_box_max(0), 0);
    EXPECT_EQ(mismatch.mismatch_box_max(1), 1);
  }
  EXPECT_TRUE(found_interp_check);
}

TEST(HloIsolationApiTest,
     ExtractMismatchDetailsPropagatesVerifiedAndNestedOps) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule m
nested_good_comp {
  np0 = f32[4] parameter(0)
  ROOT nested_good_op = f32[4] abs(np0)
}
nested_bad_comp {
  np1 = f32[4] parameter(0)
  ROOT nested_bad_op = f32[4] sine(np1)
}
nested_unverified_comp {
  np2 = f32[4] parameter(0)
  ROOT nested_unverified_op = f32[4] negate(np2)
}
fused_comp {
  fp = f32[4] parameter(0)
  nested_good_fus = f32[4] fusion(fp), kind=kLoop, calls=nested_good_comp
  nested_bad_fus = f32[4] fusion(nested_good_fus), kind=kLoop, calls=nested_bad_comp
  nested_unverified_fus = f32[4] fusion(fp), kind=kLoop, calls=nested_unverified_comp
  ROOT fused_mul = f32[4] multiply(nested_bad_fus, nested_unverified_fus)
}
ENTRY main {
  p0 = f32[4] parameter(0)
  good_entry = f32[4] abs(p0)
  ROOT fus = f32[4] fusion(good_entry), kind=kLoop, calls=fused_comp
}
)hlo"));

  HloIsolationTestResult result;
  NumericCheck* parent = result.add_numeric_checks();
  parent->set_name("TPU_VS_INTERPRETER");
  parent->add_top_mismatches()->set_rel_error(1.0);
  NumericCheck* sub = result.add_numeric_checks();
  sub->set_name("FusionDebugger:nested_bad_op");
  sub->add_top_mismatches()->set_rel_error(1.0);

  for (absl::string_view op : {"good_entry", "nested_good_op", "nested_good_op",
                               "fused_mul", "nested_bad_op", "unknown"}) {
    result.add_verified_instructions(op);
  }

  numerics::debug_info::GraphData graph =
      numerics::debug_info::PopulateMismatchGraphData(
          *module, ExtractMismatchDetails(*module, result),
          numerics::debug_info::kUnverifiedDiffScore);
  absl::flat_hash_map<std::string, double> scores;
  for (const auto& node : graph.nodes) {
    scores[node.key] = node.diff_score;
  }

  const double kGray = numerics::debug_info::kUnverifiedDiffScore;
  EXPECT_DOUBLE_EQ(scores["p0"], kGray);
  EXPECT_DOUBLE_EQ(scores["fus/fp"], kGray);
  EXPECT_DOUBLE_EQ(scores["fus/nested_unverified_fus/nested_unverified_op"],
                   kGray);
  EXPECT_DOUBLE_EQ(scores["fus/nested_unverified_fus"], kGray);
  EXPECT_DOUBLE_EQ(scores["good_entry"], 0.0);
  EXPECT_DOUBLE_EQ(scores["fus/nested_good_fus/nested_good_op"], 0.0);
  EXPECT_DOUBLE_EQ(scores["fus/nested_good_fus"], 0.0);
  EXPECT_DOUBLE_EQ(scores["fus/fused_mul"], 0.0);
  EXPECT_DOUBLE_EQ(scores["fus/nested_bad_fus/nested_bad_op"], 100.0);
  EXPECT_DOUBLE_EQ(scores["fus/nested_bad_fus"], 100.0);
  EXPECT_DOUBLE_EQ(scores["fus"], 100.0);
}

// A main thread offload to SparseCore of a fusion.
constexpr absl::string_view kSparseCoreHlo = R"hlo(
HloModule sparse_core

fused {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0)
}, execution_thread="sparsecore"

wrapped {
  p0 = f32[4] parameter(0)
  ROOT fusion = f32[4] fusion(p0), kind=kLoop, calls=fused
}, execution_thread="sparsecore"

ENTRY main {
  p0 = f32[4] parameter(0)
  start = ((f32[4]), f32[4]) async-start(p0), async_execution_thread="sparsecore", calls=wrapped
  ROOT done = f32[4] async-done(start)
}
)hlo";

// Runs every module to `output`, failing the test if a defused module is
// built. Counts runs that use the fusion debugger in `fusion_debugger_runs`.
decltype(ModuleIsolationOptions::run_module_fn) RunWithoutDefusing(
    Literal output, int* fusion_debugger_runs = nullptr) {
  auto shared_output = std::make_shared<Literal>(std::move(output));
  return [shared_output, fusion_debugger_runs](
             std::unique_ptr<HloModule> m, HloRunnerInterface* r,
             absl::Span<const Literal> i,
             const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    EXPECT_FALSE(absl::StrContains(m->name(), "defused"));
    if (fusion_debugger_runs != nullptr) {
      *fusion_debugger_runs += run_opts.use_fusion_debugger;
    }
    if (r == FakeReferenceRunner()) {
      return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
    }
    return shared_output->Clone();
  };
}

TEST(HloIsolationApiTest, IsSparseCoreModuleLooksForMainThreadOffloads) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> offload,
                       xla::ParseAndReturnUnverifiedModule(kSparseCoreHlo));
  EXPECT_TRUE(IsSparseCoreModule(*offload));

  // An offload to another thread, and a SparseCore chain that is only
  // started from another thread, do not count.
  constexpr absl::string_view kHlo = R"hlo(
HloModule m

inner {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0)
}, execution_thread="sparsecore"

wrapped {
  p0 = f32[4] parameter(0)
  start = ((f32[4]), f32[4]) async-start(p0), async_execution_thread="sparsecore", calls=inner
  ROOT done = f32[4] async-done(start)
}, execution_thread="other"

ENTRY main {
  p0 = f32[4] parameter(0)
  start = ((f32[4]), f32[4]) async-start(p0), async_execution_thread="other", calls=wrapped
  ROOT done = f32[4] async-done(start)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> other,
                       xla::ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsSparseCoreModule(*other));
}

TEST(HloIsolationApiTest, SparseCoreModuleUsesInterpreterReference) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kSparseCoreHlo));
  ModuleIsolationOptions options;
  options.run_module_fn = RunWithoutDefusing(
      LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f}));

  ASSERT_OK_AND_ASSIGN(HloIsolationTestResult result,
                       RunIsolationTestOnModule(
                           *module, nullptr, FakeReferenceRunner(), options));
  EXPECT_EQ(result.state(), State::SUCCESS);
  EXPECT_EQ(result.reason(), "STAGE_2_INTERPRETER_SUCCESS");
}

TEST(HloIsolationApiTest, SparseCoreModuleWithoutReferenceIsSkipped) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kSparseCoreHlo));
  ModuleIsolationOptions options;
  options.run_module_fn = RunWithoutDefusing(
      LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f}));

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*module, nullptr, nullptr, options));
  EXPECT_EQ(result.state(), State::SKIPPED);
  EXPECT_EQ(result.reason(), "REFERENCE_UNAVAILABLE");
}

// TODO(b/527142533): Enable & test fusion debugger for SparseCore modules.
TEST(HloIsolationApiTest, SparseCoreMismatchIsReportedWithoutFusionDebugger) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kSparseCoreHlo));
  int fusion_debugger_runs = 0;
  ModuleIsolationOptions options;
  options.run_module_fn =
      RunWithoutDefusing(LiteralUtil::CreateR1<float>({1.0f, 1.0f, 1.0f, 1.0f}),
                         &fusion_debugger_runs);
  int mismatch_reports = 0;
  options.on_mismatch_fn = [&](const HloModule&, const Literal&, const Literal&,
                               const absl::Status&) { ++mismatch_reports; };

  ASSERT_OK_AND_ASSIGN(HloIsolationTestResult result,
                       RunIsolationTestOnModule(
                           *module, nullptr, FakeReferenceRunner(), options));
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  EXPECT_EQ(mismatch_reports, 1);
  EXPECT_EQ(fusion_debugger_runs, 0);

  std::string outputs_dir;
  ASSERT_TRUE(tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir));
  std::string html;
  ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(outputs_dir, "failed-module-sparse_core.html"), &html));
  EXPECT_TRUE(absl::StrContains(html, "window.compressedGraphData"));
  EXPECT_TRUE(absl::StrContains(html, "Numeric Mismatch:"));
  EXPECT_TRUE(absl::StrContains(html, "done"));
}

TEST(HloIsolationApiTest, SparseCoreUnimplementedReferenceIsSkipped) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kSparseCoreHlo));
  for (const absl::Status& err :
       {absl::UnimplementedError("Unsupported custom call"),
        absl::NotFoundError("Custom call target was not registered")}) {
    ModuleIsolationOptions options;
    options.run_module_fn =
        [err](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
              absl::Span<const Literal>,
              const RunModuleOptions&) -> absl::StatusOr<Literal> {
      if (r == FakeReferenceRunner()) {
        return err;
      }
      return LiteralUtil::CreateR1<float>({0.0f, 0.0f, 0.0f, 0.0f});
    };

    ASSERT_OK_AND_ASSIGN(HloIsolationTestResult result,
                         RunIsolationTestOnModule(
                             *module, nullptr, FakeReferenceRunner(), options));
    EXPECT_EQ(result.state(), State::SKIPPED);
    EXPECT_EQ(result.reason(), "REFERENCE_UNIMPLEMENTED");
  }
}

TEST(HloIsolationApiTest, SparseCoreOffloadSkipReasonChecksWorkRngAndMosaic) {
  auto skip_reason =
      [](absl::string_view body) -> absl::StatusOr<std::optional<std::string>> {
    std::string hlo = absl::StrCat(R"hlo(
HloModule m

wrapped {
  x = f32[8] parameter(0)
)hlo",
                                   body, R"hlo(
}, execution_thread="sparsecore"

ENTRY main {
  p = f32[8] parameter(0)
  start = ((f32[8]), f32[8]) async-start(p), async_execution_thread="sparsecore", calls=wrapped
  ROOT done = f32[8] async-done(start)
}
)hlo");
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                     xla::ParseAndReturnUnverifiedModule(hlo));
    return SparseCoreOffloadSkipReason(*module);
  };

  ASSERT_OK_AND_ASSIGN(std::optional<std::string> ok_reason,
                       skip_reason("  ROOT neg = f32[8] negate(x)"));
  EXPECT_EQ(ok_reason, std::nullopt);

  ASSERT_OK_AND_ASSIGN(std::optional<std::string> no_work,
                       skip_reason("  ROOT b = f32[8] bitcast(x)"));
  EXPECT_EQ(no_work, "NO_WORK_IN_SPARSECORE_OFFLOAD");

  ASSERT_OK_AND_ASSIGN(
      std::optional<std::string> mosaic,
      skip_reason(
          R"(  ROOT k = f32[8] custom-call(x), custom_call_target="tpu_custom_call")"));
  EXPECT_EQ(mosaic, "CONTAINS_MOSAIC_KERNEL");

  ASSERT_OK_AND_ASSIGN(std::optional<std::string> rng, skip_reason(R"(
  zero = f32[] constant(0)
  one = f32[] constant(1)
  r = f32[8] rng(zero, one), distribution=rng_uniform
  ROOT add = f32[8] add(x, r))"));
  EXPECT_EQ(rng, "CONTAINS_RNG");
}

TEST(HloIsolationApiTest, SubnormalFlushIgnoredOnComputeOutputChain) {
  constexpr absl::string_view kComputeHlo = R"hlo(
HloModule subnormal_flush_compute_op
ENTRY main {
  p0 = f32[4] parameter(0)
  add = f32[4] add(p0, p0)
  ROOT out = f32[4] copy(add)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> compute_module,
                       xla::ParseAndReturnUnverifiedModule(kComputeHlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(
      LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f, subnormal_f32, 1.0f}));

  ModuleIsolationOptions options;
  options.run_module_fn =
      [subnormal_f32](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      const float expected = subnormal_f32 * 2.0f;
      return LiteralUtil::CreateR1<float>({expected, 2.0f, expected, 2.0f});
    }
    if (absl::StrContains(m->name(), "defused")) {
      return LiteralUtil::CreateR1<float>({0.0f, 3.0f, 0.0f, 2.0f});
    }
    return LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f});
  };
  int mismatch_reports = 0;
  options.on_mismatch_fn = [&](const HloModule&, const Literal&, const Literal&,
                               const absl::Status&) { ++mismatch_reports; };

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*compute_module, nullptr, FakeReferenceRunner(),
                               options, inputs));
  EXPECT_EQ(result.state(), State::SUCCESS);
  EXPECT_EQ(result.reason(), "STAGE_2_INTERPRETER_SUCCESS");
  EXPECT_EQ(mismatch_reports, 0);
}

TEST(HloIsolationApiTest, SubnormalFlushScopedToBitPreservingOutputLeaves) {
  constexpr absl::string_view kMixedFusionHlo = R"hlo(
HloModule mixed_multi_output_fusion
fused_comp {
  p = f32[4] parameter(0)
  c = f32[4] copy(p)
  r = f32[4] reshape(c)
  a = f32[4] add(p, p)
  ROOT t = (f32[4], f32[4]) tuple(r, a)
}
ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT f = (f32[4], f32[4]) fusion(p0), kind=kLoop, calls=fused_comp
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> mixed_module,
                       xla::ParseAndReturnUnverifiedModule(kMixedFusionHlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(
      LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f, subnormal_f32, 1.0f}));

  ModuleIsolationOptions options;
  options.run_module_fn =
      [subnormal_f32](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      Literal leaf0 = LiteralUtil::CreateR1<float>(
          {subnormal_f32, 1.0f, subnormal_f32, 1.0f});
      Literal leaf1 = LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f});
      return LiteralUtil::MakeTuple({&leaf0, &leaf1});
    }
    Literal leaf0 = LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});
    Literal leaf1 = LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f});
    return LiteralUtil::MakeTuple({&leaf0, &leaf1});
  };
  int mismatch_reports = 0;
  options.on_mismatch_fn = [&](const HloModule&, const Literal&, const Literal&,
                               const absl::Status&) { ++mismatch_reports; };

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*mixed_module, nullptr, FakeReferenceRunner(),
                               options, inputs));
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  EXPECT_EQ(mismatch_reports, 1);
}

TEST(HloIsolationApiTest, SubnormalFlushTrappedThroughTupleOptBarrier) {
  constexpr absl::string_view kOptBarrierHlo = R"hlo(
HloModule opt_barrier_tuple
ENTRY main {
  p0 = f32[4] parameter(0)
  a = f32[4] add(p0, p0)
  t = (f32[4], f32[4]) tuple(p0, a)
  ob = (f32[4], f32[4]) opt-barrier(t)
  ROOT gte = f32[4] get-tuple-element(ob), index=0
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> opt_barrier_module,
                       xla::ParseAndReturnUnverifiedModule(kOptBarrierHlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(
      LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f, subnormal_f32, 1.0f}));

  ModuleIsolationOptions options;
  options.enable_fusion_debugger = false;
  options.run_module_fn =
      [subnormal_f32](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      return LiteralUtil::CreateR1<float>(
          {subnormal_f32, 1.0f, subnormal_f32, 1.0f});
    }
    return LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});
  };
  int mismatch_reports = 0;
  options.on_mismatch_fn = [&](const HloModule&, const Literal&, const Literal&,
                               const absl::Status&) { ++mismatch_reports; };

  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*opt_barrier_module, nullptr,
                               FakeReferenceRunner(), options, inputs));
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  EXPECT_GE(mismatch_reports, 1);
}

TEST(HloIsolationApiTest,
     SubnormalFlushIgnoredOnSharedCalleeAndIntegralBitcastChains) {
  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(
      LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f, subnormal_f32, 1.0f}));

  for (absl::string_view hlo : {
           absl::string_view(R"hlo(
HloModule shared_callee
%callee (x: f32[4]) -> f32[4] {
  %x = f32[4] parameter(0)
  ROOT %copy = f32[4] copy(%x)
}
ENTRY main {
  %p0 = f32[4] parameter(0)
  %c1 = f32[4] call(%p0), to_apply=%callee
  %a = f32[4] add(%p0, %p0)
  %c2 = f32[4] call(%a), to_apply=%callee
  ROOT %out = f32[8] concatenate(%c1, %c2), dimensions={0}
}
)hlo"),
           absl::string_view(R"hlo(
HloModule bitcast_convert_through_u32
ENTRY main {
  %p0 = f32[4] parameter(0)
  %a = f32[4] add(%p0, %p0)
  %as_u32 = u32[4] bitcast-convert(%a)
  ROOT %out = f32[4] bitcast-convert(%as_u32)
}
)hlo"),
       }) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> mod,
                         xla::ParseAndReturnUnverifiedModule(hlo));
    const bool is_concat = mod->name() == "shared_callee";
    ModuleIsolationOptions options;
    options.enable_fusion_debugger = false;
    options.run_module_fn =
        [subnormal_f32, is_concat](
            std::unique_ptr<HloModule> m, HloRunnerInterface* r,
            absl::Span<const Literal> i,
            const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
      const float two_sub = subnormal_f32 * 2.0f;
      if (is_concat) {
        if (r == FakeReferenceRunner()) {
          return LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f,
                                               subnormal_f32, 1.0f, two_sub,
                                               2.0f, two_sub, 2.0f});
        }
        if (absl::StrContains(m->name(), "defused")) {
          return LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f,
                                               subnormal_f32, 1.0f, 0.0f, 3.0f,
                                               0.0f, 2.0f});
        }
        return LiteralUtil::CreateR1<float>(
            {subnormal_f32, 1.0f, subnormal_f32, 1.0f, 0.0f, 2.0f, 0.0f, 2.0f});
      }
      if (r == FakeReferenceRunner()) {
        return LiteralUtil::CreateR1<float>({two_sub, 2.0f, two_sub, 2.0f});
      }
      if (absl::StrContains(m->name(), "defused")) {
        return LiteralUtil::CreateR1<float>({0.0f, 3.0f, 0.0f, 2.0f});
      }
      return LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f});
    };
    int mismatch_reports = 0;
    options.on_mismatch_fn = [&](const HloModule&, const Literal&,
                                 const Literal&,
                                 const absl::Status&) { ++mismatch_reports; };
    ASSERT_OK_AND_ASSIGN(
        HloIsolationTestResult result,
        RunIsolationTestOnModule(*mod, nullptr, FakeReferenceRunner(), options,
                                 inputs));
    EXPECT_EQ(result.state(), State::SUCCESS) << mod->name();
    EXPECT_EQ(result.reason(), "STAGE_2_INTERPRETER_SUCCESS") << mod->name();
    EXPECT_EQ(mismatch_reports, 0) << mod->name();
  }
}

struct CallbackCheckResult {
  int failure_count = 0;
  HloIsolationTestResult test_result;
};

CallbackCheckResult RunComparisonCallbackCheck(
    const HloModule& module, absl::string_view expected_op_name,
    const Literal& expected, const Literal& actual,
    const absl::flat_hash_map<GroupKey, std::vector<std::string>>& ref_groups =
        {}) {
  std::unique_ptr<HloModule> clone = module.Clone();
  auto expected_literals = std::make_shared<ExpectedLiteralsMap>();
  (*expected_literals)[std::string(expected_op_name)] =
      std::make_shared<Literal>(expected.Clone());
  auto mutex = std::make_shared<absl::Mutex>();
  CallbackCheckResult check_result;
  std::vector<HloOutputCallback> cbs = CreateComparisonHloOutputCallbacks(
      clone.get(), ref_groups, expected_literals, module,
      ModuleIsolationOptions(), mutex, &check_result.test_result);
  auto actual_ptr = std::make_shared<Literal>(actual.Clone());
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS,
        &failures);
    for (const HloOutputCallback& cb : cbs) {
      cb.callback(0, 0, {actual_ptr});
    }
  }
  check_result.failure_count = failures.size();
  return check_result;
}

TEST(HloIsolationApiTest, FusionDebuggerCallbackIgnoresComputeFedCopy) {
  constexpr absl::string_view kFusionCopyOfAddHlo = R"hlo(
HloModule fusion_copy_of_add
%fused_comp (p: f32[4]) -> f32[4] {
  %p = f32[4] parameter(0)
  %a = f32[4] add(%p, %p)
  ROOT %copy = f32[4] copy(%a)
}
ENTRY main {
  %p0 = f32[4] parameter(0)
  ROOT %f = f32[4] fusion(%p0), kind=kLoop, calls=%fused_comp
}
)hlo";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> fusion_copy_module,
      xla::ParseAndReturnUnverifiedModule(kFusionCopyOfAddHlo));
  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  const CallbackCheckResult check = RunComparisonCallbackCheck(
      *fusion_copy_module, "copy",
      LiteralUtil::CreateR1<float>(
          {subnormal_f32 * 2.0f, 2.0f, subnormal_f32 * 2.0f, 2.0f}),
      LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f}),
      {{{HloOpcode::kCopy, "f32[4]"}, {"copy"}}});
  EXPECT_EQ(check.failure_count, 0);
  EXPECT_EQ(check.test_result.numeric_checks_size(), 0);
}

// A chain of 40 `select` diamonds inside a fusion must be walked in linear
// time; without memoization under a non-empty call stack this never finishes.
TEST(HloIsolationApiTest, BitPreservingWalkIsLinearOnFusionDiamonds) {
  std::string body =
      "  %p = f32[4] parameter(0)\n"
      "  %cond = pred[4] parameter(1)\n"
      "  %sel0 = f32[4] select(%cond, %p, %p)\n";
  for (int i = 1; i <= 40; ++i) {
    absl::StrAppend(&body, i == 40 ? "  ROOT " : "  ", "%sel", i,
                    " = f32[4] select(%cond, %sel", i - 1, ", %sel", i - 1,
                    ")\n");
  }
  const std::string hlo = absl::StrCat(
      "HloModule diamond_fusion_dag\n"
      "%fused_comp (p: f32[4], cond: pred[4]) -> f32[4] {\n",
      body,
      "}\n"
      "ENTRY main {\n"
      "  %p0 = f32[4] parameter(0)\n"
      "  %cond0 = pred[4] parameter(1)\n"
      "  ROOT %f = f32[4] fusion(%p0, %cond0), kind=kLoop, "
      "calls=%fused_comp\n"
      "}\n");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(hlo));

  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(
      LiteralUtil::CreateR1<float>({subnormal_f32, 1.0f, subnormal_f32, 1.0f}));
  inputs.push_back(LiteralUtil::CreateR1<bool>({true, false, true, false}));

  ModuleIsolationOptions options;
  options.enable_fusion_debugger = false;
  options.run_module_fn =
      [subnormal_f32](
          std::unique_ptr<HloModule> m, HloRunnerInterface* r,
          absl::Span<const Literal> i,
          const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    if (r == FakeReferenceRunner()) {
      return LiteralUtil::CreateR1<float>(
          {subnormal_f32, 1.0f, subnormal_f32, 1.0f});
    }
    return LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});
  };
  int mismatch_reports = 0;
  options.on_mismatch_fn = [&](const HloModule&, const Literal&, const Literal&,
                               const absl::Status&) { ++mismatch_reports; };
  // The output is bit preserving, so the flushed subnormal is still trapped.
  ASSERT_OK_AND_ASSIGN(
      HloIsolationTestResult result,
      RunIsolationTestOnModule(*module, nullptr, FakeReferenceRunner(), options,
                               inputs));
  EXPECT_EQ(result.state(), State::FAILURE);
  EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  EXPECT_GE(mismatch_reports, 1);
}

// A fused `copy` inside a `call` callee whose argument is `add` is not bit
// preserving, so a flushed subnormal on it is not trapped. The same callee fed
// by an entry parameter is bit preserving and still traps.
TEST(HloIsolationApiTest, FusionDebuggerTracesCalleeParametersToCallers) {
  constexpr absl::string_view kHloTemplate = R"hlo(
HloModule call_with_fusion_copy
%fused_comp (p: f32[4]) -> f32[4] {
  %p = f32[4] parameter(0)
  ROOT %copy = f32[4] copy(%p)
}
%callee (y: f32[4]) -> f32[4] {
  %y = f32[4] parameter(0)
  ROOT %f = f32[4] fusion(%y), kind=kLoop, calls=%fused_comp
}
ENTRY main {
  %p0 = f32[4] parameter(0)
  %a = f32[4] add(%p0, %p0)
  ROOT %c = f32[4] call($0), to_apply=%callee
}
)hlo";
  const float subnormal_f32 = std::numeric_limits<float>::denorm_min();
  for (const auto& [arg, expect_trap] :
       std::vector<std::pair<std::string, bool>>{{"%a", false},
                                                 {"%p0", true}}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(
                             absl::Substitute(kHloTemplate, arg)));
    const CallbackCheckResult check = RunComparisonCallbackCheck(
        *module, "copy",
        LiteralUtil::CreateR1<float>(
            {subnormal_f32, 2.0f, subnormal_f32, 2.0f}),
        LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f}),
        {{{HloOpcode::kCopy, "f32[4]"}, {"copy"}}});
    EXPECT_EQ(check.failure_count, expect_trap ? 1 : 0) << arg;
    EXPECT_EQ(check.test_result.numeric_checks_size(), expect_trap ? 1 : 0)
        << arg;
  }
}

TEST(HloIsolationApiTest, AsyncStartAndAsyncDoneScopeSubnormalFlushByLeaf) {
  constexpr absl::string_view kHlo = R"hlo(
HloModule async_mixed_tuple
%wrapped (p: f32[4]) -> (f32[4], f32[4]) {
  %p = f32[4] parameter(0)
  %a = f32[4] add(%p, %p)
  %c = f32[4] copy(%p)
  ROOT %t = (f32[4], f32[4]) tuple(%a, %c)
}
ENTRY main {
  %p0 = f32[4] parameter(0)
  %start = ((f32[4]), (f32[4], f32[4])) async-start(%p0), calls=%wrapped
  ROOT %done = (f32[4], f32[4]) async-done(%start)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(kHlo));
  const float sub = std::numeric_limits<float>::denorm_min();
  const Literal sub_lit = LiteralUtil::CreateR1<float>({sub, 1.0f, sub, 1.0f});
  const Literal zero_lit =
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});

  auto run_callback_check = [&](absl::string_view op_name,
                                const Literal& expected,
                                const Literal& actual) -> int {
    const CallbackCheckResult check =
        RunComparisonCallbackCheck(*module, op_name, expected, actual);
    EXPECT_EQ(check.failure_count, check.test_result.numeric_checks_size())
        << op_name;
    return check.failure_count;
  };

  auto make_start_literal = [](const Literal& input_leaf,
                               const Literal& out_add,
                               const Literal& out_copy) {
    Literal inputs_tuple = LiteralUtil::MakeTuple({&input_leaf});
    Literal outputs_tuple = LiteralUtil::MakeTuple({&out_add, &out_copy});
    return LiteralUtil::MakeTuple({&inputs_tuple, &outputs_tuple});
  };
  const Literal expected_start = make_start_literal(sub_lit, sub_lit, sub_lit);
  // Leaf {1, 0} (`add`) is not bit preserving and must not trap.
  EXPECT_EQ(run_callback_check("start", expected_start,
                               make_start_literal(sub_lit, zero_lit, sub_lit)),
            0);
  // Leaf {1, 1} (`copy`) is bit preserving and must trap.
  EXPECT_EQ(run_callback_check("start", expected_start,
                               make_start_literal(sub_lit, sub_lit, zero_lit)),
            1);
  // Leaf {0, 0} (passthrough entry parameter) is bit preserving and must trap.
  EXPECT_EQ(run_callback_check("start", expected_start,
                               make_start_literal(zero_lit, sub_lit, sub_lit)),
            1);

  const Literal expected_done = LiteralUtil::MakeTuple({&sub_lit, &sub_lit});
  // Leaf {0} (`add`) on `async-done` must not trap; leaf {1} (`copy`) must.
  EXPECT_EQ(run_callback_check("done", expected_done,
                               LiteralUtil::MakeTuple({&zero_lit, &sub_lit})),
            0);
  EXPECT_EQ(run_callback_check("done", expected_done,
                               LiteralUtil::MakeTuple({&sub_lit, &zero_lit})),
            1);
}

// A chain of 35 two-call diamond computations must be analyzed in linear time
// both top down and bottom up when inspecting an inner `copy`.
TEST(HloIsolationApiTest, BitPreservingWalkIsLinearOnCallDiamonds) {
  std::string hlo =
      "HloModule call_diamond_dag\n"
      "%c0 (p: f32[4], cond: pred[4]) -> f32[4] {\n"
      "  %p = f32[4] parameter(0)\n"
      "  %cond = pred[4] parameter(1)\n"
      "  ROOT %copy = f32[4] copy(%p)\n"
      "}\n";
  for (int i = 1; i <= 35; ++i) {
    absl::StrAppend(&hlo, "%c", i,
                    " (p: f32[4], cond: pred[4]) -> f32[4] {\n"
                    "  %p = f32[4] parameter(0)\n"
                    "  %cond = pred[4] parameter(1)\n"
                    "  %a = f32[4] call(%p, %cond), to_apply=%c",
                    i - 1,
                    "\n"
                    "  %b = f32[4] call(%p, %cond), to_apply=%c",
                    i - 1,
                    "\n"
                    "  ROOT %sel = f32[4] select(%cond, %a, %b)\n"
                    "}\n");
  }
  absl::StrAppend(&hlo,
                  "ENTRY main {\n"
                  "  %p0 = f32[4] parameter(0)\n"
                  "  %cond0 = pred[4] parameter(1)\n"
                  "  %a = f32[4] call(%p0, %cond0), to_apply=%c35\n"
                  "  %b = f32[4] call(%p0, %cond0), to_apply=%c35\n"
                  "  ROOT %sel = f32[4] select(%cond0, %a, %b)\n"
                  "}\n");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       xla::ParseAndReturnUnverifiedModule(hlo));

  const float sub = std::numeric_limits<float>::denorm_min();
  const CallbackCheckResult check = RunComparisonCallbackCheck(
      *module, "copy", LiteralUtil::CreateR1<float>({sub, 1.0f, sub, 1.0f}),
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f}));
  EXPECT_EQ(check.failure_count, 1);
  EXPECT_EQ(check.test_result.numeric_checks_size(), 1);
}

TEST(HloIsolationApiTest,
     Stage1SubnormalEscalationAndTupleTokenForwardingAreScopedToFloatLeaves) {
  const float sub = std::numeric_limits<float>::denorm_min();
  std::vector<Literal> inputs;
  inputs.push_back(LiteralUtil::CreateR1<float>({sub, 1.0f, sub, 1.0f}));

  // 1. Bare entry parameter output escalates past stage 1.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule bare_param
ENTRY main {
  ROOT p0 = f32[4] parameter(0)
}
)hlo"));
    int reference_runs = 0;
    ModuleIsolationOptions options;
    options.enable_fusion_debugger = false;
    options.run_module_fn =
        [&](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
            absl::Span<const Literal> i,
            const RunModuleOptions&) -> absl::StatusOr<Literal> {
      if (r == FakeReferenceRunner()) {
        ++reference_runs;
        return LiteralUtil::CreateR1<float>({sub, 1.0f, sub, 1.0f});
      }
      return LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});
    };
    options.on_mismatch_fn = [](const HloModule&, const Literal&,
                                const Literal&, const absl::Status&) {};
    ASSERT_OK_AND_ASSIGN(
        HloIsolationTestResult result,
        RunIsolationTestOnModule(*module, nullptr, FakeReferenceRunner(),
                                 options, inputs));
    EXPECT_EQ(reference_runs, 1);
    EXPECT_EQ(result.state(), State::FAILURE);
    EXPECT_EQ(result.reason(), "NUMERIC_MISMATCH");
  }

  // 2. Pure compute multi output fusion short circuits at stage 1.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule pure_compute_mof
%fused_comp (p: f32[4]) -> (f32[4], f32[4]) {
  %p = f32[4] parameter(0)
  %a = f32[4] add(%p, %p)
  %m = f32[4] multiply(%p, %p)
  ROOT %t = (f32[4], f32[4]) tuple(%a, %m)
}
ENTRY main {
  %p0 = f32[4] parameter(0)
  ROOT %f = (f32[4], f32[4]) fusion(%p0), kind=kLoop, calls=%fused_comp
}
)hlo"));
    int reference_runs = 0;
    ModuleIsolationOptions options;
    options.enable_fusion_debugger = false;
    options.run_module_fn =
        [&](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
            absl::Span<const Literal> i,
            const RunModuleOptions&) -> absl::StatusOr<Literal> {
      if (r == FakeReferenceRunner()) {
        ++reference_runs;
      }
      Literal leaf = LiteralUtil::CreateR1<float>({0.0f, 2.0f, 0.0f, 2.0f});
      return LiteralUtil::MakeTuple({&leaf, &leaf});
    };
    ASSERT_OK_AND_ASSIGN(
        HloIsolationTestResult result,
        RunIsolationTestOnModule(*module, nullptr, FakeReferenceRunner(),
                                 options, inputs));
    EXPECT_EQ(reference_runs, 0);
    EXPECT_EQ(result.state(), State::SUCCESS);
    EXPECT_EQ(result.reason(), "STAGE_1_DEFUSED_TPU_SUCCESS");
  }

  // 2b. Float to int bitcast-convert fed by a bit preserving chain escalates
  // past stage 1 even when the module output itself is integral.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule float_to_int_bitcast
ENTRY main {
  %p0 = f32[4] parameter(0)
  %bits = u32[4] bitcast-convert(%p0)
  %one = u32[4] constant({1, 1, 1, 1})
  ROOT %out = u32[4] or(%bits, %one)
}
)hlo"));
    int reference_runs = 0;
    ModuleIsolationOptions options;
    options.enable_fusion_debugger = false;
    options.run_module_fn =
        [&](std::unique_ptr<HloModule> m, HloRunnerInterface* r,
            absl::Span<const Literal> i,
            const RunModuleOptions&) -> absl::StatusOr<Literal> {
      if (r == FakeReferenceRunner()) {
        ++reference_runs;
        return LiteralUtil::CreateR1<uint32_t>(
            {1, 0x3f800001u, 1, 0x3f800001u});
      }
      return LiteralUtil::CreateR1<uint32_t>({1, 0x3f800001u, 1, 0x3f800001u});
    };
    ASSERT_OK_AND_ASSIGN(
        HloIsolationTestResult result,
        RunIsolationTestOnModule(*module, nullptr, FakeReferenceRunner(),
                                 options, inputs));
    EXPECT_EQ(reference_runs, 1);
    EXPECT_EQ(result.state(), State::SUCCESS);
    EXPECT_EQ(result.reason(), "STAGE_2_INTERPRETER_SUCCESS");
  }

  // 3. Tuple copy and token bearing ops forward per leaf bit preservation.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule tuple_copy_and_token
ENTRY main {
  %p0 = f32[4] parameter(0)
  %tok = token[] after-all()
  %sent = (f32[4], u32[], token[]) send(%p0, %tok), channel_id=1
  %a = f32[4] add(%p0, %p0)
  %ag = (f32[4], f32[4]) all-gather(%a, %p0), dimensions={0}
  %g0 = f32[4] get-tuple-element(%ag), index=0
  %g1 = f32[4] get-tuple-element(%ag), index=1
  %s0 = f32[4] get-tuple-element(%sent), index=0
  %t = (f32[4], f32[4], f32[4]) tuple(%g0, %g1, %s0)
  ROOT %c = (f32[4], f32[4], f32[4]) copy(%t)
}
)hlo"));
    const Literal sub_lit =
        LiteralUtil::CreateR1<float>({sub, 1.0f, sub, 1.0f});
    const Literal zero_lit =
        LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});
    auto run_copy_cb = [&](const Literal& l0, const Literal& l1,
                           const Literal& l2) -> int {
      return RunComparisonCallbackCheck(
                 *module, "c",
                 LiteralUtil::MakeTuple({&sub_lit, &sub_lit, &sub_lit}),
                 LiteralUtil::MakeTuple({&l0, &l1, &l2}))
          .failure_count;
    };
    // Leaf 0 comes from all-gather operand 0 (`add`) -> must not trap.
    EXPECT_EQ(run_copy_cb(zero_lit, sub_lit, sub_lit), 0);
    // Leaf 1 comes from all-gather operand 1 (`p0`) -> must trap.
    EXPECT_EQ(run_copy_cb(sub_lit, zero_lit, sub_lit), 1);
    // Leaf 2 comes from `send(p0, after-all)` -> token ignored, must trap.
    EXPECT_EQ(run_copy_cb(sub_lit, sub_lit, zero_lit), 1);
  }
}

TEST(HloIsolationApiTest,
     TupleAllToAllAndInplaceCollectivePermuteScopeSubnormalFlush) {
  const float sub = std::numeric_limits<float>::denorm_min();
  const Literal sub_lit = LiteralUtil::CreateR1<float>({sub, 1.0f, sub, 1.0f});
  const Literal zero_lit =
      LiteralUtil::CreateR1<float>({0.0f, 1.0f, 0.0f, 1.0f});

  // 1. Tuple `all-to-all(add, p0)` exchanges cross-rank operands, so leaf {1}
  // on a non-zero rank can receive `add` and must not trap subnormal flush.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule tuple_all_to_all_mixed
ENTRY main {
  %p0 = f32[4] parameter(0)
  %a = f32[4] add(%p0, %p0)
  ROOT %ata = (f32[4], f32[4]) all-to-all(%a, %p0), replica_groups={{0,1}}
}
)hlo"));
    const CallbackCheckResult check = RunComparisonCallbackCheck(
        *module, "ata", LiteralUtil::MakeTuple({&sub_lit, &sub_lit}),
        LiteralUtil::MakeTuple({&sub_lit, &zero_lit}));
    EXPECT_EQ(check.failure_count, 0);
    EXPECT_EQ(check.test_result.numeric_checks_size(), 0);
  }

  // 2. Tuple `all-to-all(p0, p1)` with all bit-preserving operands traps.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule tuple_all_to_all_pure
ENTRY main {
  %p0 = f32[4] parameter(0)
  %p1 = f32[4] parameter(1)
  ROOT %ata = (f32[4], f32[4]) all-to-all(%p0, %p1), replica_groups={{0,1}}
}
)hlo"));
    const CallbackCheckResult check = RunComparisonCallbackCheck(
        *module, "ata", LiteralUtil::MakeTuple({&sub_lit, &sub_lit}),
        LiteralUtil::MakeTuple({&sub_lit, &zero_lit}));
    EXPECT_EQ(check.failure_count, 1);
    EXPECT_EQ(check.test_result.numeric_checks_size(), 1);
  }

  // 3. In-place `collective-permute` takes (in_buf, out_buf, in_off, out_off).
  // Leaf {1} is bit preserving only when both `in_buf` and `out_buf` are bit
  // preserving at {1}; if `in_buf` has `add` at {1}, leaf {1} must not trap.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule inplace_cp_mixed
ENTRY main {
  %p0 = f32[4] parameter(0)
  %p1 = f32[4] parameter(1)
  %a = f32[4] add(%p0, %p0)
  %in_buf = (f32[4], f32[4]) tuple(%a, %a)
  %out_buf = (f32[4], f32[4]) tuple(%p1, %p1)
  %c0 = s32[] constant(0)
  %off_pair = (s32[]) tuple(%c0)
  %offs = ((s32[]), (s32[])) tuple(%off_pair, %off_pair)
  ROOT %cp = (f32[4], f32[4]) collective-permute(%in_buf, %out_buf, %offs, %offs),
      source_target_pairs={{0,1}}, slice_sizes={{2},{2}}
}
)hlo"));
    const CallbackCheckResult check = RunComparisonCallbackCheck(
        *module, "cp", LiteralUtil::MakeTuple({&sub_lit, &sub_lit}),
        LiteralUtil::MakeTuple({&sub_lit, &zero_lit}));
    EXPECT_EQ(check.failure_count, 0);
    EXPECT_EQ(check.test_result.numeric_checks_size(), 0);
  }

  // 4. In-place `collective-permute` scopes per leaf: leaf {0} (`add`) does
  // not trap, while leaf {1} (both `in_buf` and `out_buf` bit preserving)
  // traps.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         xla::ParseAndReturnUnverifiedModule(R"hlo(
HloModule inplace_cp_per_leaf
ENTRY main {
  %p0 = f32[4] parameter(0)
  %p1 = f32[4] parameter(1)
  %a = f32[4] add(%p0, %p0)
  %in_buf = (f32[4], f32[4]) tuple(%a, %p0)
  %out_buf = (f32[4], f32[4]) tuple(%a, %p1)
  %c0 = s32[] constant(0)
  %off_pair = (s32[]) tuple(%c0)
  %offs = ((s32[]), (s32[])) tuple(%off_pair, %off_pair)
  ROOT %cp = (f32[4], f32[4]) collective-permute(%in_buf, %out_buf, %offs, %offs),
      source_target_pairs={{0,1}}, slice_sizes={{2},{2}}
}
)hlo"));
    const CallbackCheckResult no_trap = RunComparisonCallbackCheck(
        *module, "cp", LiteralUtil::MakeTuple({&sub_lit, &sub_lit}),
        LiteralUtil::MakeTuple({&zero_lit, &sub_lit}));
    EXPECT_EQ(no_trap.failure_count, 0);
    EXPECT_EQ(no_trap.test_result.numeric_checks_size(), 0);

    const CallbackCheckResult trap = RunComparisonCallbackCheck(
        *module, "cp", LiteralUtil::MakeTuple({&sub_lit, &sub_lit}),
        LiteralUtil::MakeTuple({&sub_lit, &zero_lit}));
    EXPECT_EQ(trap.failure_count, 1);
    EXPECT_EQ(trap.test_result.numeric_checks_size(), 1);
  }
}

}  // namespace
}  // namespace hlo_isolation
}  // namespace xla

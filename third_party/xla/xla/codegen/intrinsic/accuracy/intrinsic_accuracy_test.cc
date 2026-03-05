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

// HLO-level accuracy tests for unary math intrinsics.
//
// These tests operate at the HLO level, compiling and running HLO modules
// through XLA's full compilation pipeline. This means:
//   1. Tests are resilient to changes in the underlying intrinsic
//      implementations (e.g., swapping LLVM intrinsics, changing polynomial
//      approximations, etc.).
//   2. They test the actual end-to-end path that user code follows.
//   3. Accuracy regressions are caught regardless of where in the pipeline
//      the regression was introduced.
//
// Golden baselines are generated offline using mpmath at 50 digits of
// precision (see generate_golden_baselines.py).
//
// NOTE: XLA already has exhaustive tests in xla/tests/exhaustive/ that test
// every representable float value for F32 and smaller types (and sampled
// subsets for F64). Those tests compare against a reference interpreter
// backend. These golden-baseline tests complement the exhaustive suite by:
//   - Comparing against an independent high-precision reference (mpmath),
//     not XLA's own interpreter.
//   - Providing explicit ULP budgets per op that serve as a contract.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/codegen/intrinsic/accuracy/accuracy_budget.h"
#include "xla/codegen/intrinsic/accuracy/golden_baselines.h"
#include "xla/fp_util.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {
using codegen::intrinsic::accuracy::AccuracyBudget;
using codegen::intrinsic::accuracy::RefPoint;
using codegen::intrinsic::accuracy::UlpBudget;

// ---------------------------------------------------------------------------
// Test fixture: uses PjRt test runner.
// ---------------------------------------------------------------------------

using HloIntrinsicAccuracyTest = HloPjRtTestBase;

// ---------------------------------------------------------------------------
// Accuracy reporting (ULP-based, independent of XLA's ErrorSpec).
// ---------------------------------------------------------------------------

struct ErrorStats {
  int64_t max_ulp_error = 0;
  double mean_ulp_error = 0.0;
  int count = 0;
  double worst_input = 0.0;
  double worst_expected = 0.0;
  double worst_actual = 0.0;
  double total_ulp = 0.0;
};

struct SpecialMismatch {
  double input;
  double expected;
  double actual;
};

struct AccuracyReport {
  ErrorStats regular;
  ErrorStats subnormal;
  int64_t special_values_mismatches = 0;
  std::vector<SpecialMismatch> sample_special_mismatches;
};

template <typename T>
AccuracyReport ComputeAccuracyReport(absl::Span<const RefPoint> golden,
                                     const T* results, size_t count) {
  AccuracyReport report;

  for (size_t i = 0; i < count; ++i) {
    T expected;
    if constexpr (std::is_same_v<T, float>) {
      expected = static_cast<T>(golden[i].expected_f32);
    } else {
      expected = static_cast<T>(golden[i].expected_f64);
    }
    T actual = results[i];
    std::optional<int64_t> ulp_opt = UlpDistance(actual, expected);
    if (!ulp_opt.has_value()) {
      report.special_values_mismatches++;
      if (report.sample_special_mismatches.size() < 10) {
        report.sample_special_mismatches.push_back(
            {golden[i].input, static_cast<double>(expected),
             static_cast<double>(actual)});
      }
      continue;
    }
    int64_t ulp = ulp_opt.value();

    T input = static_cast<T>(golden[i].input);
    bool is_subnormal = std::fpclassify(input) == FP_SUBNORMAL ||
                        std::fpclassify(expected) == FP_SUBNORMAL;
    auto& stats = is_subnormal ? report.subnormal : report.regular;
    stats.count++;
    stats.total_ulp += ulp;

    if (ulp > stats.max_ulp_error) {
      stats.max_ulp_error = ulp;
      stats.worst_input = golden[i].input;
      stats.worst_expected = static_cast<double>(expected);
      stats.worst_actual = static_cast<double>(actual);
    }
  }

  auto update_mean_ulp = [](ErrorStats& stats) {
    if (stats.count > 0) {
      stats.mean_ulp_error = stats.total_ulp / stats.count;
    }
  };
  update_mean_ulp(report.regular);
  update_mean_ulp(report.subnormal);
  return report;
}

void LogAccuracyReport(const AccuracyReport& report,
                       absl::string_view test_name) {
  LOG(INFO) << "Accuracy Report for " << test_name << ":\n"
            << "  Regular tested points: " << report.regular.count << "\n"
            << "  Regular mean ULP Error: " << report.regular.mean_ulp_error
            << "\n  Regular max ULP Error: " << report.regular.max_ulp_error
            << "\n    Worst Case: input=" << report.regular.worst_input
            << ", expected=" << report.regular.worst_expected
            << ", actual=" << report.regular.worst_actual << "\n"
            << "  Subnormal tested points: " << report.subnormal.count << "\n"
            << "  Subnormal mean ULP Error: " << report.subnormal.mean_ulp_error
            << "\n  Subnormal Max ULP Error: " << report.subnormal.max_ulp_error
            << "\n    Worst Case: input=" << report.subnormal.worst_input
            << ", expected=" << report.subnormal.worst_expected
            << ", actual=" << report.subnormal.worst_actual << "\n"
            << "  Special Value (NaN/Inf) mismatches: "
            << report.special_values_mismatches;
}

// ---------------------------------------------------------------------------
// HLO module templates.
// ---------------------------------------------------------------------------

std::string MakeUnaryHloModule(absl::string_view op_name, PrimitiveType type,
                               int64_t count) {
  std::string type_str = primitive_util::LowercasePrimitiveTypeName(type);
  return absl::StrFormat(R"(
HloModule %s_accuracy_test

ENTRY main {
  input = %s[%d] parameter(0)
  ROOT result = %s[%d] %s(input)
}
)",
                         op_name, type_str, count, type_str, count, op_name);
}

std::string ToPascalCase(absl::string_view name) {
  std::vector<std::string> words = absl::StrSplit(name, absl::ByAnyChar(" -_"));
  absl::c_transform(words, words.begin(), [](absl::string_view word) {
    return absl::AsciiStrToUpper(word.substr(0, 1)) +
           absl::AsciiStrToLower(word.substr(1));
  });
  return absl::StrJoin(words, "");
}

// ---------------------------------------------------------------------------
// Parameterized test infrastructure.
// ---------------------------------------------------------------------------

struct IntrinsicAccuracyTestParam {
  std::string hlo_op_name;
  PrimitiveType primitive_type;
  absl::Span<const RefPoint> golden_data;
  AccuracyBudget budget;
  bool fast_math = false;

  std::string name() const {
    return absl::StrCat(
        ToPascalCase(hlo_op_name), "_",
        primitive_util::LowercasePrimitiveTypeName(primitive_type));
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const IntrinsicAccuracyTestParam& param) {
    return os << param.name();
  }
};

std::vector<IntrinsicAccuracyTestParam> GetAccuracyTestParams() {
  namespace accuracy = ::xla::codegen::intrinsic::accuracy;
  std::vector<IntrinsicAccuracyTestParam> params = {
      {"tanh", F32, accuracy::kGoldenTanh, accuracy::kTanhF32Budget},
      {"tanh", F64, accuracy::kGoldenTanh, accuracy::kTanhF64Budget},

      {"exponential", F32, accuracy::kGoldenExp, accuracy::kExpF32Budget},
      {"exponential", F64, accuracy::kGoldenExp, accuracy::kExpF64Budget},

      {"log-plus-one", F32, accuracy::kGoldenLog1p, accuracy::kLog1pF32Budget},
      {"log-plus-one", F64, accuracy::kGoldenLog1p, accuracy::kLog1pF64Budget},

      {"rsqrt", F32, accuracy::kGoldenRsqrt, accuracy::kRsqrtF32Budget},
      {"rsqrt", F64, accuracy::kGoldenRsqrt, accuracy::kRsqrtF64Budget},

      {"sqrt", F32, accuracy::kGoldenSqrt, accuracy::kSqrtF32Budget},
      {"sqrt", F64, accuracy::kGoldenSqrt, accuracy::kSqrtF64Budget},

      {"erf", F32, accuracy::kGoldenErf, accuracy::kErfF32Budget},
      {"erf", F64, accuracy::kGoldenErf, accuracy::kErfF64Budget}};
  return params;
}

// Filter golden points: remove those whose input overflows when cast to T,
// or whose input underflows to zero.
template <typename T>
std::vector<RefPoint> FilterGoldenForType(absl::Span<const RefPoint> data) {
  std::vector<RefPoint> filtered;
  filtered.reserve(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    T input = static_cast<T>(data[i].input);
    // Skip points where the input overflows to inf during cast.
    if (std::isinf(input) && !std::isinf(data[i].input)) {
      continue;
    }
    // Skip points where casting to T fundamentally changes the math domain.
    // For example, if a negative subnormal underflows to -0.0, the test would
    // incorrectly expect NaN while XLA correctly computes -0.0.
    bool expected_is_nan = false;
    if constexpr (std::is_same_v<T, float>) {
      expected_is_nan = std::isnan(data[i].expected_f32);
    } else {
      expected_is_nan = std::isnan(data[i].expected_f64);
    }
    if (input == 0.0 && data[i].input != 0.0 && expected_is_nan) {
      continue;
    }
    filtered.push_back(data[i]);
  }
  return filtered;
}

class HloIntrinsicAccuracyParamTest
    : public HloIntrinsicAccuracyTest,
      public ::testing::WithParamInterface<IntrinsicAccuracyTestParam> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloRunnerAgnosticTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_enable_fast_math(GetParam().fast_math);
    debug_options.set_xla_cpu_enable_fast_min_max(GetParam().fast_math);
    debug_options.set_xla_gpu_enable_fast_min_max(GetParam().fast_math);
    return debug_options;
  }

 protected:
  void RunAccuracyTest(const IntrinsicAccuracyTestParam& param) {
    // Dynamically choose between F32 and F64 templates.
    if (param.primitive_type == F32) {
      DoRunAccuracyTest<float>(param);
    } else if (param.primitive_type == F64) {
      DoRunAccuracyTest<double>(param);
    } else {
      GTEST_SKIP() << "Unsupported type";
    }
  }

  bool is_cpu() const {
    return test_runner().HasProperty(HloRunnerPropertyTag::kCpu);
  }

 private:
  template <typename T>
  void DoRunAccuracyTest(const IntrinsicAccuracyTestParam& param) {
    // The hlo_op_name is now directly available in param.hlo_op_name.
    // No need for brittle string replacement logic.

    auto full_golden = FilterGoldenForType<T>(param.golden_data);

    std::vector<T> inputs;
    std::vector<RefPoint> golden;
    inputs.reserve(full_golden.size());
    golden.reserve(full_golden.size());

    for (const auto& point : full_golden) {
      T input = static_cast<T>(point.input);

      // Handle custom domain and overrides for rsqrt
      RefPoint test_point = point;

      inputs.push_back(input);
      golden.push_back(test_point);
    }
    auto input_literal = LiteralUtil::CreateR1<T>(inputs);
    int64_t count = inputs.size();

    std::string hlo =
        MakeUnaryHloModule(param.hlo_op_name, param.primitive_type, count);
    TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
    module->mutable_config().set_debug_options(GetDebugOptionsForTest());

    TF_ASSERT_OK_AND_ASSIGN(auto result,
                            Execute(std::move(module), {&input_literal}));

    auto result_data = result.template data<T>();
    auto report = ComputeAccuracyReport<T>(golden, result_data.data(),
                                           result_data.size());
    LogAccuracyReport(report, ToPascalCase(param.hlo_op_name));

    const UlpBudget& budget = is_cpu() ? param.budget.cpu : param.budget.gpu;

    EXPECT_LE(report.regular.max_ulp_error, budget.regular)
        << "Regular max ULP error " << report.regular.max_ulp_error
        << " exceeds budget " << budget.regular
        << ". Worst case: input=" << report.regular.worst_input << " ("
        << absl::StrFormat("%a", report.regular.worst_input) << ", cast to "
        << static_cast<T>(report.regular.worst_input)
        << "), expected=" << report.regular.worst_expected << " ("
        << absl::StrFormat("%a", report.regular.worst_expected)
        << "), actual=" << report.regular.worst_actual << " ("
        << absl::StrFormat("%a", report.regular.worst_actual) << ")";

    EXPECT_LE(report.subnormal.max_ulp_error, budget.subnormal)
        << "Subnormal max ULP error " << report.subnormal.max_ulp_error
        << " exceeds budget " << budget.subnormal
        << ". Worst case: input=" << report.subnormal.worst_input << " ("
        << absl::StrFormat("%a", report.subnormal.worst_input)
        << "), expected=" << report.subnormal.worst_expected << " ("
        << absl::StrFormat("%a", report.subnormal.worst_expected)
        << "), actual=" << report.subnormal.worst_actual << " ("
        << absl::StrFormat("%a", report.subnormal.worst_actual) << ")";

    std::string mismatches_str;
    if (report.special_values_mismatches > budget.special_values) {
      for (const auto& m : report.sample_special_mismatches) {
        absl::StrAppendFormat(&mismatches_str,
                              "\n  Input: %a, Expected: %a, Actual: %a",
                              m.input, m.expected, m.actual);
      }
    }
    EXPECT_LE(report.special_values_mismatches, budget.special_values)
        << "Too many Special Value (NaN/Inf) mismatches. Samples:"
        << mismatches_str;
  }
};

TEST_P(HloIntrinsicAccuracyParamTest, WithinUlpBudget) {
  const auto& param = GetParam();
  RunAccuracyTest(param);
}

// ---------------------------------------------------------------------------
// Test case registration.
// ---------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    UnaryIntrinsics, HloIntrinsicAccuracyParamTest,
    ::testing::ValuesIn(GetAccuracyTestParams()),
    [](const ::testing::TestParamInfo<IntrinsicAccuracyTestParam>& info) {
      return info.param.name();
    });

}  // namespace
}  // namespace xla

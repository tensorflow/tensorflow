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

#include "xla/hlo/transforms/expanders/elementary_function_expander.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/types.h"

namespace xla {
namespace {

using ElementaryFunctionExpanderTest = HloHardwareIndependentTestBase;

bool ContainsOpcode(const HloModule& module, HloOpcode opcode) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() == opcode) {
        return true;
      }
    }
  }
  return false;
}

TEST_F(ElementaryFunctionExpanderTest, SinhF32AndSub32BitTypesNotExpanded) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p_f32 = f32[4] parameter(0)
      p_f16 = f16[4] parameter(1)
      p_bf16 = bf16[4] parameter(2)
      s_f32 = f32[4] sinh(p_f32)
      s_f16 = f16[4] sinh(p_f16)
      s_bf16 = bf16[4] sinh(p_bf16)
      ROOT out = (f32[4], f16[4], bf16[4]) tuple(s_f32, s_f16, s_bf16)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ElementaryFunctionExpanderTest,
       SinhF32AndSub32BitTypesExpandedUnfiltered) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p_f32 = f32[4] parameter(0)
      p_f16 = f16[4] parameter(1)
      p_bf16 = bf16[4] parameter(2)
      s_f32 = f32[4] sinh(p_f32)
      s_f16 = f16[4] sinh(p_f16)
      s_bf16 = bf16[4] sinh(p_bf16)
      ROOT out = (f32[4], f16[4], bf16[4]) tuple(s_f32, s_f16, s_bf16)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander(/*extra_filter=*/nullptr);
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kSinh));
}

TEST_F(ElementaryFunctionExpanderTest, SinhF64ExpandedAndAccurate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p = f64[14] parameter(0)
      ROOT s = f64[14] sinh(p)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kSinh));

  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> inputs = {
      0.0,
      -0.0,
      1e-12,
      -1e-12,
      0.5,
      -0.5,
      1.0,
      -1.0,
      10.0,
      -10.0,
      // 710.0 > ln(DBL_MAX) (~709.78): exp(710.0) overflows to +inf, but
      // sinh(710.0) is finite (~1.1169973830808757e308).
      710.0,
      -710.0,
      inf,
      nan,
  };
  Literal input_literal = LiteralUtil::CreateR1<double>(inputs);
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(Literal result_literal,
                       evaluator.Evaluate(*module, {&input_literal}));
  auto results = result_literal.data<double>();

  // Verify signed zero preservation.
  EXPECT_EQ(results[0], 0.0);
  EXPECT_FALSE(std::signbit(results[0]));
  EXPECT_EQ(results[1], -0.0);
  EXPECT_TRUE(std::signbit(results[1]));

  for (size_t i = 2; i < 12; ++i) {
    double expected = std::sinh(inputs[i]);
    double actual = results[i];
    EXPECT_TRUE(std::isfinite(actual)) << "i=" << i << " x=" << inputs[i];
    double ulp_tol =
        2.0 * std::abs(expected) * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(actual, expected, ulp_tol) << "i=" << i << " x=" << inputs[i];
  }

  EXPECT_TRUE(std::isinf(results[12]) && results[12] > 0);
  EXPECT_TRUE(std::isnan(results[13]));
}

TEST_F(ElementaryFunctionExpanderTest, SinhComplexExpandedAndAccurate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p64 = c64[3] parameter(0)
      p128 = c128[3] parameter(1)
      sinh_c64 = c64[3] sinh(p64)
      sinh_c128 = c128[3] sinh(p128)
      ROOT out = (c64[3], c128[3]) tuple(sinh_c64, sinh_c128)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kSinh));

  // Include 89.0 + 0.5i for c64 (where exp(89.0f) overflows float32) and
  // 710.0 + 0.5i for c128 (where exp(710.0) overflows float64).
  std::vector<complex64> inputs_c64 = {
      {0.5f, 0.3f},
      {-2.0f, 1.5f},
      {89.0f, 0.5f},
  };
  std::vector<complex128> inputs_c128 = {
      {0.5, 0.3},
      {-2.0, 1.5},
      {710.0, 0.5},
  };
  Literal input_c64_lit = LiteralUtil::CreateR1<complex64>(inputs_c64);
  Literal input_c128_lit = LiteralUtil::CreateR1<complex128>(inputs_c128);

  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {&input_c64_lit, &input_c128_lit}));

  std::vector<Literal> tuple_elements = result_literal.DecomposeTuple();
  auto results_c64 = tuple_elements[0].data<complex64>();
  auto results_c128 = tuple_elements[1].data<complex128>();

  for (size_t i = 0; i < inputs_c64.size(); ++i) {
    std::complex<double> expected = std::sinh(
        std::complex<double>(inputs_c64[i].real(), inputs_c64[i].imag()));
    EXPECT_TRUE(std::isfinite(results_c64[i].real())) << "i=" << i;
    EXPECT_TRUE(std::isfinite(results_c64[i].imag())) << "i=" << i;
    double tol_r =
        4.0 * std::abs(expected.real()) * std::numeric_limits<float>::epsilon();
    double tol_i =
        4.0 * std::abs(expected.imag()) * std::numeric_limits<float>::epsilon();
    EXPECT_NEAR(results_c64[i].real(), expected.real(), tol_r) << "i=" << i;
    EXPECT_NEAR(results_c64[i].imag(), expected.imag(), tol_i) << "i=" << i;
  }

  for (size_t i = 0; i < inputs_c128.size(); ++i) {
    std::complex<double> expected = std::sinh(inputs_c128[i]);
    EXPECT_TRUE(std::isfinite(results_c128[i].real())) << "i=" << i;
    EXPECT_TRUE(std::isfinite(results_c128[i].imag())) << "i=" << i;
    double tol_r = 4.0 * std::abs(expected.real()) *
                   std::numeric_limits<double>::epsilon();
    double tol_i = 4.0 * std::abs(expected.imag()) *
                   std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(results_c128[i].real(), expected.real(), tol_r) << "i=" << i;
    EXPECT_NEAR(results_c128[i].imag(), expected.imag(), tol_i) << "i=" << i;
  }
}

TEST_F(ElementaryFunctionExpanderTest, CoshF32AndSub32BitTypesNotExpanded) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p_f32 = f32[4] parameter(0)
      p_f16 = f16[4] parameter(1)
      p_bf16 = bf16[4] parameter(2)
      c_f32 = f32[4] cosh(p_f32)
      c_f16 = f16[4] cosh(p_f16)
      c_bf16 = bf16[4] cosh(p_bf16)
      ROOT out = (f32[4], f16[4], bf16[4]) tuple(c_f32, c_f16, c_bf16)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ElementaryFunctionExpanderTest,
       CoshF32AndSub32BitTypesExpandedUnfiltered) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p_f32 = f32[4] parameter(0)
      p_f16 = f16[4] parameter(1)
      p_bf16 = bf16[4] parameter(2)
      c_f32 = f32[4] cosh(p_f32)
      c_f16 = f16[4] cosh(p_f16)
      c_bf16 = bf16[4] cosh(p_bf16)
      ROOT out = (f32[4], f16[4], bf16[4]) tuple(c_f32, c_f16, c_bf16)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander(/*extra_filter=*/nullptr);
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kCosh));
}

TEST_F(ElementaryFunctionExpanderTest, CoshF64ExpandedAndAccurate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p = f64[14] parameter(0)
      ROOT c = f64[14] cosh(p)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kCosh));

  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> inputs = {
      0.0,
      -0.0,
      1e-12,
      -1e-12,
      0.5,
      -0.5,
      1.0,
      -1.0,
      10.0,
      -10.0,
      // 710.0 > ln(DBL_MAX) (~709.78): exp(710.0) overflows to +inf, but
      // cosh(710.0) is finite (~1.1169973830808757e308).
      710.0,
      -710.0,
      inf,
      nan,
  };
  Literal input_literal = LiteralUtil::CreateR1<double>(inputs);
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(Literal result_literal,
                       evaluator.Evaluate(*module, {&input_literal}));
  auto results = result_literal.data<double>();

  EXPECT_EQ(results[0], 1.0);
  EXPECT_EQ(results[1], 1.0);

  for (size_t i = 2; i < 12; ++i) {
    double expected = std::cosh(inputs[i]);
    double actual = results[i];
    EXPECT_TRUE(std::isfinite(actual)) << "i=" << i << " x=" << inputs[i];
    double ulp_tol =
        2.0 * std::abs(expected) * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(actual, expected, ulp_tol) << "i=" << i << " x=" << inputs[i];
  }

  EXPECT_TRUE(std::isinf(results[12]) && results[12] > 0);
  EXPECT_TRUE(std::isnan(results[13]));
}

TEST_F(ElementaryFunctionExpanderTest, CoshComplexExpandedAndAccurate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p64 = c64[3] parameter(0)
      p128 = c128[3] parameter(1)
      cosh_c64 = c64[3] cosh(p64)
      cosh_c128 = c128[3] cosh(p128)
      ROOT out = (c64[3], c128[3]) tuple(cosh_c64, cosh_c128)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kCosh));

  std::vector<complex64> inputs_c64 = {
      {0.5f, 0.3f},
      {-2.0f, 1.5f},
      {89.0f, 0.5f},
  };
  std::vector<complex128> inputs_c128 = {
      {0.5, 0.3},
      {-2.0, 1.5},
      {710.0, 0.5},
  };
  Literal input_c64_lit = LiteralUtil::CreateR1<complex64>(inputs_c64);
  Literal input_c128_lit = LiteralUtil::CreateR1<complex128>(inputs_c128);

  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {&input_c64_lit, &input_c128_lit}));

  std::vector<Literal> tuple_elements = result_literal.DecomposeTuple();
  auto results_c64 = tuple_elements[0].data<complex64>();
  auto results_c128 = tuple_elements[1].data<complex128>();

  for (size_t i = 0; i < inputs_c64.size(); ++i) {
    std::complex<double> expected = std::cosh(
        std::complex<double>(inputs_c64[i].real(), inputs_c64[i].imag()));
    EXPECT_TRUE(std::isfinite(results_c64[i].real())) << "i=" << i;
    EXPECT_TRUE(std::isfinite(results_c64[i].imag())) << "i=" << i;
    double tol_r =
        4.0 * std::abs(expected.real()) * std::numeric_limits<float>::epsilon();
    double tol_i =
        4.0 * std::abs(expected.imag()) * std::numeric_limits<float>::epsilon();
    EXPECT_NEAR(results_c64[i].real(), expected.real(), tol_r) << "i=" << i;
    EXPECT_NEAR(results_c64[i].imag(), expected.imag(), tol_i) << "i=" << i;
  }

  for (size_t i = 0; i < inputs_c128.size(); ++i) {
    std::complex<double> expected = std::cosh(inputs_c128[i]);
    EXPECT_TRUE(std::isfinite(results_c128[i].real())) << "i=" << i;
    EXPECT_TRUE(std::isfinite(results_c128[i].imag())) << "i=" << i;
    double tol_r = 4.0 * std::abs(expected.real()) *
                   std::numeric_limits<double>::epsilon();
    double tol_i = 4.0 * std::abs(expected.imag()) *
                   std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(results_c128[i].real(), expected.real(), tol_r) << "i=" << i;
    EXPECT_NEAR(results_c128[i].imag(), expected.imag(), tol_i) << "i=" << i;
  }
}

TEST_F(ElementaryFunctionExpanderTest, AcosAndAsinF32AndSub32BitTypesHandling) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p_f32 = f32[4] parameter(0)
      p_f16 = f16[4] parameter(1)
      p_bf16 = bf16[4] parameter(2)
      acos_f32 = f32[4] acos(p_f32)
      acos_f16 = f16[4] acos(p_f16)
      acos_bf16 = bf16[4] acos(p_bf16)
      asin_f32 = f32[4] asin(p_f32)
      asin_f16 = f16[4] asin(p_f16)
      asin_bf16 = bf16[4] asin(p_bf16)
      ROOT out = (f32[4], f16[4], bf16[4], f32[4], f16[4], bf16[4]) tuple(
          acos_f32, acos_f16, acos_bf16, asin_f32, asin_f16, asin_bf16)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander default_expander;
  ASSERT_OK_AND_ASSIGN(bool changed, default_expander.Run(module.get()));
  EXPECT_FALSE(changed);

  ElementaryFunctionExpander unfiltered_expander(/*extra_filter=*/nullptr);
  ASSERT_OK_AND_ASSIGN(changed, unfiltered_expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kAcos));
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kAsin));
}

TEST_F(ElementaryFunctionExpanderTest, AcosAndAsinF64ExpandedAndAccurate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p = f64[13] parameter(0)
      acos_val = f64[13] acos(p)
      asin_val = f64[13] asin(p)
      ROOT out = (f64[13], f64[13]) tuple(acos_val, asin_val)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kAcos));
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kAsin));

  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> inputs = {0.0,  -0.0,       1e-16,       -1e-16, 0.5,
                                -0.5, 1.0 - 1e-9, -1.0 + 1e-9, 1.0,    -1.0,
                                1.5,  -1.5,       nan};
  Literal input_literal = LiteralUtil::CreateR1<double>(inputs);
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(Literal result_literal,
                       evaluator.Evaluate(*module, {&input_literal}));
  std::vector<Literal> tuple_elements = result_literal.DecomposeTuple();
  auto acos_results = tuple_elements[0].data<double>();
  auto asin_results = tuple_elements[1].data<double>();

  // Verify signed zero preservation for asin.
  EXPECT_EQ(asin_results[0], 0.0);
  EXPECT_FALSE(std::signbit(asin_results[0]));
  EXPECT_EQ(asin_results[1], -0.0);
  EXPECT_TRUE(std::signbit(asin_results[1]));

  for (size_t i = 0; i < 10; ++i) {
    double expected_acos = std::acos(inputs[i]);
    double expected_asin = std::asin(inputs[i]);
    double tol_acos =
        2.0 * std::abs(expected_acos) * std::numeric_limits<double>::epsilon();
    double tol_asin =
        2.0 * std::abs(expected_asin) * std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(acos_results[i], expected_acos, tol_acos) << "i=" << i;
    EXPECT_NEAR(asin_results[i], expected_asin, tol_asin) << "i=" << i;
  }
  for (size_t i = 10; i < inputs.size(); ++i) {
    EXPECT_TRUE(std::isnan(acos_results[i])) << "i=" << i;
    EXPECT_TRUE(std::isnan(asin_results[i])) << "i=" << i;
  }
}

TEST_F(ElementaryFunctionExpanderTest, AcosAndAsinComplexExpandedAndAccurate) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY main {
      p64 = c64[2] parameter(0)
      p128 = c128[2] parameter(1)
      acos_c64 = c64[2] acos(p64)
      acos_c128 = c128[2] acos(p128)
      asin_c64 = c64[2] asin(p64)
      asin_c128 = c128[2] asin(p128)
      ROOT out = (c64[2], c128[2], c64[2], c128[2]) tuple(
          acos_c64, acos_c128, asin_c64, asin_c128)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_string));
  ElementaryFunctionExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kAcos));
  EXPECT_FALSE(ContainsOpcode(*module, HloOpcode::kAsin));

  std::vector<complex64> inputs_c64 = {{0.5f, 0.3f}, {-1.2f, 0.8f}};
  std::vector<complex128> inputs_c128 = {{0.5, 0.3}, {-1.2, 0.8}};
  Literal input_c64_lit = LiteralUtil::CreateR1<complex64>(inputs_c64);
  Literal input_c128_lit = LiteralUtil::CreateR1<complex128>(inputs_c128);

  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(
      Literal result_literal,
      evaluator.Evaluate(*module, {&input_c64_lit, &input_c128_lit}));
  std::vector<Literal> tuple_elements = result_literal.DecomposeTuple();
  auto acos_c64 = tuple_elements[0].data<complex64>();
  auto acos_c128 = tuple_elements[1].data<complex128>();
  auto asin_c64 = tuple_elements[2].data<complex64>();
  auto asin_c128 = tuple_elements[3].data<complex128>();

  for (size_t i = 0; i < inputs_c64.size(); ++i) {
    std::complex<double> z(inputs_c64[i].real(), inputs_c64[i].imag());
    std::complex<double> exp_acos = std::acos(z);
    std::complex<double> exp_asin = std::asin(z);
    float eps = std::numeric_limits<float>::epsilon();
    EXPECT_NEAR(acos_c64[i].real(), exp_acos.real(), 4.0 * eps);
    EXPECT_NEAR(acos_c64[i].imag(), exp_acos.imag(), 4.0 * eps);
    EXPECT_NEAR(asin_c64[i].real(), exp_asin.real(), 4.0 * eps);
    EXPECT_NEAR(asin_c64[i].imag(), exp_asin.imag(), 4.0 * eps);
  }
  for (size_t i = 0; i < inputs_c128.size(); ++i) {
    std::complex<double> exp_acos = std::acos(inputs_c128[i]);
    std::complex<double> exp_asin = std::asin(inputs_c128[i]);
    double eps = std::numeric_limits<double>::epsilon();
    EXPECT_NEAR(acos_c128[i].real(), exp_acos.real(), 4.0 * eps);
    EXPECT_NEAR(acos_c128[i].imag(), exp_acos.imag(), 4.0 * eps);
    EXPECT_NEAR(asin_c128[i].real(), exp_asin.real(), 4.0 * eps);
    EXPECT_NEAR(asin_c128[i].imag(), exp_asin.imag(), 4.0 * eps);
  }
}
}  // namespace
}  // namespace xla

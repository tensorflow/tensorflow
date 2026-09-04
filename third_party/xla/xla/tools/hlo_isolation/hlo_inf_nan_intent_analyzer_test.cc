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

#include "xla/tools/hlo_isolation/hlo_inf_nan_intent_analyzer.h"

#include <cstdint>
#include <limits>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace hlo_isolation {
namespace {

TEST(HloInfNanIntentAnalyzerTest, LiteralContainsInfOrNan) {
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

  Literal c64_normal =
      LiteralUtil::CreateR1<complex64>({{1.0f, 2.0f}, {3.0f, 4.0f}});
  EXPECT_FALSE(LiteralContainsInfOrNan(c64_normal));

  Literal c64_inf = LiteralUtil::CreateR1<complex64>(
      {{1.0f, 2.0f}, {std::numeric_limits<float>::infinity(), 4.0f}});
  EXPECT_TRUE(LiteralContainsInfOrNan(c64_inf));

  Literal c64_nan = LiteralUtil::CreateR1<complex64>(
      {{1.0f, std::numeric_limits<float>::quiet_NaN()}, {3.0f, 4.0f}});
  EXPECT_TRUE(LiteralContainsInfOrNan(c64_nan));

  Literal tuple_with_inf =
      LiteralUtil::MakeTuple({&f32_normal, &f32_inf, &s32_normal});
  EXPECT_TRUE(LiteralContainsInfOrNan(tuple_with_inf));

  Literal tuple_no_inf = LiteralUtil::MakeTuple({&f32_normal, &s32_normal});
  EXPECT_FALSE(LiteralContainsInfOrNan(tuple_no_inf));
}

TEST(HloInfNanIntentAnalyzerTest, NoConstantInfNanReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule module_no_inf
ENTRY main {
  p0 = f32[10] parameter(0)
  c0 = f32[] constant(1.0)
  b0 = f32[10] broadcast(c0), dimensions={}
  ROOT add = f32[10] add(p0, b0)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, PureStructuralSelectReturnsTrue) {
  const absl::string_view kHlo = R"hlo(
HloModule module_select_inf
ENTRY main {
  p0 = pred[10] parameter(0)
  c_inf = f32[] constant(-inf)
  b_inf = f32[10] broadcast(c_inf), dimensions={}
  p1 = f32[10] parameter(1)
  ROOT sel = f32[10] select(p0, b_inf, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, PureStructuralPadReturnsTrue) {
  const absl::string_view kHlo = R"hlo(
HloModule module_pad_nan
ENTRY main {
  p0 = f32[8] parameter(0)
  c_nan = f32[] constant(nan)
  ROOT pad = f32[10] pad(p0, c_nan), padding=1_1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, PureStructuralTupleAndSliceReturnsTrue) {
  const absl::string_view kHlo = R"hlo(
HloModule module_tuple_slice
ENTRY main {
  p0 = pred[10] parameter(0)
  c_inf = f32[] constant(inf)
  b_inf = f32[10] broadcast(c_inf), dimensions={}
  p1 = f32[10] parameter(1)
  sel = f32[10] select(p0, b_inf, p1)
  t = (f32[10], f32[10]) tuple(sel, p1)
  gte = f32[10] get-tuple-element(t), index=0
  ROOT s = f32[5] slice(gte), slice={[0:5]}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, PureStructuralFusionReturnsTrue) {
  const absl::string_view kHlo = R"hlo(
HloModule direct_select_inf_fusion

%fused_comp (param_0: pred[10], param_1: f32[10]) -> f32[10] {
  %param_0 = pred[10] parameter(0)
  %c_neg_inf = f32[] constant(-inf)
  %broadcast_neg_inf = f32[10] broadcast(%c_neg_inf), dimensions={}
  %param_1 = f32[10] parameter(1)
  ROOT %select = f32[10] select(%param_0, %broadcast_neg_inf, %param_1)
}

ENTRY main (parameter.0: pred[10], parameter.1: f32[10]) -> f32[10] {
  %parameter.0 = pred[10] parameter(0)
  %parameter.1 = f32[10] parameter(1)
  ROOT %fusion = f32[10] fusion(%parameter.0, %parameter.1), kind=kLoop, calls=%fused_comp
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest,
     ArithmeticPropagationWithoutInfNanProducingOpsReturnsTrue) {
  const absl::string_view kHlo = R"hlo(
HloModule arithmetic_propagation
ENTRY main {
  p0 = pred[10] parameter(0)
  c_nan = f32[] constant(nan)
  b_nan = f32[10] broadcast(c_nan), dimensions={}
  p1 = f32[10] parameter(1)
  sel = f32[10] select(p0, b_nan, p1)
  c_scale = f32[] constant(2.0)
  b_scale = f32[10] broadcast(c_scale), dimensions={}
  mul = f32[10] multiply(sel, b_scale)
  ROOT add = f32[10] add(mul, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest,
     ArithmeticPropagationWithInfNanProducingOpReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule arithmetic_with_sqrt
ENTRY main {
  p0 = pred[10] parameter(0)
  c_nan = f32[] constant(nan)
  b_nan = f32[10] broadcast(c_nan), dimensions={}
  p1 = f32[10] parameter(1)
  sel = f32[10] select(p0, b_nan, p1)
  c_scale = f32[] constant(2.0)
  b_scale = f32[10] broadcast(c_scale), dimensions={}
  mul = f32[10] multiply(sel, b_scale)
  // sqrt can produce NaN from finite inputs, preventing arithmetic propagation.
  sq = f32[10] sqrt(p1)
  ROOT add = f32[10] add(mul, sq)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, ArithmeticPropagationWithLog1pReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule arithmetic_with_log1p
ENTRY main {
  p0 = pred[10] parameter(0)
  c_nan = f32[] constant(nan)
  b_nan = f32[10] broadcast(c_nan), dimensions={}
  p1 = f32[10] parameter(1)
  sel = f32[10] select(p0, b_nan, p1)
  c_scale = f32[] constant(2.0)
  b_scale = f32[10] broadcast(c_scale), dimensions={}
  mul = f32[10] multiply(sel, b_scale)
  // log-plus-one produces -inf for x=-1 and NaN for x<-1 from finite inputs.
  l1p = f32[10] log-plus-one(p1)
  ROOT add = f32[10] add(mul, l1p)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, GuardrailComparisonOnlyReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule nan_to_num_guardrail
ENTRY main {
  x = f32[10] parameter(0)
  c_inf = f32[] constant(inf)
  b_inf = f32[10] broadcast(c_inf), dimensions={}
  // c_inf is used as a comparison threshold to sanitize inputs; it does not
  // flow into the output data tensor.
  is_inf = pred[10] compare(x, b_inf), direction=EQ
  c_max = f32[] constant(1.0e30)
  b_max = f32[10] broadcast(c_max), dimensions={}
  ROOT sel = f32[10] select(is_inf, b_max, x)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, IntentionalMaskedReductionReturnsTrue) {
  const absl::string_view kHlo = R"hlo(
HloModule masked_reduction
%max_reducer (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum = f32[] maximum(%x, %y)
}

ENTRY main {
  mask = pred[10] parameter(0)
  data = f32[10] parameter(1)
  c_neg_inf = f32[] constant(-inf)
  b_neg_inf = f32[10] broadcast(c_neg_inf), dimensions={}
  sel = f32[10] select(mask, b_neg_inf, data)
  ROOT r = f32[] reduce(sel, c_neg_inf), dimensions={0}, to_apply=%max_reducer
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, ReductionIdentityOnlyReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule reduction_identity_only
%max_reducer (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum = f32[] maximum(%x, %y)
}

ENTRY main {
  data = f32[10] parameter(0)
  c_neg_inf = f32[] constant(-inf)
  // c_neg_inf is only used as reduction init, not injected into tensor data.
  ROOT r = f32[] reduce(data, c_neg_inf), dimensions={0}, to_apply=%max_reducer
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, NonDataOperandSelectReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule non_data_operand_select
ENTRY main {
  c_nan = pred[] constant(false)
  // Constant is used as predicate condition (not tensor data)
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  b_pred = pred[10] broadcast(c_nan), dimensions={}
  c_dummy = f32[] constant(inf)
  // c_dummy is unused, module contains inf but does not reach root.
  ROOT sel = f32[10] select(b_pred, p0, p1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest,
     MaskedReductionWithInfNanProducingOpRespectsRejectOption) {
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
                       ParseAndReturnUnverifiedModule(kHlo));
  // Default (reject_unconstrained_ops = false): allowed because dataflow
  // constraint propagation guarantees safe parameter domains (e.g. data >= 0).
  EXPECT_TRUE(IsIntentionalInfNan(*module));
  EXPECT_TRUE(
      IsIntentionalInfNan(*module, {.reject_unconstrained_ops = false}));

  // When reject_unconstrained_ops = true (legacy baseline without dataflow):
  // rejected because unconstrained inputs to sqrt may cause genuine NaN.
  EXPECT_FALSE(
      IsIntentionalInfNan(*module, {.reject_unconstrained_ops = true}));
}

TEST(HloInfNanIntentAnalyzerTest,
     MaskedReductionWithoutInfNanProducingOpsPassesBothModes) {
  const absl::string_view kHlo = R"hlo(
HloModule masked_reduction_no_inf_producing_ops
%max_reducer (x: f32[], y: f32[]) -> f32[] {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum = f32[] maximum(%x, %y)
}

ENTRY main {
  mask = pred[10] parameter(0)
  data = f32[10] parameter(1)
  c_neg_inf = f32[] constant(-inf)
  b_neg_inf = f32[10] broadcast(c_neg_inf), dimensions={}
  sel = f32[10] select(mask, data, b_neg_inf)
  ROOT r = f32[] reduce(sel, c_neg_inf), dimensions={0}, to_apply=%max_reducer
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(
      IsIntentionalInfNan(*module, {.reject_unconstrained_ops = false}));
  EXPECT_TRUE(IsIntentionalInfNan(*module, {.reject_unconstrained_ops = true}));
}

TEST(HloInfNanIntentAnalyzerTest, TupleWithFloatingElementPropagatesToRoot) {
  const absl::string_view kHlo = R"hlo(
HloModule tuple_floating_root
ENTRY main {
  c_inf = f32[] constant(inf)
  s_data = s32[5] parameter(0)
  ROOT t = (f32[], s32[5]) tuple(c_inf, s_data)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_TRUE(IsIntentionalInfNan(*module));
}

TEST(HloInfNanIntentAnalyzerTest, NonFloatingExtractionFromTupleReturnsFalse) {
  const absl::string_view kHlo = R"hlo(
HloModule non_floating_extraction
ENTRY main {
  c_inf = f32[] constant(inf)
  s_data = s32[5] parameter(0)
  t = (f32[], s32[5]) tuple(c_inf, s_data)
  ROOT extracted = s32[5] get-tuple-element(t), index=1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(kHlo));
  EXPECT_FALSE(IsIntentionalInfNan(*module));
}

}  // namespace
}  // namespace hlo_isolation
}  // namespace xla

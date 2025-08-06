/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/xla_test_backend_predicates.h"
#include "xla/array4d.h"
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace {

constexpr ErrorSpec kErrorSpec{0.001, 0.001};

using Bfloat16Test = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

TEST_F(Bfloat16Test, ScalarOperation) {
  XlaBuilder builder(TestName());
  auto x = ConstantR0<bfloat16>(&builder, static_cast<bfloat16>(2.0f));
  auto y = ConstantR0<bfloat16>(&builder, static_cast<bfloat16>(1.0f));
  Add(x, y);

  ComputeAndCompareR0<bfloat16>(&builder, static_cast<bfloat16>(3.0f), {},
                                kErrorSpec);
}

TEST_F(Bfloat16Test, LogOperation) {
  XlaBuilder builder(TestName());
  auto x = ConstantR0<bfloat16>(&builder, static_cast<bfloat16>(4.0f));
  Log(x);

  ComputeAndCompareR0<bfloat16>(&builder, static_cast<bfloat16>(1.387f), {},
                                ErrorSpec(0.01, 0.01));
}

TEST_F(Bfloat16Test, NegateScalarF16) {
  XlaBuilder builder(TestName());
  Neg(ConstantR0<bfloat16>(&builder, static_cast<bfloat16>(2.1f)));

  ComputeAndCompareR0<bfloat16>(&builder, static_cast<bfloat16>(-2.1f), {},
                                kErrorSpec);
}

// Disabled on interpreter since BatchNormExpander is not run by default on the
// interpreter backend.
TEST_F(Bfloat16Test, BatchNormTraining) {
  if (test::DeviceIs(test::kInterpreter)) {
    GTEST_SKIP();
  }
  const int kFeatureIndex = 2;
  XlaBuilder builder(TestName());

  auto operand = ConstantR4FromArray4D<bfloat16>(
      &builder,
      {{{{static_cast<bfloat16>(1.f)}, {static_cast<bfloat16>(2.f)}},
        {{static_cast<bfloat16>(3.f)}, {static_cast<bfloat16>(4.f)}}},
       {{{static_cast<bfloat16>(5.f)}, {static_cast<bfloat16>(6.f)}},
        {{static_cast<bfloat16>(7.f)}, {static_cast<bfloat16>(8.f)}}}});

  auto scale = ConstantR1<bfloat16>(
      &builder, {static_cast<bfloat16>(2.0f), static_cast<bfloat16>(3.0f)});

  auto offset = ConstantR1<bfloat16>(
      &builder, {static_cast<bfloat16>(1.0f), static_cast<bfloat16>(2.0f)});

  BatchNormTraining(operand, scale, offset, /*epsilon=*/0.001, kFeatureIndex);

  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR4<bfloat16>(
           {{{{static_cast<bfloat16>(-1.6875f)},
              {static_cast<bfloat16>(-2.04f)}},
             {{static_cast<bfloat16>(0.105f)}, {static_cast<bfloat16>(0.66f)}}},
            {{{static_cast<bfloat16>(1.89f)}, {static_cast<bfloat16>(3.35f)}},
             {{static_cast<bfloat16>(3.7f)}, {static_cast<bfloat16>(6.04f)}}}}),
       LiteralUtil::CreateR1<bfloat16>(
           {static_cast<bfloat16>(4), static_cast<bfloat16>(5)}),
       LiteralUtil::CreateR1<bfloat16>(
           {static_cast<bfloat16>(5), static_cast<bfloat16>(5)})});

  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.01, 0.02));
}

// Disabled on interpreter since BatchNormExpander is not run by default on the
// interpreter backend.
TEST_F(Bfloat16Test, BatchNormGrad) {
  if (test::DeviceIs(test::kInterpreter)) {
    GTEST_SKIP();
  }
  const int kFeatureIndex = 2;
  XlaBuilder builder(TestName());

  auto operand = ConstantR4FromArray4D<bfloat16>(
      &builder, Array4D<bfloat16>(2, 2, 2, 1, static_cast<bfloat16>(0.0f)));

  auto scale = ConstantR1<bfloat16>(
      &builder, {static_cast<bfloat16>(1.0f), static_cast<bfloat16>(1.0f)});

  auto mean = ConstantR1<bfloat16>(
      &builder, {static_cast<bfloat16>(0.0f), static_cast<bfloat16>(0.0f)});

  auto var = ConstantR1<bfloat16>(
      &builder, {static_cast<bfloat16>(1.0f), static_cast<bfloat16>(1.0f)});

  auto grad_output = ConstantR4FromArray4D<bfloat16>(
      &builder,
      {{{{static_cast<bfloat16>(1.f)}, {static_cast<bfloat16>(2.f)}},
        {{static_cast<bfloat16>(3.f)}, {static_cast<bfloat16>(4.f)}}},
       {{{static_cast<bfloat16>(5.f)}, {static_cast<bfloat16>(6.f)}},
        {{static_cast<bfloat16>(7.f)}, {static_cast<bfloat16>(8.f)}}}});

  BatchNormGrad(operand, scale, mean, var, grad_output,
                /*epsilon=*/0.0, kFeatureIndex);

  auto expected = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR4<bfloat16>(
           {{{{static_cast<bfloat16>(-3.f)}, {static_cast<bfloat16>(-3.f)}},
             {{static_cast<bfloat16>(-1.f)}, {static_cast<bfloat16>(-1.f)}}},
            {{{static_cast<bfloat16>(1.f)}, {static_cast<bfloat16>(1.f)}},
             {{static_cast<bfloat16>(3.f)}, {static_cast<bfloat16>(3.f)}}}}),
       LiteralUtil::CreateR1<bfloat16>(
           {static_cast<bfloat16>(0), static_cast<bfloat16>(0)}),
       LiteralUtil::CreateR1<bfloat16>(
           {static_cast<bfloat16>(16), static_cast<bfloat16>(20)})});

  ComputeAndCompareTuple(&builder, expected, {}, ErrorSpec(0.01));
}

}  // namespace
}  // namespace xla

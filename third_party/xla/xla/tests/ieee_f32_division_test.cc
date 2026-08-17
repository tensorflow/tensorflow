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

#include <limits>
#include <vector>

#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using IeeeF32DivisionTest =
    ClientLibraryTestRunnerMixin<HloPjRtInterpreterReferenceMixin<HloTestBase>>;

// Regression test for https://github.com/openxla/xla/issues/37181.
//
// The NVPTX backend used to override LLVM's IEEE-754 compliant f32 division
// (div.rnd) with div.full, which has up to 2 ULP of error. This made e.g.
// x / x != 1.0 for many float32 values on GPU (measured error: -5.96e-08 for
// 3163.328613). These tests assert IEEE-754 identity properties that are
// broken by the div.full override. They use parameters (not constants) so the
// division is actually executed by the backend rather than folded at compile
// time, and fast math is disabled so the property is not simplified away.
TEST_F(IeeeF32DivisionTest, DivBySelfIsOne) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  XlaOp param;
  const Literal param_data = CreateR1Parameter<float>(
      {3163.328613f, 1.0f, -1.0f, 3.14159f, 1e10f, 1e-10f,
       std::numeric_limits<float>::max(), std::numeric_limits<float>::min()},
      /*parameter_number=*/0, /*name=*/"param", /*builder=*/&builder,
      /*data_handle=*/&param);
  Div(param, param);

  std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f,
                                 1.0f, 1.0f, 1.0f, 1.0f};
  ComputeAndCompareR1<float>(&builder, expected, {param_data}, ErrorSpec(0));
}

TEST_F(IeeeF32DivisionTest, DivByOneIsIdentity) {
  SetFastMathDisabled(true);
  XlaBuilder builder(TestName());
  XlaOp param;
  const Literal param_data = CreateR1Parameter<float>(
      {3163.328613f, 1.0f, -1.0f, 3.14159f, 1e10f, 1e-10f,
       std::numeric_limits<float>::max(), std::numeric_limits<float>::min()},
      /*parameter_number=*/0, /*name=*/"param", /*builder=*/&builder,
      /*data_handle=*/&param);
  Div(param, ConstantR0<float>(&builder, 1.0f));

  std::vector<float> expected = {3163.328613f,
                                 1.0f,
                                 -1.0f,
                                 3.14159f,
                                 1e10f,
                                 1e-10f,
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::min()};
  ComputeAndCompareR1<float>(&builder, expected, {param_data}, ErrorSpec(0));
}

}  // namespace
}  // namespace xla

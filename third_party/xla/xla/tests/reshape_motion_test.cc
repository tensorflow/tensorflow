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

#include <cstdint>

#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using ReshapeMotionTest = ClientLibraryTestRunnerMixin<
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>>;

TEST_F(ReshapeMotionTest, ElementwiseOfReshapesWithNonSameInputShapes) {
  XlaBuilder builder(TestName());
  auto a = ConstantR2<int32_t>(&builder, {{2, 3, 5}, {7, 11, 13}});
  auto b = ConstantR2<int32_t>(&builder, {{17, 19}, {23, 29}, {31, 37}});
  auto c = Reshape(a, {6});
  auto d = Reshape(b, {6});
  Mul(c, d);

  ComputeAndCompareR1<int32_t>(&builder, {34, 57, 115, 203, 341, 481}, {});
}

}  // namespace
}  // namespace xla

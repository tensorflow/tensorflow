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

#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tests/client_library_test_base.h"

namespace xla {
namespace {

using Atan2Test = ClientLibraryTestBase;

TEST_F(Atan2Test, atan2) {
  XlaBuilder builder("atan2 with special and non-special float values");
  float infinity = std::numeric_limits<float>::infinity();
  float qNaN = std::numeric_limits<float>::quiet_NaN();
  float sNaN = std::numeric_limits<float>::signaling_NaN();
  auto x = ConstantR1<float>(&builder,
                             {infinity, infinity, infinity, qNaN, qNaN, sNaN});
  auto y =
      ConstantR1<float>(&builder, {1.0, -1.0, qNaN, 1.0, infinity, infinity});
  Atan2(x, y);
  float pi_over_2 = 1.5708;
  std::vector<float> expected = {pi_over_2, pi_over_2, qNaN, qNaN, qNaN, qNaN};
  ComputeAndCompareR1<float>(&builder, expected, {}, ErrorSpec(0.0001));
}

}  // namespace
}  // namespace xla

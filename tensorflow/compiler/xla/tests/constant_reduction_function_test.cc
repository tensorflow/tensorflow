/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This demonstrates how to use hlo_test_base to create textual IR based
// testcases.

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

using std::nullopt;

class ConstantReductionFunctionTest : public HloTestBase {};

TEST_F(ConstantReductionFunctionTest, Bool) {
  const std::string& hlo_string = R"(
HloModule jit_f__2.10

reduction_computation__3.4 {
  parameter.5 = pred[] parameter(0)
  parameter.6 = pred[] parameter(1)
  constant.7 = pred[] constant(false)
  ROOT constant.8 = pred[] constant(true)
}

ENTRY jit_f__2.10 {
  constant.2 = pred[] constant(false)
  parameter.1 = pred[24,1,1,5]{3,2,1,0} parameter(0)
  constant.3 = pred[] constant(false)
  ROOT reduce.9 = pred[24,1,1]{2,1,0} reduce(parameter.1, constant.3), dimensions={3}, to_apply=reduction_computation__3.4
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, nullopt));
}

}  // namespace
}  // namespace xla

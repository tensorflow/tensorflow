/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class GetDimensionSizeTest : public HloTestBase {};

// Test that the interpreter can correctly compute get_dimension_size.
TEST_F(GetDimensionSizeTest, DoIt) {
  const char* const kModuleStr = R"(
HloModule a_inference_call_110__.55

ENTRY %a_inference_call_110__.55 (arg0.1: f32[1,8], arg1.2: f32[8], arg2.3: f32[8]) -> s32[] {
  %constant.37 = f32[] constant(1e-12)
  %broadcast.38 = f32[1,1]{1,0} broadcast(f32[] %constant.37), dimensions={}
  %arg0.1 = f32[1,8]{1,0} parameter(0), parameter_replication={false}
  %reshape.4 = f32[1,8]{1,0} reshape(f32[1,8]{1,0} %arg0.1)
  %convert.5 = f32[1,8]{1,0} convert(f32[1,8]{1,0} %reshape.4)
  %constant.6 = f32[] constant(0)
  %convert.7 = f32[] convert(f32[] %constant.6)
  ROOT %get-dimension-size.13 = s32[] get-dimension-size(f32[1,8]{1,0} %convert.5), dimensions={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}));
}

}  // anonymous namespace
}  // namespace xla

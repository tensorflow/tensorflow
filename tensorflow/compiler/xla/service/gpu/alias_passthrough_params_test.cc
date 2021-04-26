/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/alias_passthrough_params.h"

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu {

class AliasPassthroughParamsTest : public HloTestBase {};

TEST_F(AliasPassthroughParamsTest, AliasPassThroughParams) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    p0 = f16[2048,1024] parameter(0)
    p1 = f16[2048,1024] parameter(1)
    sum = f16[2048,1024] add(p0, p1)
    ROOT root = (f16[2048,1024], f16[2048,1024], f16[2048,1024]) tuple(p0, sum, p1)
  })")
                    .ValueOrDie();
  EXPECT_TRUE(AliasPassthroughParams().Run(module.get()).ValueOrDie());
  const auto& alias_config = module->input_output_alias_config();
  EXPECT_EQ(0, alias_config.GetAliasedParameter({0})->parameter_number);
  EXPECT_FALSE(alias_config.OutputHasAlias({1}));
  EXPECT_EQ(1, alias_config.GetAliasedParameter({2})->parameter_number);
}

TEST_F(AliasPassthroughParamsTest, DoNotAliasPassThroughParamsMoreThanOnce) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    p0 = f16[2048,1024] parameter(0)
    ROOT root = (f16[2048,1024], f16[2048,1024]) tuple(p0, p0)
  })")
                    .ValueOrDie();
  EXPECT_TRUE(AliasPassthroughParams().Run(module.get()).ValueOrDie());
  const auto& alias_config = module->input_output_alias_config();
  EXPECT_EQ(0, alias_config.GetAliasedParameter({0})->parameter_number);
  EXPECT_FALSE(alias_config.OutputHasAlias({1}));
}

}  // namespace gpu
}  // namespace xla

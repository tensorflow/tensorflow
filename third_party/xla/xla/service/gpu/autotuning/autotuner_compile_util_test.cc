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

#include "xla/service/gpu/autotuning/autotuner_compile_util.h"

#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using AutotunerCompileUtilTest = HloTestBase;

TEST_F(AutotunerCompileUtilTest, VerifyOutputNotATuple) {
  constexpr absl::string_view kHlo = R"(
HloModule hlo
ENTRY main {
  p0 = f32[2,2] parameter(0)
  p1 = f32[4,4] parameter(1)
  p2 = f32[6,6] parameter(2)
  ROOT root = f32[1,2,3] custom-call(p0, p1, p2), custom_call_target="fake"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));

  AutotuneConfig autotune_config{DeviceConfig{executors.at(0), nullptr},
                                 GetDebugOptionsForTest()};

  auto& root = *module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputs));

  EXPECT_EQ(rzb.input_shapes().size(), 3);
  EXPECT_EQ(rzb.input_buffers().size(), 3);
  EXPECT_EQ(rzb.output_buffers().size(), 0);
  EXPECT_NE(rzb.output_shape(), root.shape());

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb2,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputsAllOutputs));

  EXPECT_EQ(rzb2.input_shapes().size(), 3);
  EXPECT_EQ(rzb2.input_buffers().size(), 3);
  EXPECT_EQ(rzb2.output_buffers().size(), 1);
  EXPECT_EQ(rzb2.output_shape(), root.shape());

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb3,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputsOutputsNoScratch));

  EXPECT_EQ(rzb3.input_shapes().size(), 3);
  EXPECT_EQ(rzb3.input_buffers().size(), 3);
  EXPECT_EQ(rzb3.output_buffers().size(), 1);
  EXPECT_EQ(rzb3.output_shape(), root.shape());
}

TEST_F(AutotunerCompileUtilTest, VerifyOutputTupleOneElement) {
  constexpr absl::string_view kHlo = R"(
HloModule hlo
ENTRY main {
  p0 = f32[2,2] parameter(0)
  p1 = f32[4,4] parameter(1)
  p2 = f32[6,6] parameter(2)
  ROOT root = (f32[1,2,3]) custom-call(p0, p1, p2), custom_call_target="fake"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));

  AutotuneConfig autotune_config{DeviceConfig{executors.at(0), nullptr},
                                 GetDebugOptionsForTest()};

  auto& root = *module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputs));

  EXPECT_EQ(rzb.input_shapes().size(), 3);
  EXPECT_EQ(rzb.input_buffers().size(), 3);
  EXPECT_EQ(rzb.output_buffers().size(), 0);
  EXPECT_NE(rzb.output_shape(), root.shape());

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb2,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputsAllOutputs));

  EXPECT_EQ(rzb2.input_shapes().size(), 3);
  EXPECT_EQ(rzb2.input_buffers().size(), 3);
  EXPECT_EQ(rzb2.output_buffers().size(), 1);
  EXPECT_FALSE(rzb2.output_shape().IsTuple());
  EXPECT_EQ(rzb2.output_shape(), root.shape().tuple_shapes(0));

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb3,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputsOutputsNoScratch));

  EXPECT_EQ(rzb3.input_shapes().size(), 3);
  EXPECT_EQ(rzb3.input_buffers().size(), 3);
  EXPECT_EQ(rzb3.output_buffers().size(), 0);
}

TEST_F(AutotunerCompileUtilTest, VerifyOutputTupleTwoElements) {
  constexpr absl::string_view kHlo = R"(
HloModule hlo
ENTRY main {
  p0 = f32[2,2] parameter(0)
  p1 = f32[4,4] parameter(1)
  p2 = f32[6,6] parameter(2)
  ROOT root = (f32[1,2,3], u8[1,2]) custom-call(p0, p1, p2), custom_call_target="fake"
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));

  AutotuneConfig autotune_config{DeviceConfig{executors.at(0), nullptr},
                                 GetDebugOptionsForTest()};

  auto& root = *module->entry_computation()->root_instruction();

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputs));

  EXPECT_EQ(rzb.input_shapes().size(), 3);
  EXPECT_EQ(rzb.input_buffers().size(), 3);
  EXPECT_EQ(rzb.output_buffers().size(), 0);
  EXPECT_NE(rzb.output_shape(), root.shape());

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb2,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputsAllOutputs));

  EXPECT_EQ(rzb2.input_shapes().size(), 3);
  EXPECT_EQ(rzb2.input_buffers().size(), 3);
  EXPECT_EQ(rzb2.output_buffers().size(), 2);
  EXPECT_TRUE(rzb2.output_shape().IsTuple());
  EXPECT_EQ(rzb2.output_shape(), root.shape());

  TF_ASSERT_OK_AND_ASSIGN(RedzoneBuffers rzb3,
                          RedzoneBuffers::FromInstruction(
                              root, autotune_config, GetDebugOptionsForTest(),
                              RedzoneBuffers::kAllInputsOutputsNoScratch));

  EXPECT_EQ(rzb3.input_shapes().size(), 3);
  EXPECT_EQ(rzb3.input_buffers().size(), 3);
  EXPECT_EQ(rzb3.output_buffers().size(), 1);
  EXPECT_FALSE(rzb3.output_shape().IsTuple());
  EXPECT_EQ(rzb3.output_shape(), root.shape().tuple_shapes(0));
}

}  // namespace
}  // namespace xla::gpu

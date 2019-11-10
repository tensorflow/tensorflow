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

#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"

#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {
class HloInputOutputAliasConfigTest : public HloTestBase {
 protected:
  void expect_aliased(const ShapeIndex& output_index, int64 param_number,
                      const ShapeIndex& param_index,
                      const HloInputOutputAliasConfig& config) {
    absl::optional<ShapeIndex> aliased_output =
        config.GetAliasedOutput(param_number, param_index);

    EXPECT_TRUE(aliased_output);
    EXPECT_EQ(aliased_output.value(), output_index);

    absl::optional<HloInputOutputAliasConfig::Alias> aliased_param =
        config.GetAliasedParameter(output_index);

    EXPECT_TRUE(aliased_param);
    EXPECT_EQ(aliased_param->parameter_number, param_number);
    EXPECT_EQ(aliased_param->parameter_index, param_index);
  }

  void expect_not_aliased(const ShapeIndex& output_index, int64 param_number,
                          const ShapeIndex& param_index,
                          const HloInputOutputAliasConfig& config) {
    absl::optional<ShapeIndex> aliased_output =
        config.GetAliasedOutput(param_number, param_index);

    EXPECT_FALSE(aliased_output && aliased_output == output_index);

    absl::optional<HloInputOutputAliasConfig::Alias> aliased_param =
        config.GetAliasedParameter(output_index);

    EXPECT_FALSE(aliased_param &&
                 aliased_param->parameter_number == param_number &&
                 aliased_param->parameter_index == param_index);
  }
};

TEST_F(HloInputOutputAliasConfigTest, SimpleAliasing) {
  const string module_str = R"(
HloModule TEST

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT root = (f32[], f32[]) tuple(%a, %b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  HloInputOutputAliasConfig config(
      module->entry_computation()->root_instruction()->shape());

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/1,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  expect_aliased(/*output_index=*/{0}, /*param_number=*/1,
                 /*param_index=*/{}, config);

  expect_not_aliased(/*output_index=*/{1}, /*param_number=*/1,
                     /*param_index=*/{}, config);

  expect_not_aliased(/*output_index=*/{0}, /*param_number=*/0,
                     /*param_index=*/{}, config);
}

TEST_F(HloInputOutputAliasConfigTest, SimpleAliasingWithTupleInput) {
  const string module_str = R"(
HloModule TEST

ENTRY main {
  param = (f32[], f32[]) parameter(0)
  gte1 = f32[] get-tuple-element(%param), index=0
  gte2 = f32[] get-tuple-element(%param), index=1
  ROOT root = (f32[], f32[]) tuple(%gte1, %gte2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  HloInputOutputAliasConfig config(
      module->entry_computation()->root_instruction()->shape());

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0,
      /*param_index=*/{0},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0,
      /*param_index=*/{1},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  expect_aliased(/*output_index=*/{0}, /*param_number=*/0,
                 /*param_index=*/{0}, config);

  expect_aliased(/*output_index=*/{1}, /*param_number=*/0,
                 /*param_index=*/{1}, config);

  expect_not_aliased(/*output_index=*/{1}, /*param_number=*/1,
                     /*param_index=*/{}, config);

  expect_not_aliased(/*output_index=*/{0}, /*param_number=*/0,
                     /*param_index=*/{}, config);
}

TEST_F(HloInputOutputAliasConfigTest, InputDoNotAliasTwice) {
  const string module_str = R"(
HloModule TEST

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT root = (f32[], f32[]) tuple(%a, %b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  HloInputOutputAliasConfig config(
      module->entry_computation()->root_instruction()->shape());

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  ASSERT_IS_NOT_OK(config.Verify(*module, [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape);
  }));
}

TEST_F(HloInputOutputAliasConfigTest, SizesMustMatch) {
  const string module_str = R"(
HloModule TEST

ENTRY main {
  a = f32[] parameter(0)
  b = f32[4096] parameter(1)
  ROOT root = (f32[], f32[4096]) tuple(%a, %b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  HloInputOutputAliasConfig config(
      module->entry_computation()->root_instruction()->shape());

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  ASSERT_IS_NOT_OK(config.Verify(*module, [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape);
  }));
}

TEST_F(HloInputOutputAliasConfigTest, OutputDoNotAliasTwice) {
  const string module_str = R"(
HloModule TEST

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT root = (f32[], f32[]) tuple(%a, %b)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));

  HloInputOutputAliasConfig config(
      module->entry_computation()->root_instruction()->shape());

  TF_ASSERT_OK(config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));

  ASSERT_IS_NOT_OK(config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/1,
      /*param_index=*/{},
      /*kind=*/HloInputOutputAliasConfig::AliasKind::kUserAlias));
}
}  // namespace
}  // namespace xla

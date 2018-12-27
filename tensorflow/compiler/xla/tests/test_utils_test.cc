/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/local_client_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

// A test fixture is used because we need a client for our computation builder.
class TestUtilsTest : public LocalClientTestBase {};

XLA_TEST_F(TestUtilsTest, UnusedParam) {
  XlaBuilder builder(TestName());
  // Make the reduction lambda.
  Shape single_float = ShapeUtil::MakeShape(F32, {});
  Parameter(&builder, 0, single_float, "unused");
  Parameter(&builder, 1, single_float, "used");
  auto computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  // Make the reduction.
  Shape pair_float = ShapeUtil::MakeShape(F32, {2});
  Reduce(Parameter(&builder, 0, pair_float, "operand"),
         Parameter(&builder, 1, single_float, "init"),
         computation_status.ValueOrDie(), {0});
  computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  auto executable_status = local_client_->Compile(
      computation_status.ValueOrDie(), {&pair_float, &single_float},
      ExecutableBuildOptions());
  TF_ASSERT_OK(executable_status.status());
  HloModule& module = const_cast<HloModule&>(
      executable_status.ValueOrDie()->executable()->module());
  TF_ASSERT_OK(MakeFakeArguments(&module).status());
}

XLA_TEST_F(TestUtilsTest, Token) {
  auto module = ParseHloString(
                    R"(HloModule outfeed_module

    ENTRY InfeedToOutfeed {
      token0 = token[] parameter(0)
      infeed = ((u32[3]{0}, pred[]), token[]) infeed(token0)
      infeed.data = (u32[3]{0}, pred[]) get-tuple-element(infeed), index=0
      outfeed = token[] outfeed(infeed.data, token0)
      ROOT infeed.1 = ((u32[3]{0}, pred[]), token[]) infeed(token0)
      infeed.1.data = (u32[3]{0}, pred[]) get-tuple-element(infeed.1), index=0
      infeed.1.token = token[] get-tuple-element(infeed.1), index=1
      outfeed.1 = token[] outfeed(infeed.1.data, infeed.1.token)
    })")
                    .ValueOrDie();
  TF_ASSERT_OK(MakeFakeArguments(module.get()).status());
}

XLA_TEST_F(TestUtilsTest, MultipleIndexSpacesForDynamicSlices) {
  auto module = ParseHloString(
                    R"(HloModule index_space_module

    ENTRY IndexSpace {
      index_param = s32[3]{0} parameter(0)
      array_param.1 = f32[123,4,789]{0,1,2} parameter(1)
      array_param.2 = f32[3,3000,5]{0,1,2} parameter(2)
      dynamic-slice.1 = f32[1,2,3] dynamic-slice(array_param.1, index_param), dynamic_slice_sizes={1,2,3}
      ROOT dynamic-slice.2 = f32[3,2,2] dynamic-slice(array_param.2, index_param), dynamic_slice_sizes={3,2,2}
    })")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 3);
  const Literal& index_arg = args[0];

  EXPECT_EQ(index_arg.Get<int32>({0}), 0);

  EXPECT_GE(index_arg.Get<int32>({1}), 0);
  EXPECT_LE(index_arg.Get<int32>({1}), 2);

  EXPECT_GE(index_arg.Get<int32>({2}), 0);
  EXPECT_LE(index_arg.Get<int32>({2}), 3);
}

XLA_TEST_F(TestUtilsTest, MultipleIndexSpacesForDynamicUpdateSlices) {
  auto module = ParseHloString(
                    R"(HloModule index_space_module

    ENTRY IndexSpace {
      index_param = s32[3]{0} parameter(0)
      array_param.1 = f32[123,4,789]{0,1,2} parameter(1)
      array_param.2 = f32[3,3000,5]{0,1,2} parameter(2)
      update_param.1 = f32[1,2,3]{0,1,2} parameter(3)
      update_param.2 = f32[3,2,2]{0,1,2} parameter(4)

      dynamic-update-slice.1 = f32[123,4,789] dynamic-update-slice(array_param.1, update_param.1, index_param)
      ROOT dynamic-update-slice.2 = f32[3,3000,5] dynamic-update-slice(array_param.2, update_param.2, index_param)
    })")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 5);
  const Literal& index_arg = args[0];

  EXPECT_EQ(index_arg.Get<int32>({0}), 0);

  EXPECT_GE(index_arg.Get<int32>({1}), 0);
  EXPECT_LE(index_arg.Get<int32>({1}), 2);

  EXPECT_GE(index_arg.Get<int32>({2}), 0);
  EXPECT_LE(index_arg.Get<int32>({2}), 3);
}

XLA_TEST_F(TestUtilsTest, NoDuplicatesFloats) {
  // Inputs which are sort keys in key/value sorts should have no duplicates.
  auto module = ParseHloString(R"(
HloModule sort.148.1589

ENTRY %sort.148.1589 (parameter.0: f32[1048576], parameter.1: s32[1048576]) -> (f32[1048576], s32[1048576]) {
  %parameter.0 = f32[1048576]{0} parameter(0)
  %parameter.1 = s32[1048576]{0} parameter(1)
  ROOT %sort.148.1589 = (f32[1048576]{0}, s32[1048576]{0}) sort(f32[1048576]{0} %parameter.0, s32[1048576]{0} %parameter.1), dimensions={0}
}
)")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  const Literal& key_arg = args[0];

  absl::flat_hash_set<uint32> key_set;
  for (const float& value : key_arg.data<float>()) {
    EXPECT_TRUE(key_set.insert(absl::bit_cast<uint32>(value)).second);
  }
}

XLA_TEST_F(TestUtilsTest, NoDuplicatesInt32) {
  // Inputs which are sort keys in key/value sorts should have no duplicates.
  auto module = ParseHloString(R"(
HloModule sort.148.1589

ENTRY %sort.148.1589 (parameter.0: s32[1048576], parameter.1: s32[1048576]) -> (s32[1048576], s32[1048576]) {
  %parameter.0 = s32[1048576]{0} parameter(0)
  %parameter.1 = s32[1048576]{0} parameter(1)
  ROOT %sort.148.1589 = (s32[1048576]{0}, s32[1048576]{0}) sort(s32[1048576]{0} %parameter.0, s32[1048576]{0} %parameter.1), dimensions={0}
}
)")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  const Literal& key_arg = args[0];

  absl::flat_hash_set<int32> key_set;
  for (const int32& value : key_arg.data<int32>()) {
    EXPECT_TRUE(key_set.insert(absl::bit_cast<uint32>(value)).second);
  }
}

XLA_TEST_F(TestUtilsTest, NoDuplicatesBfloat16) {
  // Inputs which are sort keys in key/value sorts should have no duplicates.
  auto module = ParseHloString(R"(
HloModule sort, is_scheduled=true

ENTRY %sort. (parameter.0: bf16[2,1452], parameter.1: s32[2,1452]) -> (bf16[2,1452], s32[2,1452]) {
  %parameter.0 = bf16[2,1452]{1,0} parameter(0)
  %parameter.1 = s32[2,1452]{1,0} parameter(1)
  ROOT %sort = (bf16[2,1452]{1,0}, s32[2,1452]{1,0}) sort(bf16[2,1452]{1,0} %parameter.0, s32[2,1452]{1,0} %parameter.1), dimensions={1}
}
)")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  const Literal& key_arg = args[0];

  absl::flat_hash_set<uint16> key_set;
  for (const bfloat16& value : key_arg.data<bfloat16>()) {
    EXPECT_TRUE(key_set.insert(absl::bit_cast<uint16>(value)).second);
  }
}

XLA_TEST_F(TestUtilsTest, MakeFakeArgumentsR0InputToDynamicSlice) {
  auto module = ParseHloString(R"(
HloModule Test

ENTRY %module (parameter.0: s32[], parameter.1: f32[20,20]) -> f32[] {
  %parameter.1 = f32[20,20]{1,0} parameter(1)
  %constant.1 = s32[1]{0} constant({0})
  %parameter.0 = s32[] parameter(0)
  %bitcast.3 = s32[1]{0} bitcast(s32[] %parameter.0)
  %concatenate.1 = s32[2]{0} concatenate(s32[1]{0} %constant.1, s32[1]{0} %bitcast.3), dimensions={0}
  %dynamic-slice.2 = f32[20,1]{1,0} dynamic-slice(f32[20,20]{1,0} %parameter.1, s32[2]{0} %concatenate.1), dynamic_slice_sizes={20,1}
  %bitcast.4 = f32[20]{0} bitcast(f32[20,1]{1,0} %dynamic-slice.2)
  %dynamic-slice.3 = f32[1]{0} dynamic-slice(f32[20]{0} %bitcast.4, s32[1]{0} %bitcast.3), dynamic_slice_sizes={1}
  ROOT %bitcast.5 = f32[] bitcast(f32[1]{0} %dynamic-slice.3)
}
)")
                    .ValueOrDie();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  EXPECT_TRUE(ShapeUtil::Equal(args[0].shape(), ShapeUtil::MakeShape(S32, {})))
      << ShapeUtil::HumanString(args[0].shape());
  EXPECT_TRUE(
      ShapeUtil::Equal(args[1].shape(), ShapeUtil::MakeShape(F32, {20, 20})))
      << ShapeUtil::HumanString(args[1].shape());
}

}  // namespace
}  // namespace xla

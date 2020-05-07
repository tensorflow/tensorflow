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

  TF_ASSERT_OK_AND_ASSIGN(
      auto executables, local_client_->Compile(computation_status.ValueOrDie(),
                                               {&pair_float, &single_float},
                                               ExecutableBuildOptions()));
  HloModule& module =
      const_cast<HloModule&>(executables[0]->executable()->module());
  TF_ASSERT_OK(MakeFakeArguments(&module).status());
}

XLA_TEST_F(TestUtilsTest, MultipleIndexSpacesForDynamicSlices) {
  auto module = ParseAndReturnVerifiedModule(
                    R"(HloModule index_space_module

    ENTRY IndexSpace {
      index_param.0 = s32[] parameter(0)
      index_param.1 = s32[] parameter(1)
      index_param.2 = s32[] parameter(2)
      array_param.1 = f32[123,4,789]{0,1,2} parameter(3)
      array_param.2 = f32[3,3000,5]{0,1,2} parameter(4)
      dynamic-slice.1 = f32[1,2,3] dynamic-slice(array_param.1, index_param.0, index_param.1, index_param.2), dynamic_slice_sizes={1,2,3}
      ROOT dynamic-slice.2 = f32[3,2,2] dynamic-slice(array_param.2, index_param.0, index_param.1, index_param.2), dynamic_slice_sizes={3,2,2}
    })")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 5);

  EXPECT_GE(args[0].Get<int32>({}), -1);
  EXPECT_LE(args[0].Get<int32>({}), 1);

  EXPECT_GE(args[1].Get<int32>({}), -1);
  EXPECT_LE(args[1].Get<int32>({}), 2);

  EXPECT_GE(args[2].Get<int32>({}), -1);
  EXPECT_LE(args[2].Get<int32>({}), 3);
}

XLA_TEST_F(TestUtilsTest, MultipleIndexSpacesForDynamicUpdateSlices) {
  auto module = ParseAndReturnVerifiedModule(
                    R"(HloModule index_space_module

    ENTRY IndexSpace {
      index_param.0 = s32[] parameter(0)
      index_param.1 = s32[] parameter(1)
      index_param.2 = s32[] parameter(2)
      array_param.1 = f32[123,4,789]{0,1,2} parameter(3)
      array_param.2 = f32[3,3000,5]{0,1,2} parameter(4)
      update_param.1 = f32[1,2,3]{0,1,2} parameter(5)
      update_param.2 = f32[3,2,2]{0,1,2} parameter(6)

      dynamic-update-slice.1 = f32[123,4,789] dynamic-update-slice(array_param.1, update_param.1, index_param.0, index_param.1, index_param.2)
      ROOT dynamic-update-slice.2 = f32[3,3000,5] dynamic-update-slice(array_param.2, update_param.2, index_param.0, index_param.1, index_param.2)
    })")
                    .ValueOrDie();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 7);

  EXPECT_GE(args[0].Get<int32>({}), -1);
  EXPECT_LE(args[0].Get<int32>({}), 1);

  EXPECT_GE(args[1].Get<int32>({}), -1);
  EXPECT_LE(args[1].Get<int32>({}), 2);

  EXPECT_GE(args[2].Get<int32>({}), -1);
  EXPECT_LE(args[2].Get<int32>({}), 3);
}

XLA_TEST_F(TestUtilsTest, NoDuplicatesFloats) {
  // Inputs which are sort keys in key/value sorts should have no duplicates.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule sort.148.1589

compare {
  p.0.lhs = f32[] parameter(0)
  p.0.rhs = f32[] parameter(1)
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY %sort.148.1589 (parameter.0: f32[1048576], parameter.1: s32[1048576]) -> (f32[1048576], s32[1048576]) {
  %parameter.0 = f32[1048576]{0} parameter(0)
  %parameter.1 = s32[1048576]{0} parameter(1)
  ROOT %sort.148.1589 = (f32[1048576]{0}, s32[1048576]{0}) sort(f32[1048576]{0} %parameter.0, s32[1048576]{0} %parameter.1), dimensions={0}, to_apply=compare
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
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule sort.148.1589

compare {
  p.0.lhs = s32[] parameter(0)
  p.0.rhs = s32[] parameter(1)
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY %sort.148.1589 (parameter.0: s32[1048576], parameter.1: s32[1048576]) -> (s32[1048576], s32[1048576]) {
  %parameter.0 = s32[1048576]{0} parameter(0)
  %parameter.1 = s32[1048576]{0} parameter(1)
  ROOT %sort.148.1589 = (s32[1048576]{0}, s32[1048576]{0}) sort(s32[1048576]{0} %parameter.0, s32[1048576]{0} %parameter.1), dimensions={0}, to_apply=compare
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
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule sort, is_scheduled=true

compare {
  p.0.lhs = bf16[] parameter(0)
  p.0.rhs = bf16[] parameter(1)
  p.1.lhs = s32[] parameter(2)
  p.1.rhs = s32[] parameter(3)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY %sort. (parameter.0: bf16[2,1452], parameter.1: s32[2,1452]) -> (bf16[2,1452], s32[2,1452]) {
  %parameter.0 = bf16[2,1452]{1,0} parameter(0)
  %parameter.1 = s32[2,1452]{1,0} parameter(1)
  ROOT %sort = (bf16[2,1452]{1,0}, s32[2,1452]{1,0}) sort(bf16[2,1452]{1,0} %parameter.0, s32[2,1452]{1,0} %parameter.1), dimensions={1}, to_apply=compare
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
  auto module = ParseAndReturnVerifiedModule(R"(
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

XLA_TEST_F(TestUtilsTest, MakeFakeArgumentsForGather) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule Test

ENTRY %module(parameter.0: f32[200,100,300], parameter.1: s32[10,2]) ->
                                                          f32[10,300] {
  %parameter.0 = f32[200,100,300] parameter(0)
  %parameter.1 = s32[10,2] parameter(1)
  ROOT gather = f32[10,300] gather(f32[200,100,300] %parameter.0,
                                   s32[10,2] %parameter.1),
      offset_dims={1},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1,300}
}
)")
                    .ValueOrDie();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);

  const Shape& indices_shape = args[1].shape();
  EXPECT_TRUE(
      ShapeUtil::Equal(indices_shape, ShapeUtil::MakeShape(S32, {10, 2})))
      << ShapeUtil::HumanString(indices_shape);
  auto indices = args[1].data<int32>();
  for (const auto index : indices) {
    EXPECT_GE(index, -1);
    EXPECT_LE(index, 100);
  }
}

XLA_TEST_F(TestUtilsTest, MakeFakeArgumentsForScatter) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule Test

scatter_update (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  ROOT rhs = f32[] parameter(1)
}

ENTRY main {
  operand = f32[200,100,300] parameter(0)
  indices = s32[10,2] parameter(1)
  updates = f32[10,300] parameter(2)
  ROOT scatter = f32[200,100,300] scatter(operand, indices, updates),
    to_apply=scatter_update,
    update_window_dims={1},
    inserted_window_dims={0,1},
    scatter_dims_to_operand_dims={0,1},
    index_vector_dim=1
  }
)")
                    .ValueOrDie();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 3);

  const Shape& indices_shape = args[1].shape();
  EXPECT_TRUE(
      ShapeUtil::Equal(indices_shape, ShapeUtil::MakeShape(S32, {10, 2})))
      << ShapeUtil::HumanString(indices_shape);
  auto indices = args[1].data<int32>();
  for (const auto index : indices) {
    EXPECT_GE(index, -1);
    EXPECT_LE(index, 100);
  }
}

}  // namespace
}  // namespace xla

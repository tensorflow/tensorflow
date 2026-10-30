/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/tests/test_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/types.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

// A test fixture is used because we need a client for our computation builder.
class TestUtilsTest : public HloTestBase {};

TEST_F(TestUtilsTest, UnusedParam) {
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
         computation_status.value(), {0});
  computation_status = builder.Build();
  TF_ASSERT_OK(computation_status.status());

  ExecutionOptions execution_options;
  *execution_options.mutable_debug_options() =
      GetModuleConfigForTest().debug_options();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_ptr,
                          HloModuleFromXlaComputation(
                              computation_status.value(), execution_options));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<OpaqueExecutable> executable,
                          CreateExecutable(std::move(module_ptr), true));
  TF_ASSERT_OK_AND_ASSIGN(const HloModule* optimized_module,
                          test_runner().HloModuleFromWrapped(executable.get()));
  TF_ASSERT_OK(MakeFakeArguments(optimized_module).status());
}

TEST_F(TestUtilsTest, MultipleIndexSpacesForDynamicSlices) {
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
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 5);

  EXPECT_GE(args[0].Get<int32_t>({}), -1);
  EXPECT_LE(args[0].Get<int32_t>({}), 1);

  EXPECT_GE(args[1].Get<int32_t>({}), -1);
  EXPECT_LE(args[1].Get<int32_t>({}), 2);

  EXPECT_GE(args[2].Get<int32_t>({}), -1);
  EXPECT_LE(args[2].Get<int32_t>({}), 3);
}

TEST_F(TestUtilsTest, MultipleIndexSpacesForDynamicUpdateSlices) {
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
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 7);

  EXPECT_GE(args[0].Get<int32_t>({}), -1);
  EXPECT_LE(args[0].Get<int32_t>({}), 1);

  EXPECT_GE(args[1].Get<int32_t>({}), -1);
  EXPECT_LE(args[1].Get<int32_t>({}), 2);

  EXPECT_GE(args[2].Get<int32_t>({}), -1);
  EXPECT_LE(args[2].Get<int32_t>({}), 3);
}

TEST_F(TestUtilsTest, NoDuplicatesInt32) {
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
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  const Literal& key_arg = args[0];

  absl::flat_hash_set<int32_t> key_set;
  for (const int32_t& value : key_arg.data<int32_t>()) {
    EXPECT_TRUE(key_set.insert(absl::bit_cast<uint32_t>(value)).second);
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args2,
                          MakeDataflowConstrainedArguments(module.get()));
  ASSERT_EQ(args2.size(), 2);
  const Literal& key_arg2 = args2[0];

  absl::flat_hash_set<int32_t> key_set2;
  for (const int32_t& value : key_arg2.data<int32_t>()) {
    EXPECT_TRUE(key_set2.insert(absl::bit_cast<uint32_t>(value)).second);
  }
}

TEST_F(TestUtilsTest, MakeFakeArgumentsR0InputToDynamicSlice) {
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
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  EXPECT_TRUE(ShapeUtil::Equal(args[0].shape(), ShapeUtil::MakeShape(S32, {})))
      << ShapeUtil::HumanString(args[0].shape());
  EXPECT_TRUE(
      ShapeUtil::Equal(args[1].shape(), ShapeUtil::MakeShape(F32, {20, 20})))
      << ShapeUtil::HumanString(args[1].shape());
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForGather) {
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
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);

  const Shape& indices_shape = args[1].shape();
  EXPECT_TRUE(
      ShapeUtil::Equal(indices_shape, ShapeUtil::MakeShape(S32, {10, 2})))
      << ShapeUtil::HumanString(indices_shape);
  auto indices = args[1].data<int32_t>();
  for (const auto index : indices) {
    EXPECT_GE(index, -1);
    EXPECT_LE(index, 100);
  }
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForGatherTupleParam) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule cluster_13361217111314620287__.11, entry_computation_layout={((s32[10]{0:T(1024)}, bf16[100,256]{1,0:T(8,128)(2,1)}))->(bf16[10,256]{1,0:T(8,128)(2,1)})}

ENTRY cluster_13361217111314620287__.11 {
  constant.6 = s32[] constant(0), metadata={op_type="GatherV2" op_name="GatherV2"}
  arg_tuple.1 = (s32[10]{0:T(1024)}, bf16[100,256]{1,0:T(8,128)(2,1)}) parameter(0), parameter_replication={false,true}, sharding={{maximal device=0 metadata={op_type="_TPUReplicate" op_name="cluster"}}, {maximal device=0 metadata={op_type="_TPUReplicate" op_name="cluster"}}}, metadata={op_name="XLA_Args"}
  get-tuple-element.3 = bf16[100,256]{1,0:T(8,128)(2,1)} get-tuple-element(arg_tuple.1), index=1, sharding={maximal device=0 metadata={op_type="_TPUReplicate" op_name="cluster"}}, metadata={op_name="const_0_arg"}
  reshape.5 = bf16[100,256]{1,0} reshape(get-tuple-element.3)
  get-tuple-element.2 = s32[10]{0:T(1024)} get-tuple-element(arg_tuple.1), index=0, sharding={maximal device=0 metadata={op_type="_TPUReplicate" op_name="cluster"}}, metadata={op_name="input0_0_arg"}
  reshape.4 = s32[10]{0} reshape(get-tuple-element.2)
  gather.7 = bf16[10,256]{1,0} gather(reshape.5, reshape.4), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,256}, metadata={op_type="GatherV2" op_name="GatherV2"}
  reshape.8 = bf16[10,256]{1,0:T(8,128)(2,1)} reshape(gather.7), metadata={op_name="XLA_Retvals"}
  copy.9 = bf16[10,256]{1,0:T(8,128)(2,1)} copy(reshape.8), sharding={maximal device=0 metadata={op_type="_TPUReplicate" op_name="cluster"}}, metadata={op_name="XLA_Retvals"}
  ROOT tuple.10 = (bf16[10,256]{1,0:T(8,128)(2,1)}) tuple(copy.9), sharding={{maximal device=0 metadata={op_type="_TPUReplicate" op_name="cluster"}}}, metadata={op_name="XLA_Retvals"}
}
)")
                    .value();

  FakeArgumentsOptions options;
  options.use_large_range = true;
  options.treat_gte_as_data_formatting = true;
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get(), options));
  ASSERT_EQ(args.size(), 1);

  const Shape& indices_shape = args[0].shape().tuple_shapes()[0];
  EXPECT_TRUE(ShapeUtil::Equal(indices_shape, ShapeUtil::MakeShape(S32, {10})))
      << ShapeUtil::HumanString(indices_shape);
  const std::vector<Literal> results = args[0].DecomposeTuple();
  auto indices = results[0].data<int32_t>();
  for (const auto index : indices) {
    EXPECT_GE(index, -1);
    EXPECT_LE(index, 100);
  }
}

TEST_F(TestUtilsTest, MakeDataflowConstrainedArgumentsForTupleParam) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule cluster_tuple_gather, entry_computation_layout={((s32[10], bf16[100,256]))->(bf16[10,256])}

ENTRY cluster {
  arg_tuple.1 = (s32[10], bf16[100,256]) parameter(0)
  get-tuple-element.0 = s32[10] get-tuple-element(arg_tuple.1), index=0
  get-tuple-element.1 = bf16[100,256] get-tuple-element(arg_tuple.1), index=1
  gather.2 = bf16[10,256] gather(get-tuple-element.1, get-tuple-element.0),
    offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=1, slice_sizes={1,256}
  ROOT tuple.3 = (bf16[10,256]) tuple(gather.2)
}
)")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeDataflowConstrainedArguments(module.get()));
  ASSERT_EQ(args.size(), 1);
  ASSERT_TRUE(args[0].shape().IsTuple());
  ASSERT_EQ(args[0].shape().tuple_shapes().size(), 2);

  const Shape& indices_shape = args[0].shape().tuple_shapes()[0];
  EXPECT_TRUE(ShapeUtil::Equal(indices_shape, ShapeUtil::MakeShape(S32, {10})))
      << ShapeUtil::HumanString(indices_shape);
  const std::vector<Literal> results = args[0].DecomposeTuple();
  auto indices = results[0].data<int32_t>();
  for (const auto index : indices) {
    EXPECT_GE(index, 0);
    EXPECT_LE(index, 99);
  }
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForScatter) {
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
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 3);

  const Shape& indices_shape = args[1].shape();
  EXPECT_TRUE(
      ShapeUtil::Equal(indices_shape, ShapeUtil::MakeShape(S32, {10, 2})))
      << ShapeUtil::HumanString(indices_shape);
  auto indices = args[1].data<int32_t>();
  for (const auto index : indices) {
    EXPECT_GE(index, -1);
    EXPECT_LE(index, 100);
  }
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForFusionWithDynamicSlice) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule fusion.14451

%copy_fusion.605.clone (input.626: bf16[64,128,512,1]) -> bf16[64,128,512,1] {
  %input.626 = bf16[64,128,512,1]{2,1,0,3} parameter(0)
  ROOT %copy.55330 = bf16[64,128,512,1]{2,1,0,3} copy(%input.626)
}

%fused_computation.145.clone.clone.clone (param_0.99237: u32[], param_1.115419: bf16[64,128,512,1], param_2.79387: u32[], param_3.55185: u32[], param_4.37665: u32[]) -> (bf16[4,128,512,2], bf16[4,128,512,2]) {
  %param_1.115419 = bf16[64,128,512,1]{2,1,0,3} parameter(1)
  %fusion.16278 = bf16[64,128,512,1]{2,1,0,3} fusion(%param_1.115419), kind=kLoop, calls=%copy_fusion.605.clone
  %constant.93695 = f32[] constant(-inf)
  %pad.8833 = bf16[64,128,512,2]{2,1,0,3} pad(%fusion.16278, %constant.93695), padding=0_0x0_0x0_0x0_1
  %param_0.99237 = u32[] parameter(0)
  %zero.83 = u32[] constant(0)
  %dynamic-slice.10062 = bf16[4,128,512,2]{2,1,0,3} dynamic-slice(%pad.8833, %param_0.99237, %zero.83, %zero.83, %zero.83), dynamic_slice_sizes={4,128,512,2}
  %pad.8834 = bf16[64,128,512,2]{2,1,0,3} pad(%fusion.16278, %constant.93695), padding=0_0x0_0x0_0x1_0
  %param_2.79387 = u32[] parameter(2)
  %dynamic-slice.10063 = bf16[4,128,512,2]{2,1,0,3} dynamic-slice(%pad.8834, %param_2.79387, %zero.83, %zero.83, %zero.83), dynamic_slice_sizes={4,128,512,2}
  %maximum.1334 = bf16[4,128,512,2]{2,1,0,3} maximum(%dynamic-slice.10062, %dynamic-slice.10063)
  %copy.55331 = bf16[4,128,512,2]{2,1,0,3} copy(%maximum.1334)
  %param_4.37665 = u32[] parameter(4)
  %dynamic-slice.7328.clone.3 = bf16[4,128,512,2]{2,1,0,3} dynamic-slice(%pad.8833, %param_4.37665, %zero.83, %zero.83, %zero.83), dynamic_slice_sizes={4,128,512,2}
  %param_3.55185 = u32[] parameter(3)
  %dynamic-slice.7332.clone.3 = bf16[4,128,512,2]{2,1,0,3} dynamic-slice(%pad.8834, %param_3.55185, %zero.83, %zero.83, %zero.83), dynamic_slice_sizes={4,128,512,2}
  %maximum.415.clone.3 = bf16[4,128,512,2]{2,1,0,3} maximum(%dynamic-slice.7328.clone.3, %dynamic-slice.7332.clone.3)
  %copy.55332 = bf16[4,128,512,2]{2,1,0,3} copy(%maximum.415.clone.3)
  ROOT %tuple.22033 = (bf16[4,128,512,2]{2,1,0,3}, bf16[4,128,512,2]{2,1,0,3}) tuple(%copy.55331, %copy.55332)
}

ENTRY %fusion.14451 (parameter.0: u32[], parameter.1: bf16[64,128,512,1], parameter.2: u32[], parameter.3: u32[], parameter.4: u32[]) -> (bf16[4,128,512,2], bf16[4,128,512,2]) {
  %parameter.0 = u32[] parameter(0)
  %parameter.1 = bf16[64,128,512,1]{2,1,0,3} parameter(1)
  %parameter.2 = u32[] parameter(2)
  %parameter.3 = u32[] parameter(3)
  %parameter.4 = u32[] parameter(4)
  ROOT %fusion.14451 = (bf16[4,128,512,2]{2,1,0,3}, bf16[4,128,512,2]{2,1,0,3}) fusion(%parameter.0, %parameter.1, %parameter.2, %parameter.3, %parameter.4), kind=kLoop, calls=%fused_computation.145.clone.clone.clone
}
)")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 5);

  // args[0], [2], [3], [4] are indices for dim 0.
  // Input shape dim 0 is 64. Output shape dim 0 is 4.
  // Index must be <= 64 - 4 = 60.
  int64_t bound = 60;

  for (int i : {0, 2, 3, 4}) {
    EXPECT_GE(args[i].Get<uint32_t>({}), 0);
    EXPECT_LE(args[i].Get<uint32_t>({}), bound);
  }
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForFusionWithMultipleDimsDynamicSlices) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule MultipleDimsDynamicSlices

%fused_computation (p0: u32[], p1: u32[], data: f32[100,200]) -> f32[10,20] {
  %p0 = u32[] parameter(0)
  %p1 = u32[] parameter(1)
  %data = f32[100,200]{1,0} parameter(2)
  ROOT %dynamic-slice = f32[10,20]{1,0} dynamic-slice(%data, %p0, %p1), dynamic_slice_sizes={10,20}
}

ENTRY %main (p0: u32[], p1: u32[], data: f32[100,200]) -> f32[10,20] {
  %p0 = u32[] parameter(0)
  %p1 = u32[] parameter(1)
  %data = f32[100,200]{1,0} parameter(2)
  ROOT %fusion = f32[10,20]{1,0} fusion(%p0, %p1, %data), kind=kLoop, calls=%fused_computation
}
)")
                    .value();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 3);

  // Dim 0: 100 - 10 = 90
  EXPECT_GE(args[0].Get<uint32_t>({}), 0);
  EXPECT_LE(args[0].Get<uint32_t>({}), 90);

  // Dim 1: 200 - 20 = 180
  EXPECT_GE(args[1].Get<uint32_t>({}), 0);
  EXPECT_LE(args[1].Get<uint32_t>({}), 180);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args2,
                          MakeDataflowConstrainedArguments(module.get()));
  ASSERT_EQ(args2.size(), 3);

  // Dim 0: 100 - 10 = 90
  EXPECT_GE(args2[0].Get<uint32_t>({}), 0);
  EXPECT_LE(args2[0].Get<uint32_t>({}), 90);

  // Dim 1: 200 - 20 = 180
  EXPECT_GE(args2[1].Get<uint32_t>({}), 0);
  EXPECT_LE(args2[1].Get<uint32_t>({}), 180);
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForDynamicSliceKnownBits) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY %main (param_1: s8[262144,2048], param_2: s32[]) -> s8[131072,2048] {
  %param_1 = s8[262144,2048] parameter(0)
  %param_2 = s32[] parameter(1)
  %constant = s32[] constant(0)
  ROOT %dynamic-slice = s8[131072,2048] dynamic-slice(%param_1, %param_2, %constant), dynamic_slice_sizes={131072,2048}
}
)"));

  int32_t index_known_bits_zero = 131071;  // 0x1FFFF
  GetIndexKnownZeroesFn index_known_zeroes_fn =
      [&](const HloInstruction* use,
          int64_t sliced_dim) -> std::optional<uint64_t> {
    if (use->opcode() != HloOpcode::kDynamicSlice) {
      return std::nullopt;
    }
    if (sliced_dim == 0) {
      return index_known_bits_zero;
    }
    return std::nullopt;
  };

  FakeArgumentsOptions options;
  options.get_index_known_zeroes = index_known_zeroes_fn;
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get(), options));
  ASSERT_EQ(args.size(), 2);

  int32_t index = args[1].Get<int32_t>({});
  EXPECT_EQ(index & index_known_bits_zero, 0);
}

TEST_F(TestUtilsTest, MakeFakeArgumentsForDynamicUpdateSliceKnownBits) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule test_module

ENTRY %main (param_1: s8[262144,2048], param_2: s8[131072,2048], param_3: s32[]) -> s8[262144,2048] {
  %param_1 = s8[262144,2048] parameter(0)
  %param_2 = s8[131072,2048] parameter(1)
  %param_3 = s32[] parameter(2)
  %constant = s32[] constant(0)
  ROOT %dynamic-update-slice = s8[262144,2048] dynamic-update-slice(%param_1, %param_2, %param_3, %constant)
}
)"));

  GetIndexKnownZeroesFn index_known_zeroes_fn =
      [](const HloInstruction* use,
         int64_t sliced_dim) -> std::optional<uint64_t> {
    if (use->opcode() != HloOpcode::kDynamicUpdateSlice) {
      return std::nullopt;
    }
    if (sliced_dim == 0) {
      return 131071;  // 0x1FFFF
    }
    return std::nullopt;
  };

  FakeArgumentsOptions options;
  options.get_index_known_zeroes = index_known_zeroes_fn;
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get(), options));
  ASSERT_EQ(args.size(), 3);

  int32_t index = args[2].Get<int32_t>({});
  const int32_t index_known_bits_zero = 131071;
  EXPECT_EQ(index & index_known_bits_zero, 0);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> args2,
      MakeDataflowConstrainedArguments(module.get(), options));
  ASSERT_EQ(args2.size(), 3);
  int32_t index2 = args2[2].Get<int32_t>({});
  EXPECT_EQ(index2 & index_known_bits_zero, 0);
}

TEST_F(TestUtilsTest, MakeDataflowConstrainedArgumentsForRsqrtAdd) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[4,2] parameter(0)
  constant_1 = f32[] constant(0.00390625)
  broadcast_1 = f32[4,2] broadcast(constant_1), dimensions={}
  mul = f32[4,2] multiply(param_0, broadcast_1)
  constant_2 = f32[] constant(1e-06)
  broadcast_2 = f32[4,2] broadcast(constant_2), dimensions={}
  add = f32[4,2] add(mul, broadcast_2)
  ROOT rsqrt = f32[4,2] rsqrt(add)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeDataflowConstrainedArguments(module.get()));
  ASSERT_EQ(args.size(), 1);
  args[0].EachCell<float>([](absl::Span<int64_t const> indices, float value) {
    EXPECT_GT(value, 0.0f);
  });
}

TEST_F(TestUtilsTest, MakeDataflowConstrainedArgumentsForLogConvert) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = s32[8,128] parameter(0)
  convert = f32[8,128] convert(param_0)
  ROOT log = f32[8,128] log(convert)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeDataflowConstrainedArguments(module.get()));
  ASSERT_EQ(args.size(), 1);
  args[0].EachCell<int32_t>([](absl::Span<int64_t const> indices,
                               int32_t value) { EXPECT_GE(value, 1); });
}

TEST_F(TestUtilsTest, RejectMaxBitsOfPrecision) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  ROOT param_0 = f32[4] parameter(0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  FakeArgumentsOptions options;
  options.max_bits_of_precision = 5;
  auto status_or_args = MakeDataflowConstrainedArguments(module.get(), options);
  EXPECT_FALSE(status_or_args.ok());
  EXPECT_EQ(status_or_args.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(TestUtilsTest, FakeArgsRejectParameterRanges) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  ROOT param_0 = f32[4] parameter(0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  FakeArgumentsOptions options;
  options.parameter_ranges[0] = {0.0, 1.0};
  auto status_or_args = MakeFakeArguments(module.get(), options);
  EXPECT_FALSE(status_or_args.ok());
  EXPECT_EQ(status_or_args.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(TestUtilsTest, ParameterRangesFloat) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[4,4] parameter(0)
  ROOT sqrt_op = f32[4,4] sqrt(param_0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  FakeArgumentsOptions options;
  // Graph constraint for sqrt is [0, inf).
  // User specifies range [2.0, 5.0].
  options.parameter_ranges[0] = {2.0, 5.0};
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeDataflowConstrainedArguments(module.get(), options));
  ASSERT_EQ(args.size(), 1);
  args[0].EachCell<float>([](absl::Span<int64_t const> indices, float value) {
    EXPECT_GE(value, 2.0f);
    EXPECT_LE(value, 5.0f);
  });
}

TEST_F(TestUtilsTest, ParameterRangesTuple) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[2] parameter(0)
  param_tuple = (f32[2], s32[3]) parameter(1)
  gte_0 = f32[2] get-tuple-element(param_tuple), index=0
  gte_1 = s32[3] get-tuple-element(param_tuple), index=1
  ROOT root = (f32[2], f32[2], s32[3]) tuple(param_0, gte_0, gte_1)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  // Flat index 0: param_0
  // Flat index 1: param_tuple element 0 (f32[2])
  // Flat index 2: param_tuple element 1 (s32[3])
  FakeArgumentsOptions options;
  options.parameter_ranges[1] = {10.0, 20.0};
  options.parameter_ranges[2] = {50.0, 100.0};
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeDataflowConstrainedArguments(module.get(), options));
  ASSERT_EQ(args.size(), 2);
  const std::vector<Literal> tuple_elems = args[1].DecomposeTuple();
  ASSERT_EQ(tuple_elems.size(), 2);
  tuple_elems[0].EachCell<float>(
      [](absl::Span<int64_t const> indices, float value) {
        EXPECT_GE(value, 10.0f);
        EXPECT_LE(value, 20.0f);
      });
  tuple_elems[1].EachCell<int32_t>(
      [](absl::Span<int64_t const> indices, int32_t value) {
        EXPECT_GE(value, 50);
        EXPECT_LE(value, 100);
      });
}

TEST_F(TestUtilsTest, ParameterRangeConflict) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  param_0 = f32[4] parameter(0)
  c_offset = f32[4] constant({10.0, 10.0, 10.0, 10.0})
  sub = f32[4] subtract(param_0, c_offset)
  ROOT sqrt_op = f32[4] sqrt(sub)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  // sqrt(param - 10) requires param >= 10.
  // User specifies range [0.0, 5.0].
  // Intersection is empty, so it should return an InvalidArgument status.
  FakeArgumentsOptions options;
  options.parameter_ranges[0] = {0.0, 5.0};
  auto status_or_args = MakeDataflowConstrainedArguments(module.get(), options);
  EXPECT_FALSE(status_or_args.ok());
  EXPECT_EQ(status_or_args.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(TestUtilsTest, ParameterRangeOutOfBounds) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  ROOT param_0 = f32[4] parameter(0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  FakeArgumentsOptions options;
  options.parameter_ranges[5] = {0.0,
                                 1.0};  // Only 1 parameter exists (index 0).
  auto status_or_args = MakeDataflowConstrainedArguments(module.get(), options);
  EXPECT_FALSE(status_or_args.ok());
  EXPECT_EQ(status_or_args.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(TestUtilsTest, ParameterRangeNestedTupleOutOfBounds) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  ROOT param_nested = ((f32[2], f32[2]), f32[2]) parameter(0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  // Nested tuple has 3 leaf shapes, but only 2 top-level elements (indices 0
  // and 1). Index 2 must be rejected as out of bounds.
  FakeArgumentsOptions options;
  options.parameter_ranges[2] = {0.0, 1.0};
  auto status_or_args = MakeDataflowConstrainedArguments(module.get(), options);
  EXPECT_FALSE(status_or_args.ok());
  EXPECT_EQ(status_or_args.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(TestUtilsTest, ParameterRangeIntegerOutOfRange) {
  const char* hlo = R"(
HloModule TestModule
ENTRY main {
  ROOT param_0 = s32[4] parameter(0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  {
    // Range >= 2^63 exceeds INT64_MAX.
    FakeArgumentsOptions options;
    options.parameter_ranges[0] = {1e20, 1e21};
    auto status_or_args =
        MakeDataflowConstrainedArguments(module.get(), options);
    EXPECT_FALSE(status_or_args.ok());
    EXPECT_EQ(status_or_args.status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  {
    // Range < -2^63 is below INT64_MIN.
    FakeArgumentsOptions options;
    options.parameter_ranges[0] = {-1e21, -1e20};
    auto status_or_args =
        MakeDataflowConstrainedArguments(module.get(), options);
    EXPECT_FALSE(status_or_args.ok());
    EXPECT_EQ(status_or_args.status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

// Probabilistic test to verify that we are randomly sampling the whole
// range for small bitwidth floats like f4e2m1. The chance of not getting
// all 16 possible values in 1024 trials is astronomically small.
TEST_F(TestUtilsTest, MakeFakeArgumentsSmallBitwidthFloat) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule Test

ENTRY %module (param: f4e2m1fn[1024]) -> f4e2m1fn[1024] {
  ROOT %param = f4e2m1fn[1024]{0} parameter(0)
}
  )"));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                          MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 1);
  EXPECT_TRUE(
      ShapeUtil::Equal(args[0].shape(), ShapeUtil::MakeShape(F4E2M1FN, {1024})))
      << ShapeUtil::HumanString(args[0].shape());

  TF_ASSERT_OK_AND_ASSIGN(Literal f32_arg, args[0].Convert(F32));

  absl::flat_hash_set<uint32_t> values;
  f32_arg.EachCell<float>(
      [&](absl::Span<const int64_t> /*indices*/, float val) {
        values.insert(absl::bit_cast<uint32_t>(val));
      });

  const int64_t num_possible_values = 16;
  EXPECT_EQ(values.size(), num_possible_values);
}

// Tests that max reduction uses MinValue as the identity element through copy
// pass-through.
TEST_F(TestUtilsTest, ReduceMaxIdentityElement) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule ReduceMaxIdentityModule

max_BF16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(lhs, rhs)
}

ENTRY entry {
  param_0 = bf16[4,5,128,256] parameter(0)
  param_1 = bf16[] parameter(1)
  copy = bf16[] copy(param_1)
  ROOT reduce-window = bf16[3,4,128,256] reduce-window(param_0, copy),
    window={size=2x2x1x1 pad=0_0x0_0x0_0x0_0}, to_apply=max_BF16
}
)"));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  EXPECT_EQ(args[1].Get<bfloat16>({}),
            LiteralUtil::MinValue(BF16).Get<bfloat16>({}));
}

// Tests that min reduction uses MaxValue as the identity element through copy
// pass-through and fusion.
TEST_F(TestUtilsTest, ReduceMinIdentityElement) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule ReduceMinIdentityModule

min_F32 (lhs: f32[], rhs: f32[]) -> f32[] {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT minimum = f32[] minimum(lhs, rhs)
}

fused_computation (param_0: f32[4,5,128,256], param_1: f32[]) -> f32[3,4,128,256] {
  param_0 = f32[4,5,128,256] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT reduce-window = f32[3,4,128,256] reduce-window(param_0, param_1),
    window={size=2x2x1x1 pad=0_0x0_0x0_0x0_0}, to_apply=min_F32
}

ENTRY entry {
  param_0 = f32[4,5,128,256] parameter(0)
  param_1 = f32[] parameter(1)
  copy = f32[] copy(param_1)
  ROOT fusion = f32[3,4,128,256] fusion(param_0, copy), kind=kOutput, calls=fused_computation
}
)"));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 2);
  EXPECT_EQ(args[1].Get<float>({}), LiteralUtil::MaxValue(F32).Get<float>({}));
}

// Tests that MakeFakeArguments succeeds when a fusion instruction has unused
// operands (operands beyond the number of parameters in the fused computation).
TEST_F(TestUtilsTest, FusionWithUnusedOperand) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule FusionWithUnusedOperandModule

fused_computation (param_0: f32[4,5,128,256]) -> f32[4,5,128,256] {
  param_0 = f32[4,5,128,256] parameter(0)
  ROOT copy = f32[4,5,128,256] copy(param_0)
}

ENTRY entry {
  param_0 = f32[4,5,128,256] parameter(0)
  param_1 = f32[] parameter(1)
  copy = f32[] copy(param_1)
  ROOT fusion = f32[4,5,128,256] fusion(param_0, copy), kind=kOutput, calls=fused_computation
}
)"));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get()));
  EXPECT_EQ(args.size(), 2);
}

// Verifies that when `cache_arguments` is false (the default), every parameter
// of an identical shape receives an independently generated random tensor.
TEST_F(TestUtilsTest, DefaultFakeArgumentsKeepAllRepeatedShapesDistinct) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule DistinctDefaultFakeArgsModule

ENTRY entry {
  p0 = f32[128,256] parameter(0)
  p1 = f32[128,256] parameter(1)
  p2 = f32[128,256] parameter(2)
  p3 = f32[128,256] parameter(3)
  add0 = f32[128,256] add(p0, p1)
  add1 = f32[128,256] add(p2, p3)
  ROOT out = f32[128,256] add(add0, add1)
}
)"));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get()));
  ASSERT_EQ(args.size(), 4);
  EXPECT_NE(args[0], args[1]);
  EXPECT_NE(args[0], args[2]);
  EXPECT_NE(args[1], args[3]);
}

constexpr absl::string_view kCachedFakeArgsModuleHlo = R"(
HloModule CachedFakeArgsModule, entry_computation_layout={(f32[128,256]{1,0:T(8,128)}, f32[128,256]{1,0:T(8,128)}, f32[128,256]{1,0:T(8,128)}, f32[128,256]{1,0:T(8,128)}, (s4[256,256]{1,0:T(8,128)(8,1)E(4)}, s4[256,256]{1,0:T(8,128)(8,1)E(4)}, s4[256,256]{1,0:T(8,128)(8,1)E(4)}))->f32[128,256]{1,0}}

ENTRY entry {
  p0 = f32[128,256] parameter(0)
  p1 = f32[128,256] parameter(1)
  p2 = f32[128,256] parameter(2)
  p3 = f32[128,256] parameter(3)
  tup = (s4[256,256], s4[256,256], s4[256,256]) parameter(4)
  add0 = f32[128,256] add(p0, p1)
  add1 = f32[128,256] add(p2, p3)
  ROOT out = f32[128,256] add(add0, add1)
}
)";

// Verifies that `cache_arguments` strips device tiling/packing (`T(...)` and
// `E(4)`) onto canonical host shapes and cycles through
// `max_cached_variants_per_shape` distinct variants per shape (including inside
// tuple parameters).
TEST_F(TestUtilsTest, CachedFakeArgumentsReusesVariantsAndNormalizesShapes) {
  ClearFakeArgumentCacheForTest();
  absl::Cleanup cleanup = [&] { ClearFakeArgumentCacheForTest(); };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kCachedFakeArgsModuleHlo));

  FakeArgumentsOptions opts;
  opts.cache_arguments = true;

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args,
                       MakeFakeArguments(module.get(), opts));
  ASSERT_EQ(args.size(), 5);
  // Variant 0 and Variant 1 must be distinct, and reused for p2 and p3.
  EXPECT_NE(args[0], args[1]);
  EXPECT_EQ(args[0], args[2]);
  EXPECT_EQ(args[1], args[3]);
  // Both cache misses and cache hits must return normalized untiled host
  // shapes.
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(ShapeUtil::DeviceShapeIsHostShape(args[i].shape()))
        << "arg=" << i;
  }
  EXPECT_TRUE(Shape::Equal()(args[0].shape(), args[2].shape()));

  // Verify packed E(4) s4[256,256] tuple leaves (64 KiB host Literal) were
  // cached and normalized.
  std::vector<Literal> tup_leaves = args[4].Clone().DecomposeTuple();
  ASSERT_EQ(tup_leaves.size(), 3);
  EXPECT_NE(tup_leaves[0], tup_leaves[1]);
  EXPECT_EQ(tup_leaves[0], tup_leaves[2]);
  EXPECT_TRUE(ShapeUtil::DeviceShapeIsHostShape(tup_leaves[0].shape()));
}

// Verifies that cached arguments persist as `.xlatensor` files under
// `XLA_ARGUMENT_CACHE_DIR` and reload from disk when the in-memory cache is
// cold.
TEST_F(TestUtilsTest, CachedFakeArgumentsPersistsToAndLoadsFromDisk) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ClearFakeArgumentCacheForTest();
  tsl::setenv("XLA_ARGUMENT_CACHE_DIR", temp_dir.path().c_str(),
              /*overwrite=*/1);
  absl::Cleanup cleanup = [&] {
    tsl::unsetenv("XLA_ARGUMENT_CACHE_DIR");
    ClearFakeArgumentCacheForTest();
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kCachedFakeArgsModuleHlo));

  FakeArgumentsOptions opts;
  opts.cache_arguments = true;
  opts.max_cached_variants_per_shape = 1;

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args1,
                       MakeFakeArguments(module.get(), opts));
  ASSERT_EQ(args1.size(), 5);

  tsl::Env* env = tsl::Env::Default();
  std::vector<std::string> f32_files;
  ASSERT_OK(env->GetMatchingPaths(
      tsl::io::JoinPath(temp_dir.path(), "v1", "f32_128x256", "*",
                        "*.xlatensor"),
      &f32_files));
  ASSERT_EQ(f32_files.size(), 1);

  std::vector<std::string> s4_files;
  ASSERT_OK(
      env->GetMatchingPaths(tsl::io::JoinPath(temp_dir.path(), "v1",
                                              "s4_256x256", "*", "*.xlatensor"),
                            &s4_files));
  ASSERT_EQ(s4_files.size(), 1);

  // Clear the in-memory cache and mutate the first payload float of the
  // `.xlatensor` file on disk to prove the next call reloads from
  // `TryLoadFromDisk` rather than regenerating or hitting RAM.
  ClearFakeArgumentCacheForTest();
  const std::string& target_file = f32_files[0];
  std::string file_bytes;
  ASSERT_OK(tsl::ReadFileToString(env, target_file, &file_bytes));
  ASSERT_EQ(file_bytes.size(),
            4096 + static_cast<size_t>(args1[0].size_bytes()));
  const float sentinel = 42.5f;
  std::memcpy(&file_bytes[4096], &sentinel, sizeof(float));
  ASSERT_OK(tsl::WriteStringToFile(env, target_file, file_bytes));

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args_disk,
                       MakeFakeArguments(module.get(), opts));
  ASSERT_EQ(args_disk.size(), 5);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(args_disk[i].data<float>()[0], sentinel) << "arg=" << i;
  }
}

// Verifies that a corrupted `.xlatensor` file on disk is deleted, regenerated
// deterministically, and rewritten with a valid header.
TEST_F(TestUtilsTest,
       CachedFakeArgumentsRegeneratesAndRepairsCorruptedDiskFile) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ClearFakeArgumentCacheForTest();
  tsl::setenv("XLA_ARGUMENT_CACHE_DIR", temp_dir.path().c_str(),
              /*overwrite=*/1);
  absl::Cleanup cleanup = [&] {
    tsl::unsetenv("XLA_ARGUMENT_CACHE_DIR");
    ClearFakeArgumentCacheForTest();
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kCachedFakeArgsModuleHlo));

  FakeArgumentsOptions opts;
  opts.cache_arguments = true;

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args1,
                       MakeFakeArguments(module.get(), opts));
  ASSERT_EQ(args1.size(), 5);

  tsl::Env* env = tsl::Env::Default();
  std::vector<std::string> f32_files;
  ASSERT_OK(env->GetMatchingPaths(
      tsl::io::JoinPath(temp_dir.path(), "v1", "f32_128x256", "*",
                        "*.xlatensor"),
      &f32_files));
  ASSERT_EQ(f32_files.size(), 2);

  std::string target_file = f32_files[0];
  std::string file_bytes;
  ASSERT_OK(tsl::ReadFileToString(env, target_file, &file_bytes));

  // Corrupt the `.xlatensor` header and verify graceful fallback regeneration
  // plus on-disk repair of the corrupted file.
  ClearFakeArgumentCacheForTest();
  std::memcpy(&file_bytes[0], "BADMAGIC", 8);
  ASSERT_OK(tsl::WriteStringToFile(env, target_file, file_bytes));
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> args_regen,
                       MakeFakeArguments(module.get(), opts));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(args1[i], args_regen[i]) << "arg=" << i;
  }

  std::string repaired_bytes;
  ASSERT_OK(tsl::ReadFileToString(env, target_file, &repaired_bytes));
  EXPECT_EQ(absl::string_view(repaired_bytes).substr(0, 8), "XLATNSR1");
}

// Verifies that `MakeDataflowConstrainedArguments` incorporates explicit
// per-parameter `ConstraintInterval` ranges into the cache key so constrained
// and unconstrained parameters of the same shape never collide.
TEST_F(TestUtilsTest,
       CachedDataflowConstrainedArgumentsIncludesParameterRangesInCacheKey) {
  ClearFakeArgumentCacheForTest();
  absl::Cleanup cleanup = [&] { ClearFakeArgumentCacheForTest(); };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kCachedFakeArgsModuleHlo));

  FakeArgumentsOptions base_opts;
  base_opts.cache_arguments = true;
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> base_args,
                       MakeFakeArguments(module.get(), base_opts));

  FakeArgumentsOptions df_opts;
  df_opts.cache_arguments = true;
  df_opts.parameter_ranges = {{0, {10.0, 20.0}}, {2, {10.0, 20.0}}};
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> df_args,
                       MakeDataflowConstrainedArguments(module.get(), df_opts));
  ASSERT_EQ(df_args.size(), 5);
  EXPECT_NE(df_args[0], df_args[2]);
  EXPECT_GE(df_args[0].data<float>()[0], 10.0f);
  EXPECT_LE(df_args[0].data<float>()[0], 20.0f);
  EXPECT_EQ(df_args[1], base_args[0]);
  EXPECT_EQ(df_args[3], base_args[1]);
}

// Verifies that arrays larger than 32 MiB exercise the multi-chunk parallel
// striped `pread` path in `TryLoadFromDisk` and the parallel `memcpy` path in
// `CloneArrayLiteralParallel`.
TEST_F(TestUtilsTest, CachedFakeArgumentsParallelDiskReadAndCloneOnLargeArray) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ClearFakeArgumentCacheForTest();
  tsl::setenv("XLA_ARGUMENT_CACHE_DIR", temp_dir.path().c_str(),
              /*overwrite=*/1);
  absl::Cleanup cleanup = [&] {
    tsl::unsetenv("XLA_ARGUMENT_CACHE_DIR");
    ClearFakeArgumentCacheForTest();
  };

  // 33 MiB (2 * 16 MiB + 1 MiB tail chunk) exercises multi-chunk parallel
  // pread in TryLoadFromDisk and parallel memcpy in CloneArrayLiteralParallel.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule LargeCachedArrayModule

ENTRY entry {
  p0 = u8[34603008] parameter(0)
  p1 = u8[34603008] parameter(1)
  p2 = u8[34603008] parameter(2)
  add0 = u8[34603008] add(p0, p1)
  ROOT out = u8[34603008] add(add0, p2)
}
)"));

  FakeArgumentsOptions opts;
  opts.cache_arguments = true;

  ASSERT_OK_AND_ASSIGN(std::vector<Literal> cold_args,
                       MakeFakeArguments(module.get(), opts));
  ASSERT_EQ(cold_args.size(), 3);
  EXPECT_EQ(cold_args[0], cold_args[2]);

  // Clear RAM so the next call executes the parallel striped pread path.
  ClearFakeArgumentCacheForTest();
  ASSERT_OK_AND_ASSIGN(std::vector<Literal> disk_args,
                       MakeFakeArguments(module.get(), opts));
  EXPECT_EQ(cold_args[0], disk_args[0]);
  EXPECT_EQ(cold_args[1], disk_args[1]);
  EXPECT_EQ(cold_args[2], disk_args[2]);
}

// Verifies that both the in-memory LRU cache and the on-disk `.xlatensor` LRU
// tracker promote accessed entries on lookup and evict the least-recently used
// entries when capacity limits are exceeded.
TEST_F(TestUtilsTest,
       CachedFakeArgumentsEvictsLeastRecentlyUsedInMemoryAndOnDisk) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ClearFakeArgumentCacheForTest();
  tsl::setenv("XLA_ARGUMENT_CACHE_DIR", temp_dir.path().c_str(),
              /*overwrite=*/1);
  absl::Cleanup cleanup = [&] {
    tsl::unsetenv("XLA_ARGUMENT_CACHE_DIR");
    ClearFakeArgumentCacheForTest();
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule LruEvictionModule

ENTRY entry {
  ROOT p0 = f32[128,256] parameter(0)
}
)"));

  // Each f32[128,256] literal is 128 KiB (131,072 bytes). Cap both the
  // in-memory and on-disk caches at 256 KiB (enough for exactly 2 entries).
  constexpr int64_t kTwoEntriesBytes = 2 * 128 * 1024;
  SetFakeArgumentCacheLimitsForTest(/*max_in_memory_bytes=*/kTwoEntriesBytes,
                                    /*max_disk_bytes=*/kTwoEntriesBytes);

  auto make_with_seed =
      [&](std::minstd_rand0::result_type seed) -> absl::StatusOr<Literal> {
    std::minstd_rand0 engine(seed);
    FakeArgumentsOptions opts;
    opts.engine = &engine;
    opts.cache_arguments = true;
    opts.max_cached_variants_per_shape = 1;
    ABSL_ASSIGN_OR_RETURN(std::vector<Literal> args,
                     MakeFakeArguments(module.get(), opts));
    return std::move(args[0]);
  };

  tsl::Env* env = tsl::Env::Default();
  const std::string glob_pattern = tsl::io::JoinPath(
      temp_dir.path(), "v1", "f32_128x256", "*", "*.xlatensor");

  // 1. Insert seed 1 (`file_seed1`), then seed 2 (`file_seed2`). Cache is now
  // at capacity `{seed 2 (MRU), seed 1 (LRU)}`.
  ASSERT_OK_AND_ASSIGN(Literal lit_seed1, make_with_seed(1));
  std::vector<std::string> files_after_seed1;
  ASSERT_OK(env->GetMatchingPaths(glob_pattern, &files_after_seed1));
  ASSERT_EQ(files_after_seed1.size(), 1);
  const std::string file_seed1 = files_after_seed1[0];

  ASSERT_OK_AND_ASSIGN(Literal lit_seed2, make_with_seed(2));
  EXPECT_NE(lit_seed1, lit_seed2);

  // 2. Access seed 1 again to promote it to MRU (`{seed 1 (MRU), seed 2
  // (LRU)}`), then insert seed 3. Seed 2 (the LRU entry) must be evicted from
  // both RAM and disk, while seed 1 and seed 3 remain.
  ASSERT_OK_AND_ASSIGN(Literal lit_seed1_hit, make_with_seed(1));
  EXPECT_EQ(lit_seed1_hit, lit_seed1);
  ASSERT_OK_AND_ASSIGN(Literal lit_seed3, make_with_seed(3));
  EXPECT_NE(lit_seed3, lit_seed1);

  std::vector<std::string> files_after_eviction;
  ASSERT_OK(env->GetMatchingPaths(glob_pattern, &files_after_eviction));
  EXPECT_EQ(files_after_eviction.size(), 2);
  EXPECT_THAT(files_after_eviction, ::testing::Contains(file_seed1));

  // 3. Verify that seed 1 is still served from the in-memory cache even after
  // its on-disk file is deleted.
  ASSERT_OK(env->DeleteFile(file_seed1));
  ASSERT_OK_AND_ASSIGN(Literal lit_seed1_from_ram, make_with_seed(1));
  EXPECT_EQ(lit_seed1_from_ram, lit_seed1);
}

}  // namespace
}  // namespace xla

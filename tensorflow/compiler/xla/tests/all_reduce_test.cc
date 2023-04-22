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

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class TrivialAllReduceTest : public HloTestBase {};

// Currently the CPU and GPU backends only support AllReduce with one
// replica.  But we can at least check this.

XLA_TEST_F(TrivialAllReduceTest, OneOperand) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p = f32[3] parameter(0)
    ROOT crs = f32[3] all-reduce(p), to_apply=add
  })";
  auto module =
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest())
          .ValueOrDie();
  auto literal = LiteralUtil::CreateR1<float>({1, 2, 3});
  EXPECT_EQ(literal, ExecuteAndTransfer(std::move(module), {&literal}));
}

XLA_TEST_F(TrivialAllReduceTest, MultipleOperands) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p0 = f32[3] parameter(0)
    p1 = f32[2] parameter(1)
    ROOT crs = (f32[3], f32[2]) all-reduce(p0, p1), to_apply=add
  })";
  auto module =
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest())
          .ValueOrDie();
  auto literal0 = LiteralUtil::CreateR1<float>({1, 2, 3});
  auto literal1 = LiteralUtil::CreateR1<float>({10, 20});
  EXPECT_EQ(LiteralUtil::MakeTuple({&literal0, &literal1}),
            ExecuteAndTransfer(std::move(module), {&literal0, &literal1}));
}

// On the GPU backend, constants get special handling.  Someone might pass a
// constant to CRS to e.g. count the number of replicas -- we need to make sure
// it works.
XLA_TEST_F(TrivialAllReduceTest, ConstantOperand) {
  const char* module_str = R"(
  HloModule test

  add {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    add = f32[] add(x, y)
  }

  ENTRY test_computation {
    p0 = f32[3] parameter(0)
    p1 = f32[2] constant({10, 20})
    ROOT crs = (f32[3], f32[2]) all-reduce(p0, p1), to_apply=add
  })";
  auto module =
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest())
          .ValueOrDie();
  auto literal0 = LiteralUtil::CreateR1<float>({1, 2, 3});
  auto literal1 = LiteralUtil::CreateR1<float>({10, 20});
  EXPECT_EQ(LiteralUtil::MakeTuple({&literal0, &literal1}),
            ExecuteAndTransfer(std::move(module), {&literal0}));
}

XLA_TEST_F(TrivialAllReduceTest, AllReduceU8) {
  const char* module_str = R"(
HloModule test

%AddComputation.15 {
  %x.16 = u8[] parameter(0)
  %y.17 = u8[] parameter(1)
  ROOT %add.18 = u8[] add(u8[] %x.16, u8[] %y.17)
}

ENTRY %test_computation {
  %constant.4 = u8[] constant(0), metadata={op_type="prim::Constant" source_file="main@test_all_reduce_int.py" source_line=17}
  %reshape.5 = u8[1]{0} reshape(u8[] %constant.4), metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %broadcast.6 = u8[1]{0} broadcast(u8[1]{0} %reshape.5), dimensions={0}, metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %reshape.7 = u8[] reshape(u8[1]{0} %broadcast.6), metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %broadcast.8 = u8[8]{0} broadcast(u8[] %reshape.7), dimensions={}, metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %constant.2 = u8[] constant(1), metadata={op_type="prim::Constant" source_file="main@test_all_reduce_int.py" source_line=18}
  %reshape.3 = u8[1]{0} reshape(u8[] %constant.2), metadata={op_type="aten::view" source_file="__format__@tensor.py" source_line=563}
  %constant.9 = s64[] constant(0), metadata={op_type="xla::update_slice" source_file="__format__@tensor.py" source_line=563}
  %dynamic-update-slice.10 = u8[8]{0} dynamic-update-slice(u8[8]{0} %broadcast.8, u8[1]{0} %reshape.3, s64[] %constant.9), metadata={op_type="xla::update_slice" source_file="__format__@tensor.py" source_line=563}
  %p0.1 = f32[] parameter(0), metadata={op_type="xla::device_data" source_file="_get_all_reduce_token@xla_model.py" source_line=463}
  %convert.11 = u8[] convert(f32[] %p0.1), metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %tuple.12 = (u8[8]{0}, u8[]) tuple(u8[8]{0} %dynamic-update-slice.10, u8[] %convert.11), metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.13 = u8[8]{0} get-tuple-element((u8[8]{0}, u8[]) %tuple.12), index=0, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.14 = u8[] get-tuple-element((u8[8]{0}, u8[]) %tuple.12), index=1, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %all-reduce.19 = (u8[8]{0}, u8[]) all-reduce(u8[8]{0} %get-tuple-element.13, u8[] %get-tuple-element.14), replica_groups={}, constrain_layout=true, to_apply=%AddComputation.15, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.21 = u8[] get-tuple-element((u8[8]{0}, u8[]) %all-reduce.19), index=1, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %convert.22 = f32[] convert(u8[] %get-tuple-element.21), metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.20 = u8[8]{0} get-tuple-element((u8[8]{0}, u8[]) %all-reduce.19), index=0, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  ROOT %tuple.23 = (u8[8]{0}) tuple(u8[8]{0} %get-tuple-element.20)
})";

  auto module =
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest())
          .ValueOrDie();
  auto literal_in = LiteralUtil::CreateR0<float>(0);
  auto literal0 = LiteralUtil::CreateR1<uint8_t>({1, 0, 0, 0, 0, 0, 0, 0});
  EXPECT_EQ(LiteralUtil::MakeTuple({&literal0}),
            ExecuteAndTransfer(std::move(module), {&literal_in}));
}

XLA_TEST_F(TrivialAllReduceTest, AllReduceS32) {
  const char* module_str = R"(

HloModule test

%AddComputation.15 {
  %x.16 = s32[] parameter(0)
  %y.17 = s32[] parameter(1)
  ROOT %add.18 = s32[] add(s32[] %x.16, s32[] %y.17)
}

ENTRY %test_computation {
  %constant.4 = s32[] constant(0), metadata={op_type="prim::Constant" source_file="main@test_all_reduce_int.py" source_line=17}
  %reshape.5 = s32[1]{0} reshape(s32[] %constant.4), metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %broadcast.6 = s32[1]{0} broadcast(s32[1]{0} %reshape.5), dimensions={0}, metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %reshape.7 = s32[] reshape(s32[1]{0} %broadcast.6), metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %broadcast.8 = s32[8]{0} broadcast(s32[] %reshape.7), dimensions={}, metadata={op_type="aten::expand" source_file="main@test_all_reduce_int.py" source_line=17}
  %constant.2 = s32[] constant(1), metadata={op_type="prim::Constant" source_file="main@test_all_reduce_int.py" source_line=18}
  %reshape.3 = s32[1]{0} reshape(s32[] %constant.2), metadata={op_type="aten::view" source_file="__format__@tensor.py" source_line=563}
  %constant.9 = s64[] constant(0), metadata={op_type="xla::update_slice" source_file="__format__@tensor.py" source_line=563}
  %dynamic-update-slice.10 = s32[8]{0} dynamic-update-slice(s32[8]{0} %broadcast.8, s32[1]{0} %reshape.3, s64[] %constant.9), metadata={op_type="xla::update_slice" source_file="__format__@tensor.py" source_line=563}
  %p0.1 = f32[] parameter(0), metadata={op_type="xla::device_data" source_file="_get_all_reduce_token@xla_model.py" source_line=463}
  %convert.11 = s32[] convert(f32[] %p0.1), metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %tuple.12 = (s32[8]{0}, s32[]) tuple(s32[8]{0} %dynamic-update-slice.10, s32[] %convert.11), metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.13 = s32[8]{0} get-tuple-element((s32[8]{0}, s32[]) %tuple.12), index=0, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.14 = s32[] get-tuple-element((s32[8]{0}, s32[]) %tuple.12), index=1, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %all-reduce.19 = (s32[8]{0}, s32[]) all-reduce(s32[8]{0} %get-tuple-element.13, s32[] %get-tuple-element.14), replica_groups={}, constrain_layout=true, to_apply=%AddComputation.15, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.21 = s32[] get-tuple-element((s32[8]{0}, s32[]) %all-reduce.19), index=1, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %convert.22 = f32[] convert(s32[] %get-tuple-element.21), metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  %get-tuple-element.20 = s32[8]{0} get-tuple-element((s32[8]{0}, s32[]) %all-reduce.19), index=0, metadata={op_type="xla::cross_replica_sum" source_file="all_reduce@xla_model.py" source_line=560}
  ROOT %tuple.23 = (s32[8]{0}) tuple(s32[8]{0} %get-tuple-element.20)
})";

  auto module =
      ParseAndReturnVerifiedModule(module_str, GetModuleConfigForTest())
          .ValueOrDie();
  auto literal_in = LiteralUtil::CreateR0<float>(0);
  auto literal0 = LiteralUtil::CreateR1<int32>({1, 0, 0, 0, 0, 0, 0, 0});
  EXPECT_EQ(LiteralUtil::MakeTuple({&literal0}),
            ExecuteAndTransfer(std::move(module), {&literal_in}));
}

}  // namespace
}  // namespace xla

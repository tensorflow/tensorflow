/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

#if defined(INTEL_MKL)

class LayerNormTest : public HloTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_use_thunk_runtime(false);
    return debug_options;
  }

  const char* onednn_layer_norm_ =
      R"(
  ; CHECK:     custom_call_target="__onednn$layernorm",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "onednn_layer_norm_config":{
  ; CHECK-DAG:       "rescale":"SCALE_AND_SHIFT"
  ; CHECK-DAG:   }
  ; CHECK:     }
  )";
  std::string common_hlo_region_ =
      R"(

  region_add {
    Arg_0.7555 = f32[] parameter(0)
    Arg_1.7556 = f32[] parameter(1)
    ROOT add.7557 = f32[] add(Arg_0.7555, Arg_1.7556)
  }
)";

  std::string common_hlo_entry_computation_block_ =
      R"(
    Arg_0.2 = f32[768]{0} parameter(1), sharding={replicated}
    Arg_0.3 = f32[768]{0} parameter(2), sharding={replicated}

    convert.290 = f32[84,197,768]{2,1,0} convert(Arg_0.1)
    constant.291 = f32[] constant(0)
    convert.292 = f32[] convert(constant.291)
    reduce.297 = f32[84,197]{1,0} reduce(convert.290, convert.292), dimensions={2}, to_apply=region_add
    constant.298 = s32[] constant(768)
    convert.299 = f32[] convert(constant.298)
    broadcast.300 = f32[84,197]{1,0} broadcast(convert.299), dimensions={}
    divide.301 = f32[84,197]{1,0} divide(reduce.297, broadcast.300)
    convert.302 = f32[84,197]{1,0} convert(divide.301)
    reshape.303 = f32[84,197,1]{2,1,0} reshape(convert.302)
    reshape.304 = f32[84,197]{1,0} reshape(reshape.303)
    broadcast.305 = f32[84,197,768]{2,1,0} broadcast(reshape.304), dimensions={0,1}
    subtract.306 = f32[84,197,768]{2,1,0} subtract(Arg_0.1, broadcast.305)
    multiply.307 = f32[84,197,768]{2,1,0} multiply(subtract.306, subtract.306)
    convert.308 = f32[84,197,768]{2,1,0} convert(multiply.307)
    constant.309 = f32[] constant(0)
    convert.310 = f32[] convert(constant.309)
    reduce.315 = f32[84,197]{1,0} reduce(convert.308, convert.310), dimensions={2}, to_apply=region_add
    constant.316 = s32[] constant(768)
    convert.317 = f32[] convert(constant.316)
    broadcast.318 = f32[84,197]{1,0} broadcast(convert.317), dimensions={}
    divide.319 = f32[84,197]{1,0} divide(reduce.315, broadcast.318)
    convert.320 = f32[84,197]{1,0} convert(divide.319)
    reshape.321 = f32[84,197,1]{2,1,0} reshape(convert.320)
    constant.322 = f32[] constant(1e-12)
    broadcast.323 = f32[84,197,1]{2,1,0} broadcast(constant.322), dimensions={}
    add.324 = f32[84,197,1]{2,1,0} add(reshape.321, broadcast.323)
    rsqrt.325 = f32[84,197,1]{2,1,0} rsqrt(add.324)
    reshape.328 = f32[84,197]{1,0} reshape(rsqrt.325)
    broadcast.329 = f32[84,197,768]{2,1,0} broadcast(reshape.328), dimensions={0,1}
    broadcast.327 = f32[84,197,768]{2,1,0} broadcast(Arg_0.2), dimensions={2}
    multiply.330 = f32[84,197,768]{2,1,0} multiply(broadcast.329, broadcast.327)
    multiply.331 = f32[84,197,768]{2,1,0} multiply(Arg_0.1, multiply.330)
    broadcast.336 = f32[84,197,768]{2,1,0} broadcast(Arg_0.3), dimensions={2}
    reshape.332 = f32[84,197]{1,0} reshape(reshape.303)
    broadcast.333 = f32[84,197,768]{2,1,0} broadcast(reshape.332), dimensions={0,1}
    multiply.334 = f32[84,197,768]{2,1,0} multiply(multiply.330, broadcast.333)
    subtract.337 = f32[84,197,768]{2,1,0} subtract(broadcast.336, multiply.334)
)";
};

TEST_F(LayerNormTest, LayerNormTest0_FP32) {
  std::string layer_norm_module_str =
      R"(HloModule layer_norm.test, entry_computation_layout={(f32[84,197,768]{2,1,0}, f32[768]{0}, f32[768]{0})->f32[84,197,768]{2,1,0}})" +
      common_hlo_region_ + R"(
  ENTRY main {
    Arg_0.1 = f32[84,197,768]{2,1,0} parameter(0), sharding={replicated}

  )" + common_hlo_entry_computation_block_ +
      R"(
    ROOT add.338 = f32[84,197,768]{2,1,0} add(multiply.331, subtract.337)
  }
  )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

TEST_F(LayerNormTest, LayerNormTest0_BF16) {
  if (!xla::cpu::IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }
  std::string layer_norm_module_str =
      R"(HloModule layer_norm.test, entry_computation_layout={(bf16[84,197,768]{2,1,0}, f32[768]{0}, f32[768]{0})->bf16[84,197,768]{2,1,0}})" +
      common_hlo_region_ + R"(
  ENTRY main {
    Arg_0.1.0 = bf16[84,197,768]{2,1,0} parameter(0), sharding={replicated}
    Arg_0.1 = f32[84,197,768]{2,1,0} convert(Arg_0.1.0)
  )" + common_hlo_entry_computation_block_ +
      R"(
    add.338 = f32[84,197,768]{2,1,0} add(multiply.331, subtract.337)
    ROOT convert.339 = bf16[84,197,768]{2,1,0} convert(add.338)
  }
  )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

TEST_F(LayerNormTest, LayerNormTest0_F16) {
  if (!xla::cpu::IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }
  std::string layer_norm_module_str =
      R"(HloModule layer_norm.test, entry_computation_layout={(f16[84,197,768]{2,1,0}, f32[768]{0}, f32[768]{0})->f16[84,197,768]{2,1,0}})" +
      common_hlo_region_ + R"(
  ENTRY main {
    Arg_0.1.0 = f16[84,197,768]{2,1,0} parameter(0), sharding={replicated}
    Arg_0.1 = f32[84,197,768]{2,1,0} convert(Arg_0.1.0)
  )" + common_hlo_entry_computation_block_ +
      R"(
    add.338 = f32[84,197,768]{2,1,0} add(multiply.331, subtract.337)
    ROOT convert.339 = f16[84,197,768]{2,1,0} convert(add.338)
  }
  )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

TEST_F(LayerNormTest, LayerNormTest1_F16) {
  if (!xla::cpu::IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = f16[2,4,8] parameter(0), sharding={replicated}
    convert_0 = f32[2,4,8] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[2,4] reduce(convert_0, convert_1), dimensions={2}, to_apply=region_add
    constant_1 = s32[] constant(8)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[2,4] broadcast(convert_2), dimensions={}
    divide_0 = f32[2,4] divide(reduce_0, broadcast_0)
    convert_3 = f16[2,4] convert(divide_0)
    reshape_0 = f16[2,4,1] reshape(convert_3)
    reshape_1 = f16[2,4] reshape(reshape_0)
    broadcast_1 = f16[2,4,8] broadcast(reshape_1), dimensions={0,1}
    subtract_0 = f16[2,4,8] subtract(Arg_2, broadcast_1)
    multiply_0 = f16[2,4,8] multiply(subtract_0, subtract_0)
    convert_4 = f32[2,4,8] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_2 = f32[2,4] reduce(convert_4, convert_5), dimensions={2}, to_apply=region_add
    constant_3 = s32[] constant(8)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[2,4] broadcast(convert_6), dimensions={}
    divide_1 = f32[2,4] divide(reduce_2, broadcast_2)
    convert_7 = f16[2,4] convert(divide_1)
    reshape_2 = f16[2,4,1] reshape(convert_7)
    rsqrt_0 = f16[2,4,1] rsqrt(reshape_2)
    reshape_3 = f16[2,4] reshape(rsqrt_0)
    broadcast_3 = f16[2,4,8] broadcast(reshape_3), dimensions={0,1}
    constant_4 = f16[8]{0} constant({1,1,1,1,1,1,1,1})
    broadcast_4 = f16[2,4,8] broadcast(constant_4), dimensions={2}
    multiply_1 = f16[2,4,8] multiply(broadcast_3, broadcast_4)
    multiply_2 = f16[2,4,8] multiply(Arg_2, multiply_1)
    constant_5 = f16[8]{0} constant({1,1,1,1,1,1,1,1})
    broadcast_5 = f16[2,4,8] broadcast(constant_5), dimensions={2}
    reshape_4 = f16[2,4] reshape(reshape_0)
    broadcast_6 = f16[2,4,8] broadcast(reshape_4), dimensions={0,1}
    multiply_3 = f16[2,4,8] multiply(multiply_1, broadcast_6)
    subtract_1 = f16[2,4,8] subtract(broadcast_5, multiply_3)
    ROOT add_1 = f16[2,4,8] add(multiply_2, subtract_1)
  }
 )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

// Test for reversed inputs
TEST_F(LayerNormTest, LayerNormTest2_F16) {
  if (!xla::cpu::IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = f16[2,4,8] parameter(0), sharding={replicated}
    convert_0 = f32[2,4,8] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[2,4] reduce(convert_0, convert_1), dimensions={2}, to_apply=region_add
    constant_1 = s32[] constant(8)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[2,4] broadcast(convert_2), dimensions={}
    divide_0 = f32[2,4] divide(reduce_0, broadcast_0)
    convert_3 = f16[2,4] convert(divide_0)
    reshape_0 = f16[2,4,1] reshape(convert_3)
    reshape_1 = f16[2,4] reshape(reshape_0)
    broadcast_1 = f16[2,4,8] broadcast(reshape_1), dimensions={0,1}
    subtract_0 = f16[2,4,8] subtract(broadcast_1, Arg_2)
    multiply_0 = f16[2,4,8] multiply(subtract_0, subtract_0)
    convert_4 = f32[2,4,8] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_1 = f32[2,4] reduce(convert_4, convert_5), dimensions={2}, to_apply=region_add
    constant_3 = s32[] constant(8)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[2,4] broadcast(convert_6), dimensions={}
    divide_1 = f32[2,4] divide(reduce_1, broadcast_2)
    convert_7 = f16[2,4] convert(divide_1)
    reshape_2 = f16[2,4,1] reshape(convert_7)
    rsqrt_0 = f16[2,4,1] rsqrt(reshape_2)
    reshape_3 = f16[2,4] reshape(rsqrt_0)
    broadcast_3 = f16[2,4,8] broadcast(reshape_3), dimensions={0,1}
    constant_4 = f16[8] constant({1,1,1,1,1,1,1,1})
    broadcast_4 = f16[2,4,8] broadcast(constant_4), dimensions={2}
    multiply_1 = f16[2,4,8] multiply(broadcast_3, broadcast_4)
    multiply_2 = f16[2,4,8] multiply(multiply_1, Arg_2)
    constant_5 = f16[8] constant({1,1,1,1,1,1,1,1})
    broadcast_5 = f16[2,4,8] broadcast(constant_5), dimensions={2}
    reshape_4 = f16[2,4] reshape(reshape_0)
    broadcast_6 = f16[2,4,8] broadcast(reshape_4), dimensions={0,1}
    multiply_3 = f16[2,4,8] multiply(multiply_1, broadcast_6)
    subtract_1 = f16[2,4,8] subtract(broadcast_5, multiply_3)
    ROOT add_1 = f16[2,4,8] add(multiply_2, subtract_1)
  }
 )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

// Test case encountered in models like TFViTForImageClassification in
// HuggingFace
// (https://huggingface.co/docs/transformers/model_doc/vit#transformers.TFViTForImageClassification)
TEST_F(LayerNormTest, LayerNormTest1_BF16) {
  if (!xla::cpu::IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0.7555 = f32[] parameter(0)
    Arg_1.7556 = f32[] parameter(1)
    ROOT add.7557 = f32[] add(Arg_0.7555, Arg_1.7556)
  }
  ENTRY main {
    Arg_0.1 = bf16[160,197,768] parameter(0), sharding={replicated}
    Arg_0.2 = bf16[768] parameter(1), sharding={replicated}
    Arg_0.3 = bf16[768] parameter(2), sharding={replicated}
    convert.80 = f32[160,197,768] convert(Arg_0.1)
    constant.81 = f32[] constant(0)
    convert.82 = f32[] convert(constant.81)
    reduce.87 = f32[160,197] reduce(convert.80, convert.82), dimensions={2}, to_apply=region_add
    constant.88 = s32[] constant(768)
    convert.89 = f32[] convert(constant.88)
    broadcast.90 = f32[160,197] broadcast(convert.89), dimensions={}
    divide.91 = f32[160,197] divide(reduce.87, broadcast.90)
    convert.92 = bf16[160,197] convert(divide.91)
    reshape.93 = bf16[160,197,1] reshape(convert.92)
    reshape.94 = bf16[160,197] reshape(reshape.93)
    broadcast.95 = bf16[160,197,768] broadcast(reshape.94), dimensions={0,1}
    subtract.96 = bf16[160,197,768] subtract(Arg_0.1, broadcast.95)
    multiply.97 = bf16[160,197,768] multiply(subtract.96, subtract.96)
    convert.98 = f32[160,197,768] convert(multiply.97)
    constant.99 = f32[] constant(0)
    convert.100 = f32[] convert(constant.99)
    reduce.105 = f32[160,197] reduce(convert.98, convert.100), dimensions={2}, to_apply=region_add
    constant.106 = s32[] constant(768)
    convert.107 = f32[] convert(constant.106)
    broadcast.108 = f32[160,197] broadcast(convert.107), dimensions={}
    divide.109 = f32[160,197] divide(reduce.105, broadcast.108)
    convert.110 = bf16[160,197] convert(divide.109)
    reshape.111 = bf16[160,197,1] reshape(convert.110)
    constant.112 = bf16[] constant(1.002e-12)
    broadcast.113 = bf16[160,197,1] broadcast(constant.112), dimensions={}
    add.114 = bf16[160,197,1] add(reshape.111, broadcast.113)
    rsqrt.115 = bf16[160,197,1] rsqrt(add.114)
    reshape.118 = bf16[160,197] reshape(rsqrt.115)
    broadcast.119 = bf16[160,197,768] broadcast(reshape.118), dimensions={0,1}
    broadcast.117 = bf16[160,197,768] broadcast(Arg_0.2), dimensions={2}
    multiply.120 = bf16[160,197,768] multiply(broadcast.119, broadcast.117)
    multiply.121 = bf16[160,197,768] multiply(Arg_0.1, multiply.120)
    broadcast.126 = bf16[160,197,768] broadcast(Arg_0.3), dimensions={2}
    reshape.122 = bf16[160,197] reshape(reshape.93)
    broadcast.123 = bf16[160,197,768] broadcast(reshape.122), dimensions={0,1}
    multiply.124 = bf16[160,197,768] multiply(multiply.120, broadcast.123)
    subtract.127 = bf16[160,197,768] subtract(broadcast.126, multiply.124)
    ROOT add.128 = bf16[160,197,768] add(multiply.121, subtract.127)
  }
 )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

#endif  // INTEL_MKL

// Ensure at least one test case is linked to avoid test failures.
TEST(Dummy, Test) {}

}  // namespace
}  // namespace xla


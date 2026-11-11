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

#include <string>

#include "xla/error_spec.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class LayerNormTest : public HloInterpreterReferenceMixin<HloTestBase> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloInterpreterReferenceMixin::GetDebugOptionsForTest();
    debug_options.set_xla_cpu_experimental_onednn_custom_call(true);
    debug_options.clear_xla_cpu_experimental_ynn_fusion_type();
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
  const char* onednn_no_layer_norm_ =
      R"(
  ; CHECK-NOT: custom_call_target="__onednn$layernorm"
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
    reduce.297 = f32[84,197]{1,0} reduce(convert.290, convert.292),
                 dimensions={2}, to_apply=region_add
    constant.298 = s32[] constant(768)
    convert.299 = f32[] convert(constant.298)
    broadcast.300 = f32[84,197]{1,0} broadcast(convert.299), dimensions={}
    divide.301 = f32[84,197]{1,0} divide(reduce.297, broadcast.300)
    convert.302 = f32[84,197]{1,0} convert(divide.301)
    reshape.303 = f32[84,197,1]{2,1,0} reshape(convert.302)
    reshape.304 = f32[84,197]{1,0} reshape(reshape.303)
    broadcast.305 = f32[84,197,768]{2,1,0} broadcast(reshape.304),
                    dimensions={0,1}
    subtract.306 = f32[84,197,768]{2,1,0} subtract(Arg_0.1, broadcast.305)
    multiply.307 = f32[84,197,768]{2,1,0} multiply(subtract.306, subtract.306)
    convert.308 = f32[84,197,768]{2,1,0} convert(multiply.307)
    constant.309 = f32[] constant(0)
    convert.310 = f32[] convert(constant.309)
    reduce.315 = f32[84,197]{1,0} reduce(convert.308, convert.310),
                 dimensions={2}, to_apply=region_add
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
    broadcast.329 = f32[84,197,768]{2,1,0} broadcast(reshape.328),
                    dimensions={0,1}
    broadcast.327 = f32[84,197,768]{2,1,0} broadcast(Arg_0.2), dimensions={2}
    multiply.330 = f32[84,197,768]{2,1,0} multiply(broadcast.329, broadcast.327)
    multiply.331 = f32[84,197,768]{2,1,0} multiply(Arg_0.1, multiply.330)
    broadcast.336 = f32[84,197,768]{2,1,0} broadcast(Arg_0.3), dimensions={2}
    reshape.332 = f32[84,197]{1,0} reshape(reshape.303)
    broadcast.333 = f32[84,197,768]{2,1,0} broadcast(reshape.332),
                    dimensions={0,1}
    multiply.334 = f32[84,197,768]{2,1,0} multiply(multiply.330, broadcast.333)
    subtract.337 = f32[84,197,768]{2,1,0} subtract(broadcast.336, multiply.334)
)";
};

TEST_F(LayerNormTest, LayerNormTest0_FP32) {
  std::string layer_norm_module_str = R"(
      HloModule layer_norm.test, entry_computation_layout={(f32[84,197,768]{2,1,0}, f32[768]{0}, f32[768]{0})->f32[84,197,768]{2,1,0}})" +
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
    reduce_0 = f32[2,4] reduce(convert_0, convert_1), dimensions={2},
               to_apply=region_add
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
    reduce_2 = f32[2,4] reduce(convert_4, convert_5), dimensions={2},
               to_apply=region_add
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
    reduce_0 = f32[2,4] reduce(convert_0, convert_1), dimensions={2},
               to_apply=region_add
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
    reduce_1 = f32[2,4] reduce(convert_4, convert_5), dimensions={2},
               to_apply=region_add
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
    reduce.87 = f32[160,197] reduce(convert.80, convert.82), dimensions={2},
                to_apply=region_add
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
    reduce.105 = f32[160,197] reduce(convert.98, convert.100), dimensions={2},
                 to_apply=region_add
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

// Max(Mean(D^2) - Mean(D)^2, 0) with
// D = X - Mean(X), as emitted by tf.nn.moments
TEST_F(LayerNormTest, LayerNormTwoPassVariance_FP32) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = f32[4,8,16] parameter(0), sharding={replicated}
    Arg_3 = f32[16] parameter(1), sharding={replicated}
    Arg_4 = f32[16] parameter(2), sharding={replicated}

    // mean = Mean(X)
    convert_0 = f32[4,8,16] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[4,8] reduce(convert_0, convert_1), dimensions={2},
               to_apply=region_add
    constant_1 = s32[] constant(16)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[4,8] broadcast(convert_2), dimensions={}
    divide_0 = f32[4,8] divide(reduce_0, broadcast_0)
    convert_3 = f32[4,8] convert(divide_0)
    reshape_0 = f32[4,8,1] reshape(convert_3)

    // D = X - mean
    reshape_1 = f32[4,8] reshape(reshape_0)
    broadcast_1 = f32[4,8,16] broadcast(reshape_1), dimensions={0,1}
    subtract_0 = f32[4,8,16] subtract(Arg_2, broadcast_1)

    // Mean(D^2)
    multiply_0 = f32[4,8,16] multiply(subtract_0, subtract_0)
    convert_4 = f32[4,8,16] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_1 = f32[4,8] reduce(convert_4, convert_5), dimensions={2},
               to_apply=region_add
    constant_3 = s32[] constant(16)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[4,8] broadcast(convert_6), dimensions={}
    divide_1 = f32[4,8] divide(reduce_1, broadcast_2)
    convert_7 = f32[4,8] convert(divide_1)
    reshape_2 = f32[4,8,1] reshape(convert_7)

    // Mean(D)^2
    convert_8 = f32[4,8,16] convert(subtract_0)
    constant_4 = f32[] constant(0)
    convert_9 = f32[] convert(constant_4)
    reduce_2 = f32[4,8] reduce(convert_8, convert_9), dimensions={2},
               to_apply=region_add
    constant_5 = s32[] constant(16)
    convert_10 = f32[] convert(constant_5)
    broadcast_3 = f32[4,8] broadcast(convert_10), dimensions={}
    divide_2 = f32[4,8] divide(reduce_2, broadcast_3)
    convert_11 = f32[4,8] convert(divide_2)
    reshape_3 = f32[4,8,1] reshape(convert_11)
    multiply_1 = f32[4,8,1] multiply(reshape_3, reshape_3)

    // variance = Max(Mean(D^2) - Mean(D)^2, 0)
    subtract_1 = f32[4,8,1] subtract(reshape_2, multiply_1)
    constant_6 = f32[] constant(0)
    broadcast_4 = f32[4,8,1] broadcast(constant_6), dimensions={}
    maximum_0 = f32[4,8,1] maximum(subtract_1, broadcast_4)

    constant_7 = f32[] constant(1e-12)
    broadcast_5 = f32[4,8,1] broadcast(constant_7), dimensions={}
    add_1 = f32[4,8,1] add(broadcast_5, maximum_0)
    rsqrt_0 = f32[4,8,1] rsqrt(add_1)
    reshape_4 = f32[4,8] reshape(rsqrt_0)
    broadcast_6 = f32[4,8,16] broadcast(reshape_4), dimensions={0,1}
    broadcast_7 = f32[4,8,16] broadcast(Arg_3), dimensions={2}
    multiply_2 = f32[4,8,16] multiply(broadcast_6, broadcast_7)
    multiply_3 = f32[4,8,16] multiply(Arg_2, multiply_2)
    broadcast_8 = f32[4,8,16] broadcast(Arg_4), dimensions={2}
    reshape_5 = f32[4,8] reshape(reshape_0)
    broadcast_9 = f32[4,8,16] broadcast(reshape_5), dimensions={0,1}
    multiply_4 = f32[4,8,16] multiply(multiply_2, broadcast_9)
    subtract_2 = f32[4,8,16] subtract(broadcast_8, multiply_4)
    ROOT add_2 = f32[4,8,16] add(multiply_3, subtract_2)
  }
  )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

TEST_F(LayerNormTest, LayerNormTwoPassVariance_BF16) {
  if (!xla::cpu::IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = bf16[4,8,16] parameter(0), sharding={replicated}
    Arg_3 = bf16[16] parameter(1), sharding={replicated}
    Arg_4 = bf16[16] parameter(2), sharding={replicated}

    // mean = Mean(X)
    convert_0 = f32[4,8,16] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[4,8] reduce(convert_0, convert_1), dimensions={2},
               to_apply=region_add
    constant_1 = s32[] constant(16)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[4,8] broadcast(convert_2), dimensions={}
    divide_0 = f32[4,8] divide(reduce_0, broadcast_0)
    convert_3 = bf16[4,8] convert(divide_0)
    reshape_0 = bf16[4,8,1] reshape(convert_3)

    // D = X - mean
    reshape_1 = bf16[4,8] reshape(reshape_0)
    broadcast_1 = bf16[4,8,16] broadcast(reshape_1), dimensions={0,1}
    subtract_0 = bf16[4,8,16] subtract(Arg_2, broadcast_1)

    // Mean(D^2)
    multiply_0 = bf16[4,8,16] multiply(subtract_0, subtract_0)
    convert_4 = f32[4,8,16] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_1 = f32[4,8] reduce(convert_4, convert_5), dimensions={2},
               to_apply=region_add
    constant_3 = s32[] constant(16)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[4,8] broadcast(convert_6), dimensions={}
    divide_1 = f32[4,8] divide(reduce_1, broadcast_2)
    convert_7 = bf16[4,8] convert(divide_1)
    reshape_2 = bf16[4,8,1] reshape(convert_7)

    // Mean(D)^2
    convert_8 = f32[4,8,16] convert(subtract_0)
    constant_4 = f32[] constant(0)
    convert_9 = f32[] convert(constant_4)
    reduce_2 = f32[4,8] reduce(convert_8, convert_9), dimensions={2},
               to_apply=region_add
    constant_5 = s32[] constant(16)
    convert_10 = f32[] convert(constant_5)
    broadcast_3 = f32[4,8] broadcast(convert_10), dimensions={}
    divide_2 = f32[4,8] divide(reduce_2, broadcast_3)
    convert_11 = bf16[4,8] convert(divide_2)
    reshape_3 = bf16[4,8,1] reshape(convert_11)
    multiply_1 = bf16[4,8,1] multiply(reshape_3, reshape_3)

    // variance = Max(Mean(D^2) - Mean(D)^2, 0)
    subtract_1 = bf16[4,8,1] subtract(reshape_2, multiply_1)
    constant_6 = bf16[] constant(0)
    broadcast_4 = bf16[4,8,1] broadcast(constant_6), dimensions={}
    maximum_0 = bf16[4,8,1] maximum(subtract_1, broadcast_4)

    constant_7 = bf16[] constant(1.002e-12)
    broadcast_5 = bf16[4,8,1] broadcast(constant_7), dimensions={}
    add_1 = bf16[4,8,1] add(maximum_0, broadcast_5)
    rsqrt_0 = bf16[4,8,1] rsqrt(add_1)
    reshape_4 = bf16[4,8] reshape(rsqrt_0)
    broadcast_6 = bf16[4,8,16] broadcast(reshape_4), dimensions={0,1}
    broadcast_7 = bf16[4,8,16] broadcast(Arg_3), dimensions={2}
    multiply_2 = bf16[4,8,16] multiply(broadcast_6, broadcast_7)
    multiply_3 = bf16[4,8,16] multiply(Arg_2, multiply_2)
    broadcast_8 = bf16[4,8,16] broadcast(Arg_4), dimensions={2}
    reshape_5 = bf16[4,8] reshape(reshape_0)
    broadcast_9 = bf16[4,8,16] broadcast(reshape_5), dimensions={0,1}
    multiply_4 = bf16[4,8,16] multiply(multiply_2, broadcast_9)
    subtract_2 = bf16[4,8,16] subtract(broadcast_8, multiply_4)
    ROOT add_2 = bf16[4,8,16] add(multiply_3, subtract_2)
  }
 )";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str, onednn_layer_norm_);
}

// Mean(D^2) and Mean(D)^2 are computed from different centered inputs, so this
// is not the variance of a single LayerNorm and must not be rewritten.
TEST_F(LayerNormTest, LayerNormTwoPassVarianceMismatchedCenteredInputs) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = f32[4,8,16] parameter(0), sharding={replicated}
    Arg_3 = f32[16] parameter(1), sharding={replicated}
    Arg_4 = f32[16] parameter(2), sharding={replicated}
    Arg_5 = f32[4,8,16] parameter(3), sharding={replicated}

    // mean = Mean(X)
    convert_0 = f32[4,8,16] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[4,8] reduce(convert_0, convert_1), dimensions={2},
               to_apply=region_add
    constant_1 = s32[] constant(16)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[4,8] broadcast(convert_2), dimensions={}
    divide_0 = f32[4,8] divide(reduce_0, broadcast_0)
    convert_3 = f32[4,8] convert(divide_0)
    reshape_0 = f32[4,8,1] reshape(convert_3)

    // D = X - mean
    reshape_1 = f32[4,8] reshape(reshape_0)
    broadcast_1 = f32[4,8,16] broadcast(reshape_1), dimensions={0,1}
    subtract_0 = f32[4,8,16] subtract(Arg_2, broadcast_1)

    // Mean(D^2)
    multiply_0 = f32[4,8,16] multiply(subtract_0, subtract_0)
    convert_4 = f32[4,8,16] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_1 = f32[4,8] reduce(convert_4, convert_5), dimensions={2},
               to_apply=region_add
    constant_3 = s32[] constant(16)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[4,8] broadcast(convert_6), dimensions={}
    divide_1 = f32[4,8] divide(reduce_1, broadcast_2)
    convert_7 = f32[4,8] convert(divide_1)
    reshape_2 = f32[4,8,1] reshape(convert_7)

    // D' = Y - Mean(Y)
    convert_12 = f32[4,8,16] convert(Arg_5)
    reduce_3 = f32[4,8] reduce(convert_12, convert_1), dimensions={2},
               to_apply=region_add
    divide_3 = f32[4,8] divide(reduce_3, broadcast_0)
    convert_13 = f32[4,8] convert(divide_3)
    reshape_6 = f32[4,8,1] reshape(convert_13)
    reshape_7 = f32[4,8] reshape(reshape_6)
    broadcast_10 = f32[4,8,16] broadcast(reshape_7), dimensions={0,1}
    subtract_3 = f32[4,8,16] subtract(Arg_5, broadcast_10)

    // Mean(D')^2
    convert_8 = f32[4,8,16] convert(subtract_3)
    constant_4 = f32[] constant(0)
    convert_9 = f32[] convert(constant_4)
    reduce_2 = f32[4,8] reduce(convert_8, convert_9), dimensions={2},
               to_apply=region_add
    constant_5 = s32[] constant(16)
    convert_10 = f32[] convert(constant_5)
    broadcast_3 = f32[4,8] broadcast(convert_10), dimensions={}
    divide_2 = f32[4,8] divide(reduce_2, broadcast_3)
    convert_11 = f32[4,8] convert(divide_2)
    reshape_3 = f32[4,8,1] reshape(convert_11)
    multiply_1 = f32[4,8,1] multiply(reshape_3, reshape_3)

    // Max(Mean(D^2) - Mean(D')^2, 0)
    subtract_1 = f32[4,8,1] subtract(reshape_2, multiply_1)
    constant_6 = f32[] constant(0)
    broadcast_4 = f32[4,8,1] broadcast(constant_6), dimensions={}
    maximum_0 = f32[4,8,1] maximum(subtract_1, broadcast_4)

    constant_7 = f32[] constant(1e-12)
    broadcast_5 = f32[4,8,1] broadcast(constant_7), dimensions={}
    add_1 = f32[4,8,1] add(broadcast_5, maximum_0)
    rsqrt_0 = f32[4,8,1] rsqrt(add_1)
    reshape_4 = f32[4,8] reshape(rsqrt_0)
    broadcast_6 = f32[4,8,16] broadcast(reshape_4), dimensions={0,1}
    broadcast_7 = f32[4,8,16] broadcast(Arg_3), dimensions={2}
    multiply_2 = f32[4,8,16] multiply(broadcast_6, broadcast_7)
    multiply_3 = f32[4,8,16] multiply(Arg_2, multiply_2)
    broadcast_8 = f32[4,8,16] broadcast(Arg_4), dimensions={2}
    reshape_5 = f32[4,8] reshape(reshape_0)
    broadcast_9 = f32[4,8,16] broadcast(reshape_5), dimensions={0,1}
    multiply_4 = f32[4,8,16] multiply(multiply_2, broadcast_9)
    subtract_2 = f32[4,8,16] subtract(broadcast_8, multiply_4)
    ROOT add_2 = f32[4,8,16] add(multiply_3, subtract_2)
  }
  )";

  MatchOptimizedHlo(layer_norm_module_str, onednn_no_layer_norm_);
}

// The variance is computed from D = Y - Mean(Y) while the normalized input is
// X, so this must not be rewritten into LayerNorm(X).
TEST_F(LayerNormTest, LayerNormCenteredInputNotSrc) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = f32[4,8,16] parameter(0), sharding={replicated}
    Arg_3 = f32[16] parameter(1), sharding={replicated}
    Arg_4 = f32[16] parameter(2), sharding={replicated}
    Arg_5 = f32[4,8,16] parameter(3), sharding={replicated}

    // mean = Mean(X)
    convert_0 = f32[4,8,16] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[4,8] reduce(convert_0, convert_1), dimensions={2},
               to_apply=region_add
    constant_1 = s32[] constant(16)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[4,8] broadcast(convert_2), dimensions={}
    divide_0 = f32[4,8] divide(reduce_0, broadcast_0)
    convert_3 = f32[4,8] convert(divide_0)
    reshape_0 = f32[4,8,1] reshape(convert_3)

    // D = Y - Mean(Y)
    convert_12 = f32[4,8,16] convert(Arg_5)
    reduce_3 = f32[4,8] reduce(convert_12, convert_1), dimensions={2},
               to_apply=region_add
    divide_3 = f32[4,8] divide(reduce_3, broadcast_0)
    convert_13 = f32[4,8] convert(divide_3)
    reshape_6 = f32[4,8,1] reshape(convert_13)
    reshape_1 = f32[4,8] reshape(reshape_6)
    broadcast_1 = f32[4,8,16] broadcast(reshape_1), dimensions={0,1}
    subtract_0 = f32[4,8,16] subtract(Arg_5, broadcast_1)

    // Mean(D^2)
    multiply_0 = f32[4,8,16] multiply(subtract_0, subtract_0)
    convert_4 = f32[4,8,16] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_1 = f32[4,8] reduce(convert_4, convert_5), dimensions={2},
               to_apply=region_add
    constant_3 = s32[] constant(16)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[4,8] broadcast(convert_6), dimensions={}
    divide_1 = f32[4,8] divide(reduce_1, broadcast_2)
    convert_7 = f32[4,8] convert(divide_1)
    reshape_2 = f32[4,8,1] reshape(convert_7)

    constant_7 = f32[] constant(1e-12)
    broadcast_5 = f32[4,8,1] broadcast(constant_7), dimensions={}
    add_1 = f32[4,8,1] add(reshape_2, broadcast_5)
    rsqrt_0 = f32[4,8,1] rsqrt(add_1)
    reshape_4 = f32[4,8] reshape(rsqrt_0)
    broadcast_6 = f32[4,8,16] broadcast(reshape_4), dimensions={0,1}
    broadcast_7 = f32[4,8,16] broadcast(Arg_3), dimensions={2}
    multiply_2 = f32[4,8,16] multiply(broadcast_6, broadcast_7)
    multiply_3 = f32[4,8,16] multiply(Arg_2, multiply_2)
    broadcast_8 = f32[4,8,16] broadcast(Arg_4), dimensions={2}
    reshape_5 = f32[4,8] reshape(reshape_0)
    broadcast_9 = f32[4,8,16] broadcast(reshape_5), dimensions={0,1}
    multiply_4 = f32[4,8,16] multiply(multiply_2, broadcast_9)
    subtract_2 = f32[4,8,16] subtract(broadcast_8, multiply_4)
    ROOT add_2 = f32[4,8,16] add(multiply_3, subtract_2)
  }
  )";

  MatchOptimizedHlo(layer_norm_module_str, onednn_no_layer_norm_);
}

// Normalization over a non-last axis, e.g.
// tf.keras.layers.LayerNormalization(axis=1) on [N, C, W]. oneDNN only
// normalizes over the last dimension, so this must not be rewritten.
TEST_F(LayerNormTest, LayerNormNonLastAxis) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test
  region_add {
    Arg_0 = f32[] parameter(0)
    Arg_1 = f32[] parameter(1)
    ROOT add_0 = f32[] add(Arg_0, Arg_1)
  }
  ENTRY main {
    Arg_2 = f32[4,8,8] parameter(0), sharding={replicated}
    Arg_3 = f32[8] parameter(1), sharding={replicated}
    Arg_4 = f32[8] parameter(2), sharding={replicated}

    // mean = Mean(X) over axis 1
    convert_0 = f32[4,8,8] convert(Arg_2)
    constant_0 = f32[] constant(0)
    convert_1 = f32[] convert(constant_0)
    reduce_0 = f32[4,8] reduce(convert_0, convert_1), dimensions={1},
               to_apply=region_add
    constant_1 = s32[] constant(8)
    convert_2 = f32[] convert(constant_1)
    broadcast_0 = f32[4,8] broadcast(convert_2), dimensions={}
    divide_0 = f32[4,8] divide(reduce_0, broadcast_0)
    convert_3 = f32[4,8] convert(divide_0)
    reshape_0 = f32[4,1,8] reshape(convert_3)

    // D = X - mean
    reshape_1 = f32[4,8] reshape(reshape_0)
    broadcast_1 = f32[4,8,8] broadcast(reshape_1), dimensions={0,2}
    subtract_0 = f32[4,8,8] subtract(Arg_2, broadcast_1)

    // Mean(D^2) over axis 1
    multiply_0 = f32[4,8,8] multiply(subtract_0, subtract_0)
    convert_4 = f32[4,8,8] convert(multiply_0)
    constant_2 = f32[] constant(0)
    convert_5 = f32[] convert(constant_2)
    reduce_1 = f32[4,8] reduce(convert_4, convert_5), dimensions={1},
               to_apply=region_add
    constant_3 = s32[] constant(8)
    convert_6 = f32[] convert(constant_3)
    broadcast_2 = f32[4,8] broadcast(convert_6), dimensions={}
    divide_1 = f32[4,8] divide(reduce_1, broadcast_2)
    convert_7 = f32[4,8] convert(divide_1)
    reshape_2 = f32[4,1,8] reshape(convert_7)

    constant_7 = f32[] constant(1e-12)
    broadcast_5 = f32[4,1,8] broadcast(constant_7), dimensions={}
    add_1 = f32[4,1,8] add(reshape_2, broadcast_5)
    rsqrt_0 = f32[4,1,8] rsqrt(add_1)
    reshape_4 = f32[4,8] reshape(rsqrt_0)
    broadcast_6 = f32[4,8,8] broadcast(reshape_4), dimensions={0,2}
    broadcast_7 = f32[4,8,8] broadcast(Arg_3), dimensions={1}
    multiply_2 = f32[4,8,8] multiply(broadcast_6, broadcast_7)
    multiply_3 = f32[4,8,8] multiply(Arg_2, multiply_2)
    broadcast_8 = f32[4,8,8] broadcast(Arg_4), dimensions={1}
    reshape_5 = f32[4,8] reshape(reshape_0)
    broadcast_9 = f32[4,8,8] broadcast(reshape_5), dimensions={0,2}
    multiply_4 = f32[4,8,8] multiply(multiply_2, broadcast_9)
    subtract_2 = f32[4,8,8] subtract(broadcast_8, multiply_4)
    ROOT add_2 = f32[4,8,8] add(multiply_3, subtract_2)
  }
  )";

  MatchOptimizedHlo(layer_norm_module_str, onednn_no_layer_norm_);
}

}  // namespace
}  // namespace xla

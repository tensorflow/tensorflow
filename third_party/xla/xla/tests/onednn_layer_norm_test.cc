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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class LayerNormTest : public HloTestBase {};

TEST_F(LayerNormTest, SimpleTest) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm.test, entry_computation_layout={(f32[4,1,256]{2,1,0}, f32[1,1,256]{2,1,0}, f32[1,1,256]{2,1,0})->f32[4,1,256]{2,1,0}}

  region_add {
    Arg_0.7555 = f32[] parameter(0)
    Arg_1.7556 = f32[] parameter(1)
    ROOT add.7557 = f32[] add(Arg_0.7555, Arg_1.7556)
  }

  ENTRY main {
    Arg_0.1 = f32[4,1,256]{2,1,0} parameter(0), sharding={replicated}
    Arg_0.2 = f32[1,1,256]{2,1,0} parameter(1), sharding={replicated}
    Arg_0.3 = f32[1,1,256]{2,1,0} parameter(2), sharding={replicated}
    reshape.9744 = f32[1,4,1,256]{3,2,1,0} reshape(Arg_0.1)
    multiply.9743 = f32[4,1,256]{2,1,0} multiply(Arg_0.1, Arg_0.1)
    reshape.9745 = f32[1,4,1,256]{3,2,1,0} reshape(multiply.9743)
    concatenate.9746 = f32[2,4,1,256]{3,2,1,0} concatenate(reshape.9744, reshape.9745), dimensions={0}
    constant.9731 = f32[] constant(0)
    reduce.9747 = f32[2,4,1]{2,1,0} reduce(concatenate.9746, constant.9731), dimensions={3}, to_apply=region_add
    constant.9729 = f32[] constant(256)
    broadcast.9730 = f32[2,4,1]{2,1,0} broadcast(constant.9729), dimensions={}
    divide.9748 = f32[2,4,1]{2,1,0} divide(reduce.9747, broadcast.9730)
    slice.9749 = f32[1,4,1]{2,1,0} slice(divide.9748), slice={[0:1], [0:4], [0:1]}
    reshape.9756 = f32[4,1,1]{2,1,0} reshape(slice.9749)
    broadcast.9758 = f32[4,1,1]{2,1,0} broadcast(reshape.9756), dimensions={0,1,2}
    reshape.9759 = f32[4,1]{1,0} reshape(broadcast.9758)
    broadcast.9760 = f32[4,1,256]{2,1,0} broadcast(reshape.9759), dimensions={0,1}
    subtract.9761 = f32[4,1,256]{2,1,0} subtract(Arg_0.1, broadcast.9760)
    slice.9751 = f32[1,4,1]{2,1,0} slice(divide.9748), slice={[1:2], [0:4], [0:1]}
    reshape.9752 = f32[4,1]{1,0} reshape(slice.9751)
    reshape.9750 = f32[4,1]{1,0} reshape(slice.9749)
    multiply.9753 = f32[4,1]{1,0} multiply(reshape.9750, reshape.9750)
    subtract.9754 = f32[4,1]{1,0} subtract(reshape.9752, multiply.9753)
    constant.9727 = f32[] constant(0)
    broadcast.9728 = f32[4,1]{1,0} broadcast(constant.9727), dimensions={}
    maximum.9755 = f32[4,1]{1,0} maximum(subtract.9754, broadcast.9728)
    reshape.9757 = f32[4,1,1]{2,1,0} reshape(maximum.9755)
    constant.9725 = f32[] constant(1e-05)
    broadcast.9726 = f32[4,1,1]{2,1,0} broadcast(constant.9725), dimensions={}
    add.9762 = f32[4,1,1]{2,1,0} add(reshape.9757, broadcast.9726)
    rsqrt.9763 = f32[4,1,1]{2,1,0} rsqrt(add.9762)
    broadcast.9764 = f32[4,1,1]{2,1,0} broadcast(rsqrt.9763), dimensions={0,1,2}
    reshape.9765 = f32[4,1]{1,0} reshape(broadcast.9764)
    broadcast.9766 = f32[4,1,256]{2,1,0} broadcast(reshape.9765), dimensions={0,1}
    broadcast.9767 = f32[1,1,256]{2,1,0} broadcast(Arg_0.2), dimensions={0,1,2}
    reshape.9768 = f32[1,256]{1,0} reshape(broadcast.9767)
    broadcast.9769 = f32[4,1,256]{2,1,0} broadcast(reshape.9768), dimensions={1,2}
    multiply.9770 = f32[4,1,256]{2,1,0} multiply(broadcast.9766, broadcast.9769)
    multiply.9771 = f32[4,1,256]{2,1,0} multiply(subtract.9761, multiply.9770)
    broadcast.9772 = f32[1,1,256]{2,1,0} broadcast(Arg_0.3), dimensions={0,1,2}
    reshape.9773 = f32[1,256]{1,0} reshape(broadcast.9772)
    broadcast.9774 = f32[4,1,256]{2,1,0} broadcast(reshape.9773), dimensions={1,2}
    ROOT add.9775 = f32[4,1,256]{2,1,0} add(multiply.9771, broadcast.9774)
  }  
)";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(layer_norm_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$layernorm",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "onednn_layer_norm_config":{
  ; CHECK-DAG:       "fused_ops":"SCALE_AND_SHIFT"
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

TEST_F(LayerNormTest, SimpleTestBF16) {
  const char* layer_norm_module_str = R"(
  HloModule layer_norm_bf16.test, entry_computation_layout={(f32[768]{0}, f32[768]{0}, bf16[16,128,768]{2,1,0})->bf16[16,128,768]{2,1,0}}, allow_spmd_sharding_propagation_to_output={true}

  region_0.16 {
    Arg_0.17 = f32[] parameter(0)
    Arg_1.18 = f32[] parameter(1)
    ROOT add.19 = f32[] add(Arg_0.17, Arg_1.18)
  }

  ENTRY main.53 {
    Arg_2.3 = bf16[16,128,768]{2,1,0} parameter(2), sharding={replicated}
    convert.31 = f32[16,128,768]{2,1,0} convert(Arg_2.3)
    convert.11 = f32[16,128,768]{2,1,0} convert(Arg_2.3)
    reshape.13 = f32[1,16,128,768]{3,2,1,0} reshape(convert.11)
    multiply.12 = f32[16,128,768]{2,1,0} multiply(convert.11, convert.11)
    reshape.14 = f32[1,16,128,768]{3,2,1,0} reshape(multiply.12)
    concatenate.15 = f32[2,16,128,768]{3,2,1,0} concatenate(reshape.13, reshape.14), dimensions={0}
    constant.10 = f32[] constant(0)
    reduce.20 = f32[2,16,128]{2,1,0} reduce(concatenate.15, constant.10), dimensions={3}, to_apply=region_0.16
    constant.8 = f32[] constant(768)
    broadcast.9 = f32[2,16,128]{2,1,0} broadcast(constant.8), dimensions={}
    divide.21 = f32[2,16,128]{2,1,0} divide(reduce.20, broadcast.9)
    slice.22 = f32[1,16,128]{2,1,0} slice(divide.21), slice={[0:1], [0:16], [0:128]}
    reshape.29 = f32[16,128,1]{2,1,0} reshape(slice.22)
    broadcast.32 = f32[16,128,1]{2,1,0} broadcast(reshape.29), dimensions={0,1,2}
    reshape.33 = f32[16,128]{1,0} reshape(broadcast.32)
    broadcast.34 = f32[16,128,768]{2,1,0} broadcast(reshape.33), dimensions={0,1}
    subtract.35 = f32[16,128,768]{2,1,0} subtract(convert.31, broadcast.34)
    slice.24 = f32[1,16,128]{2,1,0} slice(divide.21), slice={[1:2], [0:16], [0:128]}
    reshape.25 = f32[16,128]{1,0} reshape(slice.24)
    reshape.23 = f32[16,128]{1,0} reshape(slice.22)
    multiply.26 = f32[16,128]{1,0} multiply(reshape.23, reshape.23)
    subtract.27 = f32[16,128]{1,0} subtract(reshape.25, multiply.26)
    constant.6 = f32[] constant(0)
    broadcast.7 = f32[16,128]{1,0} broadcast(constant.6), dimensions={}
    maximum.28 = f32[16,128]{1,0} maximum(subtract.27, broadcast.7)
    reshape.30 = f32[16,128,1]{2,1,0} reshape(maximum.28)
    constant.4 = f32[] constant(1e-06)
    broadcast.5 = f32[16,128,1]{2,1,0} broadcast(constant.4), dimensions={}
    add.36 = f32[16,128,1]{2,1,0} add(reshape.30, broadcast.5)
    rsqrt.37 = f32[16,128,1]{2,1,0} rsqrt(add.36)
    broadcast.39 = f32[16,128,1]{2,1,0} broadcast(rsqrt.37), dimensions={0,1,2}
    reshape.40 = f32[16,128]{1,0} reshape(broadcast.39)
    broadcast.41 = f32[16,128,768]{2,1,0} broadcast(reshape.40), dimensions={0,1}
    Arg_1.2 = f32[768]{0} parameter(1), sharding={replicated}
    reshape.38 = f32[1,1,768]{2,1,0} reshape(Arg_1.2)
    broadcast.42 = f32[1,1,768]{2,1,0} broadcast(reshape.38), dimensions={0,1,2}
    reshape.43 = f32[768]{0} reshape(broadcast.42)
    broadcast.44 = f32[16,128,768]{2,1,0} broadcast(reshape.43), dimensions={2}
    multiply.45 = f32[16,128,768]{2,1,0} multiply(broadcast.41, broadcast.44)
    multiply.46 = f32[16,128,768]{2,1,0} multiply(subtract.35, multiply.45)
    Arg_0.1 = f32[768]{0} parameter(0), sharding={replicated}
    reshape.47 = f32[1,1,768]{2,1,0} reshape(Arg_0.1)
    broadcast.48 = f32[1,1,768]{2,1,0} broadcast(reshape.47), dimensions={0,1,2}
    reshape.49 = f32[768]{0} reshape(broadcast.48)
    broadcast.50 = f32[16,128,768]{2,1,0} broadcast(reshape.49), dimensions={2}
    add.51 = f32[16,128,768]{2,1,0} add(multiply.46, broadcast.50)
    ROOT convert.52 = bf16[16,128,768]{2,1,0} convert(add.51)
  }
)";

  EXPECT_TRUE(RunAndCompare(layer_norm_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(layer_norm_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$layernorm",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "onednn_layer_norm_config":{
  ; CHECK-DAG:       "fused_ops":"SCALE_AND_SHIFT"
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

}  // namespace
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3

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

#include <utility>

#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/literal.h"
#include "xla/service/cpu/onednn_matmul_rewriter.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/cpu_info.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace cpu {

class MatmulTest : public HloTestBase {
 protected:
  const char* fused_matmul_bias_ = R"(
    ; CHECK:     custom_call_target="__onednn$matmul",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:     "onednn_matmul_config":{
    ; CHECK-DAG:       "fused_ops":["BIAS"]
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";
  const char* fused_matmul_binary_add_ = R"(
    ; CHECK:     custom_call_target="__onednn$matmul",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:     "onednn_matmul_config":{
    ; CHECK-DAG:       "fused_ops":["BINARY_ADD"]
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";
  const char* matmul_rewrite_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$matmul",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:     "onednn_matmul_config":{
    ; CHECK-DAG:       "fused_ops":[]
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";
  const char* fused_matmul_bias_gelu_tanh_ = R"(
    ; CHECK:     custom_call_target="__onednn$matmul",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:     "onednn_matmul_config":{
    ; CHECK-DAG:       "fused_ops":["BIAS","GELU_TANH"]
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";
};

TEST_F(MatmulTest, SimpleTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
    arg.0 = f32[32,8,128,64] parameter(0), parameter_replication={false}
    arg.1 = f32[32,8,64,128] parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[32,8,128,128] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, SimpleTestBF16) {
  // TODO(penporn): Refactor IsBF16SupportedByOneDNNOnThisCPU() from
  // tensorflow/core/graph/mkl_graph_util.h and call the function instead.
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16

  ENTRY matmul.test.bf16 {
    arg.0 = bf16[32,8,128,64] parameter(0), parameter_replication={false}
    arg.1 = bf16[32,8,64,128] parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = bf16[32,8,128,128] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, SimpleTestF16) {
  if (!IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.f16

  ENTRY matmul.test.f16 {
    arg.0 = f16[32,8,128,64] parameter(0), parameter_replication={false}
    arg.1 = f16[32,8,64,128] parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f16[32,8,128,128] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, SimpleTestF32TransposeB) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1

  ENTRY matmul.test.1 {
    arg.0 = f32[32,8,128,64]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[32,8,128,64]{3,1,2,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[32,8,128,128] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAddFusion1) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[32,32,40,30] parameter(0), parameter_replication={false}
    reshape.2 = f32[32,32,40,30] reshape(arg0.1)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[32,32,30,40] broadcast(constant.3), dimensions={}
    dot.7 = f32[32,32,40,40] dot(reshape.2, broadcast.4), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[40] broadcast(constant.5), dimensions={}
    broadcast.9 = f32[32,32,40,40] broadcast(broadcast.6), dimensions={3}
    add.10 = f32[32,32,40,40] add(dot.7, broadcast.9)
    reshape.11 = f32[32,32,40,40] reshape(add.10)
    tuple.12 = (f32[32,32,40,40]) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,32,40,40] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_binary_add_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAddFusion2) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[400,300] parameter(0), parameter_replication={false}
    reshape.2 = f32[400,300] reshape(arg0.1)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[300,400] broadcast(constant.3), dimensions={}
    dot.7 = f32[400,400] dot(reshape.2, broadcast.4), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0}
    reshape.1 = f32[400,1,400] reshape(dot.7)
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[400] broadcast(constant.5), dimensions={}
    broadcast.9 = f32[400,1,400] broadcast(broadcast.6), dimensions={2}
    add.10 = f32[400,1,400] add(reshape.1, broadcast.9)
    tuple.12 = (f32[400,1,400]) tuple(add.10)
    ROOT get-tuple-element.13 = f32[400,1,400] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_binary_add_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter1) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[32,32,40,30] parameter(0), parameter_replication={false}
    arg0.2 = f32[32,32,30,40] parameter(1), parameter_replication={false}
    arg0.3 = f32[32,32,40,40] parameter(2), parameter_replication={false}
    dot.7 = f32[32,32,40,40] dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    add.10 = f32[32,32,40,40] add(dot.7, arg0.3)
    reshape.11 = f32[32,32,40,40] reshape(add.10)
    tuple.12 = (f32[32,32,40,40]) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,32,40,40] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_binary_add_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter2) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[32,32,40,30] parameter(0), parameter_replication={false}
    arg0.2 = f32[32,32,30,40] parameter(1), parameter_replication={false}
    arg0.3 = f32[40]{0} parameter(2), parameter_replication={false}
    dot.7 = f32[32,32,40,40] dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = f32[32,32,40,40] broadcast(arg0.3), dimensions={3}
    add.10 = f32[32,32,40,40] add(dot.7, broad.1)
    reshape.11 = f32[32,32,40,40] reshape(add.10)
    tuple.12 = (f32[32,32,40,40]) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,32,40,40] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter2D) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[2,2,400,30] parameter(0), parameter_replication={false}
    arg0.2 = f32[2,2,30,400] parameter(1), parameter_replication={false}
    arg0.3 = f32[2,400] parameter(2), parameter_replication={false}
    dot.7 = f32[2,2,400,400] dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = f32[2,2,400,400] broadcast(arg0.3), dimensions={0,3}
    add.10 = f32[2,2,400,400] add(dot.7, broad.1)
    reshape.11 = f32[2,2,400,400] reshape(add.10)
    tuple.12 = (f32[2,2,400,400]) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[2,2,400,400] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_binary_add_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter2D1B) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[1,2,400,30] parameter(0), parameter_replication={false}
    arg0.2 = f32[1,2,30,400] parameter(1), parameter_replication={false}
    arg0.3 = f32[1,400] parameter(2), parameter_replication={false}
    dot.7 = f32[1,2,400,400] dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = f32[1,2,400,400] broadcast(arg0.3), dimensions={0,3}
    add.10 = f32[1,2,400,400] add(dot.7, broad.1)
    reshape.11 = f32[1,2,400,400] reshape(add.10)
    tuple.12 = (f32[1,2,400,400]) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[1,2,400,400] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter3) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32

  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[16,128,768] parameter(0), sharding={replicated}
    arg0.2 = f32[768,768] parameter(1), sharding={replicated}
    dot.84 = f32[16,128,768] dot(arg0.1, arg0.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    arg0.3 = f32[768]{0} parameter(2), sharding={replicated}
    reshape.85 = f32[1,1,768] reshape(arg0.3)
    broadcast.86 = f32[1,1,768] broadcast(reshape.85), dimensions={0,1,2}
    reshape.87 = f32[768]{0} reshape(broadcast.86)
    broadcast.88 = f32[16,128,768] broadcast(reshape.87), dimensions={2}
    ROOT add.89 = f32[16,128,768] add(dot.84, broadcast.88)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32TransposeBWithBiasAddFusion) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1

  ENTRY matmul.test.1 {
    arg.0 = f32[32,8,4,16]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[32,8,16,16]{3,1,2,0} parameter(1), parameter_replication={false}
    dot.7 = f32[32,8,4,16]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[16]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[32,8,4,16]{3,2,1,0} broadcast(broadcast.6), dimensions={3}
    add.10 = f32[32,8,4,16]{3,2,1,0} add(dot.7, broadcast.9)
    reshape.11 = f32[32,8,4,16]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[32,8,4,16]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,8,4,16]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_binary_add_);
}

TEST_F(MatmulTest, F32BiasAddFusionNonCompatibleBias) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.1 {
    arg.0 = f32[12288,2] parameter(0), parameter_replication={false}
    arg.1 = f32[2,1024] parameter(1), parameter_replication={false}
    dot.0 = f32[12288,1024] dot(arg.0, arg.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    reshape.0 = f32[32,384,1024] reshape(dot.0)
    constant.0 = f32[1,384,1024] constant(15)
    reshape.1 = f32[384,1024] reshape(constant.0)
    broadcast.0 = f32[32,384,1024] broadcast(reshape.1), dimensions={1,2}
    add.0 = f32[32,384,1024] add(reshape.0, broadcast.0)
    tuple.0 = (f32[32,384,1024]) tuple(add.0)
    ROOT get-tuple-element.0 = f32[32,384,1024] get-tuple-element(tuple.0), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, ApproxGELUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
    arg.0 = f32[32,32,4,16] parameter(0), parameter_replication={false}
    arg.1 = f32[32,32,16,32] parameter(1), parameter_replication={false}
    onednn.matmul.0 = f32[32,32,4,32] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    mul.0 = f32[32,32,4,32] multiply(onednn.matmul.0, onednn.matmul.0)
    mul.1 = f32[32,32,4,32] multiply(onednn.matmul.0, mul.0)
    const.0 = f32[] constant(0.044715)
    bcast.0 = f32[32,32,4,32] broadcast(const.0), dimensions={}
    mul.2 = f32[32,32,4,32] multiply(mul.1, bcast.0)
    add.0 = f32[32,32,4,32] add(onednn.matmul.0, mul.2)
    const.1 = f32[] constant(0.797884583)
    bcast.1 = f32[32,32,4,32] broadcast(const.1), dimensions={}
    mul.3 = f32[32,32,4,32] multiply(add.0, bcast.1)
    tanh = f32[32,32,4,32] tanh(mul.3)
    const.2 = f32[] constant(1)
    bcast.2 = f32[32,32,4,32] broadcast(const.2), dimensions={}
    add.2 = f32[32,32,4,32] add(tanh, bcast.2)
    const.3 = f32[] constant(0.5)
    bcast.3 = f32[32,32,4,32] broadcast(const.3), dimensions={}
    mul.4 = f32[32,32,4,32] multiply(add.2, bcast.3)
    ROOT out = f32[32,32,4,32] multiply(onednn.matmul.0, mul.4)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["GELU_TANH"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

// GPT-J Bias+GELU pattern with reduced sizes for test time:
// batch=32; seq_len=32; hidden_size=64; intermediate_size=256
TEST_F(MatmulTest, BiasAndApproxGELUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
  Arg_5.6 = f32[32,32,64] parameter(0), sharding={replicated}
  Arg_7.8 = f32[64,256] parameter(1), sharding={replicated}
  dot.232 = f32[32,32,256] dot(Arg_5.6, Arg_7.8), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  Arg_6.7 = f32[256] parameter(2), sharding={replicated}
  reshape.233 = f32[1,1,256] reshape(Arg_6.7)
  broadcast.234 = f32[1,1,256] broadcast(reshape.233), dimensions={0,1,2}
  reshape.235 = f32[256] reshape(broadcast.234)
  broadcast.236 = f32[32,32,256] broadcast(reshape.235), dimensions={2}
  add.237 = f32[32,32,256] add(dot.232, broadcast.236)
  multiply.238 = f32[32,32,256] multiply(add.237, add.237)
  multiply.239 = f32[32,32,256] multiply(add.237, multiply.238)
  constant.20 = f32[] constant(0.044715)
  broadcast.21 = f32[32,32,256] broadcast(constant.20), dimensions={}
  multiply.240 = f32[32,32,256] multiply(multiply.239, broadcast.21)
  add.241 = f32[32,32,256] add(add.237, multiply.240)
  constant.18 = f32[] constant(0.797884583)
  broadcast.19 = f32[32,32,256] broadcast(constant.18), dimensions={}
  multiply.242 = f32[32,32,256] multiply(add.241, broadcast.19)
  tanh.243 = f32[32,32,256] tanh(multiply.242)
  constant.16 = f32[] constant(1)
  broadcast.17 = f32[32,32,256] broadcast(constant.16), dimensions={}
  add.244 = f32[32,32,256] add(tanh.243, broadcast.17)
  constant.14 = f32[] constant(0.5)
  broadcast.15 = f32[32,32,256] broadcast(constant.14), dimensions={}
  multiply.245 = f32[32,32,256] multiply(add.244, broadcast.15)
  ROOT out = f32[32,32,256] multiply(add.237, multiply.245)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_gelu_tanh_);
}

// Tests GELU approximate pattern from tf.nn.gelu(approximate=True).
TEST_F(MatmulTest, BiasAndApproxTFGELUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
  arg0.1 = f32[1024,512] parameter(0), parameter_replication={false}
  arg1.2 = f32[256,512] parameter(1), parameter_replication={false}
  dot.7 = f32[1024,256] dot(arg0.1, arg1.2), lhs_contracting_dims={1}, rhs_contracting_dims={1}, frontend_attributes={grad_x="false",grad_y="false"}
  arg2.3 = f32[256] parameter(2), parameter_replication={false}
  broadcast.9 = f32[1024,256] broadcast(arg2.3), dimensions={1}
  add.10 = f32[1024,256] add(dot.7, broadcast.9)
  constant.12 = f32[] constant(0.044715)
  broadcast.13 = f32[1024,256] broadcast(constant.12), dimensions={}
  multiply.14 = f32[1024,256] multiply(broadcast.13, add.10)
  multiply.11 = f32[1024,256] multiply(add.10, add.10)
  multiply.15 = f32[1024,256] multiply(multiply.14, multiply.11)
  add.16 = f32[1024,256] add(add.10, multiply.15)
  constant.17 = f32[] constant(0.797884583)
  broadcast.18 = f32[1024,256] broadcast(constant.17), dimensions={}
  multiply.19 = f32[1024,256] multiply(add.16, broadcast.18)
  tanh.20 = f32[1024,256] tanh(multiply.19)
  constant.21 = f32[] constant(1)
  broadcast.22 = f32[1024,256] broadcast(constant.21), dimensions={}
  add.23 = f32[1024,256] add(tanh.20, broadcast.22)
  constant.24 = f32[] constant(0.5)
  broadcast.25 = f32[1024,256] broadcast(constant.24), dimensions={}
  multiply.26 = f32[1024,256] multiply(add.23, broadcast.25)
  ROOT multiply.27 = f32[1024,256] multiply(add.10, multiply.26)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_gelu_tanh_);
}

// Tests GELU approximate pattern from tf.nn.gelu(approximate=True) with
// auto_mixed_precision_onednn_bfloat16.
TEST_F(MatmulTest, BiasAndApproxTFGELUTestBF16) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
  arg0.1 = f32[1024,512] parameter(0), parameter_replication={false}
  convert.8 = bf16[1024,512] convert(arg0.1)
  arg1.2 = f32[256,512] parameter(1), parameter_replication={false}
  convert.9 = bf16[256,512] convert(arg1.2)
  dot.10 = bf16[1024,256] dot(convert.8, convert.9), lhs_contracting_dims={1}, rhs_contracting_dims={1}, frontend_attributes={grad_x="false",grad_y="false"}
  convert = f32[1024,256] convert(dot.10)
  arg2.3 = f32[256] parameter(2), parameter_replication={false}
  broadcast = f32[1024,256] broadcast(arg2.3), dimensions={1}
  add.13 = f32[1024,256] add(convert, broadcast)
  constant.16 = f32[] constant(0.044715)
  broadcast.17 = f32[1024,256] broadcast(constant.16), dimensions={}
  multiply.18 = f32[1024,256] multiply(broadcast.17, add.13)
  multiply.15 = f32[1024,256] multiply(add.13, add.13)
  multiply.19 = f32[1024,256] multiply(multiply.18, multiply.15)
  add.20 = f32[1024,256] add(add.13, multiply.19)
  constant.21 = f32[] constant(0.797884583)
  broadcast.22 = f32[1024,256] broadcast(constant.21), dimensions={}
  multiply.23 = f32[1024,256] multiply(add.20, broadcast.22)
  tanh.24 = f32[1024,256] tanh(multiply.23)
  constant.25 = f32[] constant(1)
  broadcast.26 = f32[1024,256] broadcast(constant.25), dimensions={}
  add.27 = f32[1024,256] add(tanh.24, broadcast.26)
  constant.1 = f32[] constant(0.5)
  broadcast.2 = f32[1024,256] broadcast(constant.1), dimensions={}
  multiply.30 = f32[1024,256] multiply(add.13, broadcast.2)
  ROOT multiply.32 = f32[1024,256] multiply(add.27, multiply.30)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_gelu_tanh_);
}

// Tests GELU approximate pattern from tf.nn.gelu(approximate=True)
TEST_F(MatmulTest, BiasAndApproxTFGELUTestF16) {
  if (!IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
  arg0.1 = f16[1024,512] parameter(0), parameter_replication={false}
  reshape.4 = f16[1024,512] reshape(arg0.1)
  arg1.2 = f16[256,512] parameter(1), parameter_replication={false}
  reshape.5 = f16[256,512] reshape(arg1.2)
  dot.7 = f16[1024,256] dot(reshape.4, reshape.5), lhs_contracting_dims={1}, rhs_contracting_dims={1}, frontend_attributes={grad_x="false",grad_y="false"}
  transpose.8 = f16[1024,256] transpose(dot.7), dimensions={0,1}
  arg2.3 = f16[256] parameter(2), parameter_replication={false}
  reshape.6 = f16[256] reshape(arg2.3)
  broadcast.9 = f16[1024,256] broadcast(reshape.6), dimensions={1}
  add.10 = f16[1024,256] add(transpose.8, broadcast.9)
  constant.12 = f16[] constant(0.044708)
  broadcast.13 = f16[1024,256] broadcast(constant.12), dimensions={}
  multiply.14 = f16[1024,256] multiply(broadcast.13, add.10)
  multiply.11 = f16[1024,256] multiply(add.10, add.10)
  multiply.15 = f16[1024,256] multiply(multiply.14, multiply.11)
  add.16 = f16[1024,256] add(add.10, multiply.15)
  constant.17 = f16[] constant(0.79785)
  broadcast.18 = f16[1024,256] broadcast(constant.17), dimensions={}
  multiply.19 = f16[1024,256] multiply(add.16, broadcast.18)
  tanh.20 = f16[1024,256] tanh(multiply.19)
  constant.21 = f16[] constant(1)
  broadcast.22 = f16[1024,256] broadcast(constant.21), dimensions={}
  add.23 = f16[1024,256] add(tanh.20, broadcast.22)
  constant.24 = f16[] constant(0.5)
  broadcast.25 = f16[1024,256] broadcast(constant.24), dimensions={}
  multiply.26 = f16[1024,256] multiply(add.23, broadcast.25)
  ROOT multiply.27 = f16[1024,256] multiply(add.10, multiply.26)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_gelu_tanh_);
}

TEST_F(MatmulTest, ReLUTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  relu.1 {
    Arg_0.3 = f32[32,32,4,32] parameter(0)
    constant.4 = f32[] constant(0)
    broadcast.5 = f32[32,32,4,32] broadcast(constant.4), dimensions={}
    ROOT maximum.6 = f32[32,32,4,32] maximum(Arg_0.3, broadcast.5)
  }

  ENTRY matmul.test.f32 {
    arg.0 = f32[32,32,4,16] parameter(0), parameter_replication={false}
    arg.1 = f32[32,32,16,32] parameter(1), parameter_replication={false}
    onednn.matmul.0 = f32[32,32,4,32] dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    ROOT call.7 = f32[32,32,4,32] call(onednn.matmul.0), to_apply=relu.1
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["RELU"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

TEST_F(MatmulTest, SimpleBiasTestBF16_PARAM_F32) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule jit_apply

  ENTRY matmul.test.bf16 {
    Arg_2.3 = f32[16,128,768] parameter(2), sharding={replicated}
    convert.4 = bf16[16,128,768] convert(Arg_2.3)
    Arg_1.2 = f32[768,3072] parameter(1), sharding={replicated}
    convert.5 = bf16[768,3072] convert(Arg_1.2)
    dot.7 = bf16[16,128,3072] dot(convert.4, convert.5), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    Arg_0.1 = f32[3072] parameter(0), sharding={replicated}
    convert.6 = bf16[3072] convert(Arg_0.1)
    reshape.8 = bf16[1,1,3072] reshape(convert.6)
    broadcast.9 = bf16[1,1,3072] broadcast(reshape.8), dimensions={0,1,2}
    reshape.10 = bf16[3072] reshape(broadcast.9)
    broadcast.11 = bf16[16,128,3072] broadcast(reshape.10), dimensions={2}
    ROOT add.12 = bf16[16,128,3072] add(dot.7, broadcast.11)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleBiasTestBF16_PARAM_BF16) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule jit_apply

  ENTRY matmul.test.bf16 {
    Arg_2.3 = f32[16,128,768] parameter(2), sharding={replicated}
    convert.4 = bf16[16,128,768] convert(Arg_2.3)
    Arg_1.2 = bf16[768,3072] parameter(1), sharding={replicated}
    dot.5 = bf16[16,128,3072] dot(convert.4, Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    Arg_0.1 = bf16[3072] parameter(0), sharding={replicated}
    reshape.6 = bf16[1,1,3072] reshape(Arg_0.1)
    broadcast.7 = bf16[1,1,3072] broadcast(reshape.6), dimensions={0,1,2}
    reshape.8 = bf16[3072] reshape(broadcast.7)
    broadcast.9 = bf16[16,128,3072] broadcast(reshape.8), dimensions={2}
    ROOT add.10 = bf16[16,128,3072] add(dot.5, broadcast.9)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, DivisionByConstantWithEltwiseLinearF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.divide.test.1

  ENTRY matmul.divide.test.f32 {
    Arg_4.5 = f32[16,128,768] parameter(0), sharding={replicated}
    Arg_2.3 = f32[768,12,64] parameter(1), sharding={replicated}
    onednn.matmul.0 = f32[16,128,12,64] dot(Arg_4.5, Arg_2.3), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    constant.8 = f32[] constant(8)
    broadcast.9 = f32[16,128,12,64] broadcast(constant.8), dimensions={}
    ROOT divide.16 = f32[16,128,12,64] divide(onednn.matmul.0, broadcast.9)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec(1e-4, 1e-4)));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":["LINEAR"]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

TEST_F(MatmulTest, SimpleBiasTestF16_PARAM_F32) {
  if (!IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }

  const char* matmul_module_str = R"(
  HloModule jit_apply

  ENTRY matmul.test.f16 {
    Arg_2.3 = f32[16,128,768] parameter(2), sharding={replicated}
    convert.4 = f16[16,128,768] convert(Arg_2.3)
    Arg_1.2 = f32[768,3072] parameter(1), sharding={replicated}
    convert.5 = f16[768,3072] convert(Arg_1.2)
    dot.7 = f16[16,128,3072] dot(convert.4, convert.5), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    Arg_0.1 = f32[3072] parameter(0), sharding={replicated}
    convert.6 = f16[3072] convert(Arg_0.1)
    reshape.8 = f16[1,1,3072] reshape(convert.6)
    broadcast.9 = f16[1,1,3072] broadcast(reshape.8), dimensions={0,1,2}
    reshape.10 = f16[3072] reshape(broadcast.9)
    broadcast.11 = f16[16,128,3072] broadcast(reshape.10), dimensions={2}
    ROOT add.12 = f16[16,128,3072] add(dot.7, broadcast.11)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleBiasTestF16_PARAM_F16) {
  if (!IsSupportedType(PrimitiveType::F16)) {
    GTEST_SKIP() << "CPU does not support F16.";
  }
  const char* matmul_module_str = R"(
  HloModule jit_apply

  ENTRY matmul.test.f16 {
    Arg_2.3 = f32[16,128,768] parameter(2), sharding={replicated}
    convert.4 = f16[16,128,768] convert(Arg_2.3)
    Arg_1.2 = f16[768,3072] parameter(1), sharding={replicated}
    dot.5 = f16[16,128,3072] dot(convert.4, Arg_1.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    Arg_0.1 = f16[3072] parameter(0), sharding={replicated}
    reshape.6 = f16[1,1,3072] reshape(Arg_0.1)
    broadcast.7 = f16[1,1,3072] broadcast(reshape.6), dimensions={0,1,2}
    reshape.8 = f16[3072] reshape(broadcast.7)
    broadcast.9 = f16[16,128,3072] broadcast(reshape.8), dimensions={2}
    ROOT add.10 = f16[16,128,3072] add(dot.5, broadcast.9)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-2, 1e-2}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, TestF32NonConstantWeights) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
    arg.0 = f32[64,256,16] parameter(0), parameter_replication={false}
    arg.1 = f32[16,32] parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[64,256,32] dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     %matmul.test.f32
  ; CHECK-NOT: custom_call_target="__onednn$matmul_reorder",
  ; CHECK:     custom-call(%{{[a-z,A-Z,0-9,\.]*}}, %arg.1), custom_call_target="__onednn$matmul",
  )");
}

TEST_F(MatmulTest, TestF32ConstantWeights) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
    arg.0 = f32[64,256,16] parameter(0), parameter_replication={false}
    constant = f32[] constant(1)
    arg.1 = f32[16,32] broadcast(constant), dimensions={}
    ROOT onednn.matmul.0 = f32[64,256,32] dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     %matmul.test.f32
  ; CHECK-NOT: custom_call_target="__onednn$matmul_reorder",
  ; CHECK:     custom-call(%{{[a-z,A-Z,0-9,\.]*}}, %constant{{[a-z,A-Z,0-9,\.]*}}), custom_call_target="__onednn$matmul",
  )");
}

TEST_F(MatmulTest, SimpleTestBF16Gemv1) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16

  ENTRY matmul.test.bf16 {
    arg.0 = bf16[1000,10000] parameter(0)
    arg.1 = bf16[10000] parameter(1)
    ROOT onednn.matmul.0 = bf16[1000] dot(arg.0, arg.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{2e-2, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, SimpleTestBF16Gemv2) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16
  
  ENTRY matmul.test.bf16 {
    arg.0 = bf16[100,300,300] parameter(0)
    arg.1 = bf16[300] parameter(1)
    ROOT onednn.matmul.0 = bf16[100,300] dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{2e-2, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, matmul_rewrite_str_);
}

TEST_F(MatmulTest, TestTransposeBNoRewriteF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32

  ENTRY matmul.test.f32 {
    arg.0 = f32[384,1024]{1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,1024]{1,0} parameter(1), parameter_replication={false}
    ROOT dot.2 = f32[384,2]{1,0} dot(arg.0, arg.1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     %matmul.test.f32
  ; CHECK-NOT: custom_call_target="__onednn$matmul",
  ; CHECK:     f32[384,2]{1,0} dot(%arg.0, %arg.1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  )");
}

TEST_F(MatmulTest, SimpleTestF32WithMulAndAddFusion) {
  const char* matmul_module_str = R"(
  ENTRY matmul.mul.add.test.f32 {
    arg0.1 = f32[32,32,40,30] parameter(0), parameter_replication={false}
    arg0.2 = f32[32,32,30,40] parameter(1), parameter_replication={false}
    dot.7 = f32[32,32,40,40] dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    const.0 = f32[] constant(0.044715)
    bcast.0 = f32[32,32,40,40] broadcast(const.0), dimensions={}
    mul.0 = f32[32,32,40,40] multiply(dot.7,bcast.0)
    const.1 = f32[] constant(0.65)
    bcast.1 = f32[32,32,40,40] broadcast(const.1), dimensions={}
    add.0 = f32[32,32,40,40] add(mul.0, bcast.1)
    const.2 = f32[] constant(0.65)
    bcast.2 = f32[32,32,40,40] broadcast(const.2), dimensions={}
    add.1 = f32[32,32,40,40] add(bcast.2, bcast.1)
    tuple.12 = (f32[32,32,40,40]) tuple(add.0)
    ROOT get-tuple-element.13 = f32[32,32,40,40] get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
    ; CHECK:     custom_call_target="__onednn$matmul",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:     "onednn_matmul_config":{
    ; CHECK-DAG:       "fused_ops":["LINEAR","BINARY_ADD"]
    ; CHECK-DAG:   }
    ; CHECK:     }
    )");
}

TEST_F(MatmulTest, WeightsPrepackAndScratch) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32
  ENTRY matmul.test.f32 {
    arg.0 = f32[64,256,16] parameter(0), parameter_replication={false}
    constant = f32[] constant(1)
    arg.1 = f32[16,32] broadcast(constant), dimensions={}
    ROOT onednn.matmul.0 = f32[64,256,32] dot(arg.0, arg.1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:        %matmul.test.f32
  ; CHECK:        custom_call_target="__onednn$matmul",
  ; CHECK-SAME:       backend_config={
  ; CHECK-SAME:           "outer_dimension_partitions":[],
  ; CHECK-SAME:           "onednn_matmul_config":{
  ; CHECK-SAME:               "weights_prepacked":true,"user_scratchpad":true
  ; CHECK-SAME:           }
  ; CHECK-SAME:       }
  )");
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3

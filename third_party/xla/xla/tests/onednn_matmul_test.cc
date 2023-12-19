/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/cpu_info.h"

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
};

TEST_F(MatmulTest, SimpleTestF32) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.f32, entry_computation_layout={(f32[2,8,4,16]{3,2,1,0},f32[2,8,16,32]{3,2,1,0})->f32[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.f32 {
    arg.0 = f32[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestBF16) {
  // TODO(penporn): Refactor IsBF16SupportedByOneDNNOnThisCPU() from
  // tensorflow/core/graph/mkl_graph_util.h and call the function instead.
  using tsl::port::TestCPUFeature;
  if (!TestCPUFeature(tsl::port::CPUFeature::AVX512_BF16) &&
      !TestCPUFeature(tsl::port::CPUFeature::AMX_BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const char* matmul_module_str = R"(
  HloModule matmul.test.bf16, entry_computation_layout={(bf16[2,8,4,16]{3,2,1,0},bf16[2,8,16,32]{3,2,1,0})->bf16[2,8,4,32]{3,2,1,0}}

  ENTRY matmul.test.bf16 {
    arg.0 = bf16[2,8,4,16]{3,2,1,0} parameter(0), parameter_replication={false}
    arg.1 = bf16[2,8,16,32]{3,2,1,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = bf16[2,8,4,32]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str,
                    R"(
  ; CHECK:     custom_call_target="__onednn$matmul",
  ; CHECK:       backend_config={
  ; CHECK-DAG:     "outer_dimension_partitions":[],
  ; CHECK-DAG:     "onednn_matmul_config":{
  ; CHECK-DAG:       "fused_ops":[]
  ; CHECK-DAG:   }
  ; CHECK:     }
  )");
}

TEST_F(MatmulTest, SimpleTestF32TransposeB) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1, entry_computation_layout={(f32[2,8,4,16]{3,1,2,0},f32[2,8,4,16]{3,1,2,0})->f32[2,8,4,4]{3,2,1,0}}

  ENTRY matmul.test.1 {
    arg.0 = f32[2,8,4,16]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[2,8,4,16]{3,1,2,0} parameter(1), parameter_replication={false}
    ROOT onednn.matmul.0 = f32[2,8,4,4]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAddFusion1) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[32,32,40,30]{3,2,1,0})->f32[32,32,40,40]{3,2,1,0}}
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[32,32,40,30]{3,2,1,0} parameter(0), parameter_replication={false}
    reshape.2 = f32[32,32,40,30]{3,2,1,0} reshape(arg0.1)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[32,32,30,40]{3,2,1,0} broadcast(constant.3), dimensions={}
    dot.7 = f32[32,32,40,40]{3,2,1,0} dot(reshape.2, broadcast.4), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[40]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[32,32,40,40]{3,2,1,0} broadcast(broadcast.6), dimensions={3}
    add.10 = f32[32,32,40,40]{3,2,1,0} add(dot.7, broadcast.9)
    reshape.11 = f32[32,32,40,40]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[32,32,40,40]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,32,40,40]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAddFusion2) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[400,300]{1,0})->f32[400,1,400]{2,1,0}}
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[400,300]{1,0} parameter(0), parameter_replication={false}
    reshape.2 = f32[400,300]{1,0} reshape(arg0.1)
    constant.3 = f32[] constant(1)
    broadcast.4 = f32[300,400]{1,0} broadcast(constant.3), dimensions={}
    dot.7 = f32[400,400]{1,0} dot(reshape.2, broadcast.4), lhs_batch_dims={}, lhs_contracting_dims={1}, rhs_batch_dims={}, rhs_contracting_dims={0}
    reshape.1 = f32[400,1,400]{2,1,0} reshape(dot.7)
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[400]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[400,1,400]{2,1,0} broadcast(broadcast.6), dimensions={2}
    add.10 = f32[400,1,400]{2,1,0} add(reshape.1, broadcast.9)
    tuple.12 = (f32[400,1,400]{2,1,0}) tuple(add.10)
    ROOT get-tuple-element.13 = f32[400,1,400]{2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter1) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[32,32,40,30]{3,2,1,0}, f32[32,32,30,40]{3,2,1,0}, f32[32,32,40,40]{3,2,1,0})->f32[32,32,40,40]{3,2,1,0}}
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[32,32,40,30]{3,2,1,0} parameter(0), parameter_replication={false}
    arg0.2 = f32[32,32,30,40]{3,2,1,0} parameter(1), parameter_replication={false}
    arg0.3 = f32[32,32,40,40]{3,2,1,0} parameter(2), parameter_replication={false}
    dot.7 = f32[32,32,40,40]{3,2,1,0} dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    add.10 = f32[32,32,40,40]{3,2,1,0} add(dot.7, arg0.3)
    reshape.11 = f32[32,32,40,40]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[32,32,40,40]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,32,40,40]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter2) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[32,32,40,30]{3,2,1,0}, f32[32,32,30,40]{3,2,1,0}, f32[40]{0})->f32[32,32,40,40]{3,2,1,0}}
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[32,32,40,30]{3,2,1,0} parameter(0), parameter_replication={false}
    arg0.2 = f32[32,32,30,40]{3,2,1,0} parameter(1), parameter_replication={false}
    arg0.3 = f32[40]{0} parameter(2), parameter_replication={false}
    dot.7 = f32[32,32,40,40]{3,2,1,0} dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = f32[32,32,40,40]{3,2,1,0} broadcast(arg0.3), dimensions={3}
    add.10 = f32[32,32,40,40]{3,2,1,0} add(dot.7, broad.1)
    reshape.11 = f32[32,32,40,40]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[32,32,40,40]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,32,40,40]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter2D) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[2,2,400,30]{3,2,1,0}, f32[2,2,30,400]{3,2,1,0}, f32[2,400]{1,0})->f32[2,2,400,400]{3,2,1,0}}
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[2,2,400,30]{3,2,1,0} parameter(0), parameter_replication={false}
    arg0.2 = f32[2,2,30,400]{3,2,1,0} parameter(1), parameter_replication={false}
    arg0.3 = f32[2,400]{1,0} parameter(2), parameter_replication={false}
    dot.7 = f32[2,2,400,400]{3,2,1,0} dot(arg0.1, arg0.2), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
    broad.1 = f32[2,2,400,400]{3,2,1,0} broadcast(arg0.3), dimensions={0,3}
    add.10 = f32[2,2,400,400]{3,2,1,0} add(dot.7, broad.1)
    reshape.11 = f32[2,2,400,400]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[2,2,400,400]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[2,2,400,400]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32WithBiasAsParameter3) {
  const char* matmul_module_str = R"(
  HloModule matmul.biasadd.test.f32, entry_computation_layout={(f32[16,128,768]{2,1,0}, f32[768,768]{1,0}, f32[768]{0})->f32[16,128,768]{2,1,0}}
  
  ENTRY matmul.biasadd.test.f32 {
    arg0.1 = f32[16,128,768]{2,1,0} parameter(0), sharding={replicated}
    arg0.2 = f32[768,768]{1,0} parameter(1), sharding={replicated}
    dot.84 = f32[16,128,768]{2,1,0} dot(arg0.1, arg0.2), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    arg0.3 = f32[768]{0} parameter(2), sharding={replicated}
    reshape.85 = f32[1,1,768]{2,1,0} reshape(arg0.3)
    broadcast.86 = f32[1,1,768]{2,1,0} broadcast(reshape.85), dimensions={0,1,2}
    reshape.87 = f32[768]{0} reshape(broadcast.86)
    broadcast.88 = f32[16,128,768]{2,1,0} broadcast(reshape.87), dimensions={2}
    ROOT add.89 = f32[16,128,768]{2,1,0} add(dot.84, broadcast.88)
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

TEST_F(MatmulTest, SimpleTestF32TransposeBWithBiasAddFusion) {
  const char* matmul_module_str = R"(
  HloModule matmul.test.1, entry_computation_layout={(f32[32,8,4,16]{3,1,2,0},f32[32,8,4,16]{3,1,2,0})->f32[32,8,4,4]{3,2,1,0}}
  
  ENTRY matmul.test.1 {
    arg.0 = f32[32,8,4,16]{3,1,2,0} parameter(0), parameter_replication={false}
    arg.1 = f32[32,8,4,16]{3,1,2,0} parameter(1), parameter_replication={false}
    dot.7 = f32[32,8,4,4]{3,2,1,0} dot(arg.0, arg.1), lhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={3}
    constant.5 = f32[] constant(15)
    broadcast.6 = f32[4]{0} broadcast(constant.5), dimensions={}
    broadcast.9 = f32[32,8,4,4]{3,2,1,0} broadcast(broadcast.6), dimensions={3}
    add.10 = f32[32,8,4,4]{3,2,1,0} add(dot.7, broadcast.9)
    reshape.11 = f32[32,8,4,4]{3,2,1,0} reshape(add.10)
    tuple.12 = (f32[32,8,4,4]{3,2,1,0}) tuple(reshape.11)
    ROOT get-tuple-element.13 = f32[32,8,4,4]{3,2,1,0} get-tuple-element(tuple.12), index=0
  })";

  EXPECT_TRUE(RunAndCompare(matmul_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(matmul_module_str, fused_matmul_bias_);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3

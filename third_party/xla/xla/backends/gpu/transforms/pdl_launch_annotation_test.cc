/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/pdl_launch_annotation.h"

#include <optional>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace gpu {
namespace {

using PdlLaunchAnnotationTest = HloHardwareIndependentTestBase;

TEST_F(PdlLaunchAnnotationTest, DependentOperandGetsMarkedNonInvariant) {
  RunAndFilecheckHloRewrite(R"(
HloModule m, is_scheduled=true

triton_gemm {
  lhs = f16[16,32] parameter(0)
  rhs = f16[32,16] parameter(1)
  dot = f16[16,16] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

add_comp {
  a = f16[16,16] parameter(0)
  b = f16[16,16] parameter(1)
  ROOT out = f16[16,16] add(a, b)
}

e {
  lhs = f16[16,32] parameter(0)
  rhs = f16[32,16] parameter(1)
  y = f16[16,16] parameter(2)
  // CHECK: fusion({{.*}}), kind=kCustom
  // CHECK-SAME: frontend_attributes={xla.pdl_launch="true"}
  gemm = f16[16,16] fusion(lhs, rhs), kind=kCustom,
    calls=triton_gemm, backend_config={
      fusion_backend_config:{
        kind:"__triton_nested_gemm_fusion",
        block_level_fusion_config:{
          output_tiles:[{sizes:[64,32]}], num_warps:8, num_ctas:1}}}
  t = (f16[16,16]) tuple(gemm)
  g = f16[16,16] get-tuple-element(t), index=0
  bc = f16[16,16] bitcast(g)
  // CHECK: fusion({{.*}}), kind=kLoop
  // CHECK-SAME: frontend_attributes={xla.no_invariant_operands="1"}
  add = f16[16,16] fusion(y, bc), kind=kLoop, calls=add_comp
})",
                            PdlLaunchAnnotationPass());
}

TEST_F(PdlLaunchAnnotationTest, NoLaunchWithoutSuccessorFusion) {
  RunAndFilecheckHloRewrite(R"(
HloModule m, is_scheduled=true

triton_gemm {
  lhs = f16[16,32] parameter(0)
  rhs = f16[32,16] parameter(1)
  dot = f16[16,16] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

e {
  lhs = f16[16,32] parameter(0)
  rhs = f16[32,16] parameter(1)
  gemm = f16[16,16] fusion(lhs, rhs), kind=kCustom,
    calls=triton_gemm, backend_config={
      fusion_backend_config:{
        kind:"__triton_nested_gemm_fusion",
        block_level_fusion_config:{
          output_tiles:[{sizes:[64,32]}], num_warps:8, num_ctas:1}}}
  ROOT bc = f16[16,16] bitcast(gemm)
})",
                            PdlLaunchAnnotationPass(), std::nullopt);
}

TEST_F(PdlLaunchAnnotationTest, AnnotationIsNotDuplicated) {
  RunAndFilecheckHloRewrite(R"(
HloModule m, is_scheduled=true

triton_gemm {
  lhs = f16[16,32] parameter(0)
  rhs = f16[32,16] parameter(1)
  dot = f16[16,16] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

n {
  a = f16[16,16] parameter(0)
  b = f16[16,16] negate(a)
}

e {
  lhs = f16[16,32] parameter(0)
  rhs = f16[32,16] parameter(1)
  gemm = f16[16,16] fusion(lhs, rhs), kind=kCustom,
    calls=triton_gemm, backend_config={
      fusion_backend_config:{
        kind:"__triton_nested_gemm_fusion",
        block_level_fusion_config:{
          output_tiles:[{sizes:[64,32]}], num_warps:8, num_ctas:1}}},
    frontend_attributes={xla.pdl_launch="true"}
  f = f16[16,16] fusion(gemm), kind=kLoop, calls=n,
    frontend_attributes={xla.no_invariant_operands="0"}
})",
                            PdlLaunchAnnotationPass(), std::nullopt);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

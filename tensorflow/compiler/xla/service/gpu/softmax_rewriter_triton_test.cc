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
#include "tensorflow/compiler/xla/service/gpu/softmax_rewriter_triton.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using SoftmaxRewriterTritonTest = HloTestBase;

TEST_F(SoftmaxRewriterTritonTest, CanFuseExactSoftmaxF32) {
  const std::string& hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f32[127,125]{1,0} exponential(subtract)
  constant_zero = f32[] constant(0)
  second_reduce = f32[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f32[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f32[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  GpuVersion gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  SoftmaxRewriterTriton fusion_rewriter(gpu_version);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseExactSoftmaxF16) {
  const std::string& hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f16[] parameter(0)
  arg_1 = f16[] parameter(1)
  ROOT maximum = f16[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f16[] parameter(0)
  arg_1.1 = f16[] parameter(1)
  ROOT add = f16[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f16[127,125]{1,0} parameter(0)
  constant_neg_inf = f16[] constant(-inf)
  reduce = f16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f16[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f16[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f16[127,125]{1,0} exponential(subtract)
  constant_zero = f16[] constant(0)
  second_reduce = f16[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f16[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f16[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  GpuVersion gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  SoftmaxRewriterTriton fusion_rewriter(gpu_version);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseExactSoftmaxF64) {
  const std::string& hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f64[] parameter(0)
  arg_1 = f64[] parameter(1)
  ROOT maximum = f64[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = f64[] parameter(0)
  arg_1.1 = f64[] parameter(1)
  ROOT add = f64[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f64[127,125]{1,0} parameter(0)
  constant_neg_inf = f64[] constant(-inf)
  reduce = f64[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f64[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = f64[127,125]{1,0} subtract(param_0, broadcast)
  exponential = f64[127,125]{1,0} exponential(subtract)
  constant_zero = f64[] constant(0)
  second_reduce = f64[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = f64[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = f64[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  GpuVersion gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  SoftmaxRewriterTriton fusion_rewriter(gpu_version);
  EXPECT_TRUE(fusion_rewriter.Run(module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanNotFuseExactSoftmaxBF16) {
  const std::string& hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = bf16[] parameter(0)
  arg_1 = bf16[] parameter(1)
  ROOT maximum = bf16[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = bf16[] parameter(0)
  arg_1.1 = bf16[] parameter(1)
  ROOT add = bf16[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  constant_neg_inf = bf16[] constant(-inf)
  reduce = bf16[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = bf16[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = bf16[127,125]{1,0} subtract(param_0, broadcast)
  exponential = bf16[127,125]{1,0} exponential(subtract)
  constant_zero = bf16[] constant(0)
  second_reduce = bf16[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = bf16[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = bf16[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  GpuVersion gpu_version{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  SoftmaxRewriterTriton fusion_rewriter(gpu_version);
  EXPECT_FALSE(fusion_rewriter.Run(module.get()).value());
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla

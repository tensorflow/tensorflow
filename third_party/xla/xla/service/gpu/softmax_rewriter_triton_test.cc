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
#include "xla/service/gpu/softmax_rewriter_triton.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using ::testing::HasSubstr;

// Wrapper around SoftmaxRewriterTriton(gpu_version).Run(module) that finds
// and fuses as many diamond chains as possible without invoking any kind of
// cost analysis.
absl::StatusOr<bool> SoftmaxRewriterTritonMatchAndRewrite(
    se::GpuComputeCapability gpu_version, HloModule* module) {
  CHECK_NE(module, nullptr);
  SoftmaxRewriterTriton softmax_rewriter_triton(gpu_version);
  TF_ASSIGN_OR_RETURN(std::vector<DiamondChainDescriptor> diamond_chains,
                      softmax_rewriter_triton.FindAllFusibleDiamondChains(
                          *module, /*execution_threads=*/{}));

  for (auto diamond_chain = diamond_chains.rbegin();
       diamond_chain != diamond_chains.rend(); ++diamond_chain) {
    TF_RETURN_IF_ERROR(
        softmax_rewriter_triton.FuseDiamondChain(*diamond_chain));
  }

  return !diamond_chains.empty();
}

class SoftmaxRewriterTritonTest
    : public HloTestBase,
      public ::testing::WithParamInterface<PrimitiveType> {
 public:
  void SetUp() override {
    gpu_version_ = se::GpuComputeCapability{
        se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};
  }

 protected:
  se::GpuComputeCapability gpu_version_;
};

TEST_P(SoftmaxRewriterTritonTest, CanFuseExactSoftmax) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Fusion(m::Parameter())));
      break;
    case F16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Divide(m::Exp(), m::Broadcast())));
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(SoftmaxRewriterTritonTest, CanFuseFirstSoftmaxDiamond) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, CanNotFuseExactSoftmaxF64) {
  const std::string hlo_string = R"(
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
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, CanFuseExactSoftmaxBF16) {
  const std::string hlo_string = R"(
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
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxWhenResultingComputationCanNotBeTiledCorrectly) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[130,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  bitcasted_param_0 = $0[65,2,125] bitcast(param_0)
  reduce = $0[65,2]{1,0} reduce(bitcasted_param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  bitcasted_reduce = $0[130] bitcast(reduce)
  broadcast = $0[130,125]{1,0} broadcast(bitcasted_reduce), dimensions={0}
  bitcasted_broadcast = $0[65,2,125] bitcast(broadcast)
  subtract = $0[65,2,125]{2,1,0} subtract(bitcasted_param_0, bitcasted_broadcast)
  bitcasted_subtract = $0[130,125] bitcast(subtract)
  exponential = $0[130,125]{1,0} exponential(bitcasted_subtract)
  constant_zero = $0[] constant(0)
  bitcasted_exponential = $0[2,65,125] bitcast(exponential)
  second_reduce = $0[2,65]{1,0} reduce(bitcasted_exponential, constant_zero), dimensions={2}, to_apply=add_computation
  second_bitcasted_reduce = $0[130] bitcast(second_reduce)
  second_broadcast = $0[130,125]{1,0} broadcast(second_bitcasted_reduce), dimensions={0}
  second_bitcasted_broadcast = $0[2,65,125] bitcast(second_broadcast)
  divide = $0[2,65,125]{2,1,0} divide(bitcasted_exponential, second_bitcasted_broadcast)
  ROOT bitcasted_divide = $0[130,125] bitcast(divide)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest, CanNotFuseSoftmaxDiamondWithWrongLayout) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{0,1} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithWrongReduceDimension) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[125]{0} reduce(param_0, constant_neg_inf), dimensions={0}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={1}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithWrongBroadcastDimension) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[125,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[125]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[125,125]{1,0} broadcast(reduce), dimensions={1}
  ROOT subtract = $0[125,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

// TODO(bchetioui): expand so this can be supported?
TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithExtraBroadcastUsage) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT multiply = $0[127,125]{1,0} multiply(broadcast, subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxWithIntermediateUnaryElementwise) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  abs = $0[127,125]{1,0} abs(subtract)
  exponential = $0[127,125]{1,0} exponential(abs)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Fusion(m::Parameter())));
      break;
    case F16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Divide()));
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseTwoDiamondsWithSecondDiamondProducerEqualToFirstDiamondRoot) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(subtract, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT divide = $0[127,125]{1,0} divide(subtract, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Fusion(m::Parameter())));
      break;
    case F16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Divide()));
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseDiamondWithTrailingUnaryElementwiseAtTheRoot) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT abs = $0[127,125]{1,0} abs(subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest, CanFuseDiamondWithUnaryElementwisePrefix) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  abs = $0[127,125]{1,0} abs(param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(abs, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseDiamondWithMultipleBroadcastDimensions) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[1,3,125,125]{3,2,1,0} parameter(0)
  bitcast = $0[3,125,125]{2,1,0} bitcast($0[1,3,125,125]{3,2,1,0} param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[3,125]{1,0} reduce($0[3,125,125]{2,1,0} bitcast, $0[] constant_neg_inf), dimensions={2}, to_apply=max_computation
  broadcast = $0[1,3,125,125]{3,2,1,0} broadcast($0[3,125]{1,0} reduce), dimensions={1,2}
  ROOT subtract = $0[1,3,125,125]{3,2,1,0} subtract($0[1,3,125,125]{3,2,1,0} param_0, $0[1,3,125,125]{3,2,1,0} broadcast)
})";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithNonConstantReducerIdentity) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  identity = $0[] parameter(1)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, identity), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTritonIncompatibleRoot) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  divide = $0[127,125]{1,0} divide(param_0, broadcast)
  ROOT remainder = $0[127,125]{1,0} remainder(divide, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTritonIncompatibleReducer) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  if_0 = pred[] is-finite(arg_0)
  c = $0[] convert(if_0)
  ROOT maximum = $0[] maximum(c, arg_1)
}

ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxDiamondWithLastDimensionBitcastAfterReduce) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[3,127,125]{2,1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[3,127]{1,0} reduce(param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  bitcasted_reduce = $0[381]{0} bitcast(reduce)
  broadcast = $0[381,125]{1,0} broadcast(bitcasted_reduce), dimensions={0}
  bitcasted_broadcast = $0[3,127,125]{2,1,0} bitcast(broadcast)
  ROOT subtract = $0[3,127,125]{2,1,0} subtract(param_0, bitcasted_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithTransposeBitcast) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[1,127,125]{2,1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  bitcasted_param_0 = $0[127,1,125]{2,0,1} bitcast(param_0)
  reduce = $0[127,1]{0,1} reduce(bitcasted_param_0, constant_neg_inf), dimensions={2}, to_apply=max_computation
  broadcast = $0[127,1,125]{2,0,1} broadcast(reduce), dimensions={0,1}
  bitcasted_broadcast = $0[1,127,125]{2,1,0} bitcast(broadcast)
  ROOT subtract = $0[1,127,125]{2,1,0} subtract(param_0, bitcasted_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseTwoDiamondsWithDifferentReductionAxisSizeTogether) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,625]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,625]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,625]{1,0} subtract(param_0, broadcast)
  bitcasted_subtract = $0[127,5,125] bitcast(subtract)
  exponential = $0[127,5,125] exponential(bitcasted_subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127,5] reduce(exponential, constant_zero), dimensions={2}, to_apply=add_computation
  second_broadcast = $0[127,5,125] broadcast(second_reduce), dimensions={0,1}
  ROOT divide = $0[127,5,125] divide(exponential, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Fusion(m::Bitcast(m::Fusion(m::Parameter())))));
      break;
    case F16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Divide(m::Exp(), m::Broadcast())));
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseTwoDiamondsWithExtraUsageForFirstDiamondRoot) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
  ROOT tuple = ($0[127,125]{1,0}, $0[127,125]{1,0}) tuple(divide, subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Tuple(m::Fusion(m::Fusion()),
                                      m::Fusion(m::Parameter()))));
      break;
    case F16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Tuple(m::Divide(), m::Fusion(m::Parameter()))));
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseTwoDiamondsWithExtraUsageForSecondDiamondProducer) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  exponential = $0[127,125]{1,0} exponential(subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(exponential, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  divide = $0[127,125]{1,0} divide(exponential, second_broadcast)
  ROOT tuple = ($0[127,125]{1,0}, $0[127,125]{1,0}) tuple(divide, exponential)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Tuple(m::Fusion(m::Fusion()),
                                      m::Fusion(m::Parameter()))));
      break;
    case F16:
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Tuple(m::Divide(), m::Exp())));
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxDiamondWithTritonIncompatibleProducer) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  if_0 = pred[127,125] is-finite(param_0)
  c = $0[127,125] convert(if_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(c, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(c, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::IsFinite(m::Parameter()))));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanNotFuseSoftmaxDiamondWithNonFusibleBitcastBetweenReduceAndProducer) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax

max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[1,127,5,25]{3,2,1,0} parameter(0)
  bitcast_0 = $0[127,125] bitcast(param_0)
  bitcast_1 = $0[127,125] bitcast(param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(bitcast_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(bitcast_1, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseSoftmaxDiamondWithBitcastProducerFollowedByBitcastsOnEachUse) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax

max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[1,1,127,125]{3,2,1,0} parameter(0)
  bitcast_parent = $0[127,125]{1,0} bitcast(param_0)
  bitcast_0 = $0[127,125]{1,0} bitcast(bitcast_parent)
  bitcast_1 = $0[127,125]{1,0} bitcast(bitcast_parent)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(bitcast_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(bitcast_1, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(
    SoftmaxRewriterTritonTest,
    CanNotFuseSoftmaxDiamondWithBitcastProducerFollowedByThreeBitcastsOnTheLeftIncludingTwoNonFusibleOnes) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax

max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}

ENTRY main {
  param_0 = $0[1,1,127,125]{3,2,1,0} parameter(0)
  bitcast_parent = $0[127,125] bitcast(param_0)
  bitcast_0 = $0[127,5,25] bitcast(bitcast_parent)
  bitcast_1 = $0[1,127,125] bitcast(bitcast_0)
  bitcast_2 = $0[127,125] bitcast(bitcast_1)
  bitcast_3 = $0[127,125] bitcast(bitcast_parent)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(bitcast_3, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(bitcast_2, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest, CanFuseSoftmaxDiamondWithSmallRows) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,7]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,7]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,7]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_THAT(SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()),
              tsl::testing::IsOkAndHolds(true));
  TF_EXPECT_OK(verifier().Run(module.get()).status());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseConvertInvolvingBF16InputIntoSoftmaxDiamond) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  param_0_$0 = $0[127,125]{1,0} convert(param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0_$0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0_$0, broadcast)
})";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(
          se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0},
          module.get())
          .value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(SoftmaxRewriterTritonTest, RewriterBailsOutOnPreAmpereGpu) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  param_0_f32 = f32[127,125]{1,0} convert(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0_f32, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0_f32, broadcast)
})";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_THAT(
      SoftmaxRewriterTriton(
          se::CudaComputeCapability{se::CudaComputeCapability::VOLTA, 0})
          .Run(module.get()),
      tsl::testing::StatusIs(
          tsl::error::FAILED_PRECONDITION,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

TEST_F(SoftmaxRewriterTritonTest, RewriterBailsOutOnNonCudaGpu) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = bf16[127,125]{1,0} parameter(0)
  param_0_f32 = f32[127,125]{1,0} convert(param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(param_0_f32, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0_f32, broadcast)
})";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  EXPECT_THAT(
      SoftmaxRewriterTriton(se::RocmComputeCapability{}).Run(module.get()),
      tsl::testing::StatusIs(
          tsl::error::FAILED_PRECONDITION,
          ::testing::StrEq("Triton support is only enabled for CUDA GPUs.")));
}

TEST_P(SoftmaxRewriterTritonTest, DoesNotFuseConvertWithC64DataType) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT convert = c64[127,125]{1,0} convert(subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(m::Fusion(m::Parameter()))));
}

TEST_P(SoftmaxRewriterTritonTest, DoesNotFuseConvertWithC128DataType) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT convert = c128[127,125]{1,0} convert(subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Convert(m::Fusion(m::Parameter()))));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseBinaryElementwiseProducerIntoDiamondWhenBothOperandsAreTheSame) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  multiply =  $0[127,125]{1,0} multiply(param_0, param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(multiply, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(
    SoftmaxRewriterTritonTest,
    CanFuseIntermediateBinaryElementwiseWithinDiamondWhenBothOperandsAreTheSame) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  multiply =  $0[127]{0} multiply(reduce, reduce)
  broadcast = $0[127,125]{1,0} broadcast(multiply), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseBinaryElementwiseWhenBothOperandsAreTheSameBetweenDiamonds) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamonds
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  multiply = $0[127,125]{1,0} multiply(subtract, subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT subtract_second = $0[127,125]{1,0} subtract(multiply, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseBinaryElementwiseConsumerWhereBothOperandsAreTheSameIntoDiamond) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  ROOT multiply = $0[127,125]{1,0} multiply(subtract, subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(
    SoftmaxRewriterTritonTest,
    DoesNotFuseIntermediateBinaryElementwiseWhereBothOperandsAreTheSameIntoDiamondWithoutTritonSupport) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule softmax
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  remainder = $0[127,125]{1,0} remainder(param_0, param_0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(remainder, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseTwoBinaryElementwiseWhereBothOperandsAreTheSameBetweenDiamonds) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamonds
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  add = $0[127,125]{1,0} add(subtract, subtract)
  multiply = $0[127,125]{1,0} multiply(add, add)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT subtract_second = $0[127,125]{1,0} subtract(multiply, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseIntermediateBinaryElementwiseWithBothSplatOperandsIntoDiamond) {
  const std::string hlo_string = R"(
HloModule nonfusible_splat
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  constant_0 = f32[] constant(0.333333343)
  splat_0 = f32[127,125]{1,0} broadcast(constant_0), dimensions={}
  constant_1 = f32[] constant(0.66666)
  splat_1 = f32[127,125]{1,0} broadcast(constant_1), dimensions={}
  param_0 = f32[127,125]{1,0} parameter(0)
  multiply_splats = f32[127,125]{1,0} multiply(splat_0, splat_1)
  multiply_splat_param = f32[127,125]{1,0} multiply(multiply_splats, param_0)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(multiply_splat_param, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseIntermediateBinaryElementwiseWithSameSplatOperandsIntoDiamond) {
  const std::string hlo_string = R"(
HloModule nonfusible_splat_diamond
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  constant_0 = f32[] constant(0.333333343)
  splat = f32[127,125]{1,0} broadcast(constant_0), dimensions={}
  param_0 = f32[127,125]{1,0} parameter(0)
  multiply = f32[127,125]{1,0} multiply(splat, splat)
  add = f32[127,125]{1,0} add(param_0, multiply)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(add, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton fusion_rewriter(gpu_version_);
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_P(SoftmaxRewriterTritonTest, CanFuseRMSNormDiamond) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule rms_norm
add_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT add.1 = $0[] add(arg_0, arg_1)
}
ENTRY main.30 {
  param_0 = $0[10,10,10,128]{3,2,1,0} parameter(0)
  multiply_param = $0[10,10,10,128]{3,2,1,0} multiply(param_0, param_0)
  constant_0 = $0[] constant(0)
  reduce = $0[10,10,10]{2,1,0} reduce(multiply_param, constant_0), dimensions={3}, to_apply=add_computation
  constant_1 = $0[] constant(0.333333343)
  splat = $0[10,10,10]{2,1,0} broadcast(constant_1), dimensions={}
  multiply_splat = $0[10,10,10]{2,1,0} multiply(reduce, splat)
  epsilon = $0[] constant(1e-06)
  splat_epsilon = $0[10,10,10]{2,1,0} broadcast(epsilon), dimensions={}
  add = $0[10,10,10]{2,1,0} add(multiply_splat, splat_epsilon)
  rsqrt = $0[10,10,10]{2,1,0} rsqrt(add)
  broadcast = $0[10,10,10,128]{3,2,1,0} broadcast(rsqrt), dimensions={0,1,2}
  ROOT multiply = $0[10,10,10,128]{3,2,1,0} multiply(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();

  switch (data_type) {
    case F32:
    case BF16:
      EXPECT_TRUE(
          SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get())
              .value());
      EXPECT_TRUE(verifier().Run(module.get()).status().ok());
      EXPECT_THAT(module->entry_computation()->root_instruction(),
                  GmockMatch(m::Fusion(m::Parameter())));
      break;
    case F16:
      // Triton does not support F16 rsqrt.
      EXPECT_FALSE(
          SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get())
              .value());
      break;
    default:
      ABSL_UNREACHABLE();
  }
}

TEST_P(
    SoftmaxRewriterTritonTest,
    CanFuseAndEmitBinaryElementwiseWhereTheFirstOperandIsASplatConstantBetweenDiamonds) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamonds
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  multiply = $0[127,125]{1,0} multiply(broadcast_splat, subtract)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT second_subtract = $0[127,125]{1,0} subtract(multiply, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(
    SoftmaxRewriterTritonTest,
    CanFuseAndEmitBinaryElementwiseWhereTheSecondOperandIsASplatConstantBetweenDiamonds) {  // NOLINT(whitespace/line_length)
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamonds
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  multiply = $0[127,125]{1,0} multiply(subtract, broadcast_splat)
  constant_zero = $0[] constant(0)
  second_reduce = $0[127]{0} reduce(multiply, constant_zero), dimensions={1}, to_apply=add_computation
  second_broadcast = $0[127,125]{1,0} broadcast(second_reduce), dimensions={0}
  ROOT second_subtract = $0[127,125]{1,0} subtract(multiply, second_broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(
    SoftmaxRewriterTritonTest,
    CanFuseBinaryElementwiseWhereTheFirstOperandIsASplatConstantWithinDiamond) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT maximum = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=max_computation
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127]{0} broadcast(constant), dimensions={}
  multiply = $0[127]{0} multiply(broadcast_splat, reduce)
  broadcast = $0[127,125]{1,0} broadcast(multiply), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseBinaryElementwiseConsumerWhereTheFirstOperandIsASplatConstant) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule fusible_diamond
add_computation {
  arg_0.1 = $0[] parameter(0)
  arg_1.1 = $0[] parameter(1)
  ROOT add = $0[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(param_0, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
  constant = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant), dimensions={}
  ROOT multiply = $0[127,125]{1,0} multiply(broadcast_splat, subtract)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_P(SoftmaxRewriterTritonTest,
       CanFuseBinaryElementwiseOperationWhereOneOperandIsASharedSplatProducer) {
  PrimitiveType data_type = GetParam();
  const std::string hlo_string_template = R"(
HloModule nonfusible_diamond
max_computation {
  arg_0 = $0[] parameter(0)
  arg_1 = $0[] parameter(1)
  ROOT max = $0[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = $0[127,125]{1,0} parameter(0)
  constant_2 = $0[] constant(0.333333343)
  broadcast_splat = $0[127,125]{1,0} broadcast(constant_2), dimensions={}
  param_1 = $0[127,125]{1,0} parameter(1)
  multiply_splat = $0[127,125]{1,0} multiply(broadcast_splat, param_1)
  multiply = $0[127,125]{1,0} multiply(param_0, broadcast_splat)
  constant_neg_inf = $0[] constant(-inf)
  reduce = $0[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=max_computation
  broadcast = $0[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = $0[127,125]{1,0} subtract(param_0, broadcast)
}
)";
  const std::string hlo_string =
      absl::Substitute(hlo_string_template,
                       primitive_util::LowercasePrimitiveTypeName(data_type));

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
  EXPECT_TRUE(verifier().Run(module.get()).status().ok());
  VLOG(2) << module->ToString();
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseBinaryElementwiseOperationWhereFirstOperandIsASplatAndSecondOperandIsASharedSplatProducer) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule nonfusible_diamond
add_computation {
  arg_0.1 = f32[] parameter(0)
  arg_1.1 = f32[] parameter(1)
  ROOT add = f32[] add(arg_0.1, arg_1.1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  constant_2 = f32[] constant(0.333333343)
  broadcast_splat_shared = f32[127,125]{1,0} broadcast(constant_2), dimensions={}
  param_1 = f32[127,125]{1,0} parameter(1)
  multiply_splat_shared = f32[127,125]{1,0} multiply(broadcast_splat_shared, param_1)
  constant_3 = f32[] constant(0.5)
  broadcast_splat = f32[127,125]{1,0} broadcast(constant_3), dimensions={}
  multiply_splat = f32[127,125]{1,0} multiply(broadcast_splat, broadcast_splat_shared)
  multiply = f32[127,125]{1,0} multiply(param_0, multiply_splat)
  constant_neg_inf = f32[] constant(-inf)
  reduce = f32[127]{0} reduce(multiply, constant_neg_inf), dimensions={1}, to_apply=add_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(SoftmaxRewriterTritonTest, FusionDecisionIsCapturedExplicitly) {
  const std::string hlo_string = R"(
HloModule softmax
max_computation {
  arg_0 = f32[] parameter(0)
  arg_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(arg_0, arg_1)
}
ENTRY main {
  param_0 = f32[127,125]{1,0} parameter(0)
  identity = f32[] parameter(1)
  reduce = f32[127]{0} reduce(param_0, identity), dimensions={1}, to_apply=max_computation
  broadcast = f32[127,125]{1,0} broadcast(reduce), dimensions={0}
  ROOT subtract = f32[127,125]{1,0} subtract(param_0, broadcast)
}
)";

  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  SoftmaxRewriterTriton softmax_rewriter_triton(gpu_version_);
  int unmatched = 0, matched = 0;
  for (HloInstruction* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    DiamondMatchingDecision decision =
        softmax_rewriter_triton.MatchesTritonCompatibleClosedReductionDiamond(
            instruction);
    if (std::holds_alternative<FusionDecision>(decision)) {
      std::string actual_decision =
          std::get<FusionDecision>(decision).Explain();
      EXPECT_THAT(
          actual_decision,
          AnyOf(HasSubstr("Root is not elementwise binary"),
                HasSubstr("Reduction init value should be a constant or a "
                          "convert of a constant.")));
      unmatched++;
    } else {
      matched++;
    }
  }
  EXPECT_EQ(unmatched, 5);
  EXPECT_EQ(matched, 0);
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithBroadcastAlongReductionDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[32]{0} parameter(0)
  p1 = f32[32,16]{1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[32]{0} reduce(p1, c), dimensions={1}, to_apply=add_computation
  b0 = f32[32,16]{1,0} broadcast(r0), dimensions={0}
  b1 = f32[32,16]{1,0} broadcast(p0), dimensions={0}
  add0 = f32[32,16]{1,0} add(b1, p1)
  ROOT add1 = f32[32,16]{1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithBroadcastAlongBatchDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[16]{0} parameter(0)
  p1 = f32[32,16]{1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[32]{0} reduce(p1, c), dimensions={1}, to_apply=add_computation
  b0 = f32[32,16]{1,0} broadcast(r0), dimensions={0}
  b1 = f32[32,16]{1,0} broadcast(p0), dimensions={1}
  add0 = f32[32,16]{1,0} add(b1, p1)
  ROOT add1 = f32[32,16]{1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithMultiDimTensorBroadcastAlongBatchDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[32,16]{1,0} parameter(0)
  p1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[64,32]{1,0} reduce(p1, c), dimensions={2}, to_apply=add_computation
  b0 = f32[64,32,16]{2,1,0} broadcast(r0), dimensions={0,1}
  b1 = f32[64,32,16]{2,1,0} broadcast(p0), dimensions={1,2}
  add0 = f32[64,32,16]{2,1,0} add(b1, p1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpWithZeroDimTensorBroadcastAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  parameter_0 = f32[] parameter(0)
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  reduce_0 = f32[64,32]{1,0} reduce(parameter_1, c), dimensions={2}, to_apply=add_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_0), dimensions={}
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    FusesBinaryElementwiseIfIntermediateDiamondOpIsBroadcastOf1DParameterAlongNonReductionDimensions) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  parameter_0 = f32[16] parameter(0)
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  reduce_0 = f32[64,32]{1,0} reduce(parameter_1, c), dimensions={2}, to_apply=add_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_0), dimensions={2}
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_TRUE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseBinaryElementwiseIfIntermediateDiamondOpIsBroadcastOf1DParameterAlongBothBatchAndReductionDimensions) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  parameter_0 = f32[64] parameter(0)
  parameter_1 = f32[64,32,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  reduce_0 = f32[64,32]{1,0} reduce(parameter_1, c), dimensions={2}, to_apply=add_computation
  broadcast_0 = f32[64,32,16]{2,1,0} broadcast(reduce_0), dimensions={0,1}
  broadcast_1 = f32[64,32,16]{2,1,0} broadcast(parameter_0), dimensions={0}
  add_0 = f32[64,32,16]{2,1,0} add(broadcast_1, parameter_1)
  ROOT add1 = f32[64,32,16]{2,1,0} add(add_0, broadcast_0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseBinaryElementwiseIfIntermediateDiamondOpWithBroadcastAlongBatchAndReductionDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[8]{0} parameter(0)
  p1 = f32[32,8,16]{2,1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[32,8]{1,0} reduce(p1, c), dimensions={2}, to_apply=add_computation
  b0 = f32[32,8,16]{2,1,0} broadcast(r0), dimensions={0,1}
  b1 = f32[32,8,16]{2,1,0} broadcast(p0), dimensions={1}
  add0 = f32[32,8,16]{2,1,0} add(b1, p1)
  ROOT add1 = f32[32,8,16]{2,1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseBinaryElementwiseIfIntermediateDiamondOpWithPartialBroadcastToBatchDim) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[16,64]{1,0} parameter(0)
  p1 = f32[8,16,32,64]{3,2,1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[8,16,32]{2,1,0} reduce(p1, c), dimensions={3}, to_apply=add_computation
  b0 = f32[8,16,32,64]{3,2,1,0} broadcast(r0), dimensions={0,1,2}
  b1 = f32[8,16,32,64]{3,2,1,0} broadcast(p0), dimensions={1,3}
  add0 = f32[8,16,32,64]{3,2,1,0} add(b1, p1)
  ROOT add1 = f32[8,16,32,64]{3,2,1,0} add(add0, b0)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

TEST_F(
    SoftmaxRewriterTritonTest,
    DoesNotFuseBinaryElementwiseIfIntermediateDiamondOpWithMultiDimBroadcastAlongBatchDimAsParameter) {  // NOLINT(whitespace/line_length)
  const std::string hlo_string = R"(
HloModule h1

add_computation {
  y = f32[] parameter(1)
  x = f32[] parameter(0)
  ROOT add = f32[] add(x, y)
}

ENTRY main {
  p0 = f32[32,16]{1,0} parameter(0)
  p1 = f32[128,64,32,16]{3,2,1,0} parameter(1)
  c = f32[] constant(0)

  r0 = f32[128,64,32]{2,1,0} reduce(p1, c), dimensions={3}, to_apply=add_computation
  b0 = f32[128,64,32,16]{3,2,1,0} broadcast(r0), dimensions={0,1,2}
  b1 = f32[128,64,32,16]{3,2,1,0} broadcast(p0), dimensions={2,3}
  add0 = f32[128,64,32,16]{3,2,1,0} add(b1, p1)
  ROOT add1 = f32[128,64,32,16]{3,2,1,0} add(add0, b0)
})";
  auto module = ParseAndReturnVerifiedModule(hlo_string).value();
  EXPECT_FALSE(
      SoftmaxRewriterTritonMatchAndRewrite(gpu_version_, module.get()).value());
}

INSTANTIATE_TEST_SUITE_P(SoftmaxRewriterTritonTestSuite,
                         SoftmaxRewriterTritonTest,
                         ::testing::Values(F32, F16, BF16));

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla

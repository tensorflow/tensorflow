/* Copyright 2024 The OpenXLA Authors.

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

// TODO(b/343158720): Simplify the tests in this file after a generic emitter
// has landed.
#include "xla/service/gpu/triton_support.h"

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emitter_triton.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/triton_test_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::Not;
using ::testing::status::IsOk;

auto AllXlaDataTypes() {
  std::vector<xla::PrimitiveType> xla_data_types;
  std::vector<xla::PrimitiveType> to_filter_out = {PRIMITIVE_TYPE_INVALID,
                                                   TUPLE, OPAQUE_TYPE, TOKEN};
  const tsl::protobuf::EnumDescriptor* xla_type_descriptor =
      tsl::protobuf::GetEnumDescriptor<xla::PrimitiveType>();
  for (int enum_ix = 0; enum_ix < xla_type_descriptor->value_count();
       ++enum_ix) {
    xla::PrimitiveType xla_type = static_cast<xla::PrimitiveType>(
        xla_type_descriptor->value(enum_ix)->number());
    if (!absl::c_linear_search(to_filter_out, xla_type)) {
      xla_data_types.push_back(xla_type);
    }
  }
  return ::testing::ValuesIn(xla_data_types);
}

class TritonSupportTest : public TritonSupportTestBase {
 public:
  // Runs a support test for the given `TestedInstruction`. The support test
  // verifies that `IsTritonSupportedInstruction` is in sync with the
  // implemented Triton emitter, i.e., given an instruction `instr`, either
  //  -  `IsTritonSupportedInstruction(instr)` =>  Triton lowering is OK
  //  -  `!IsTritonSupportedInstruction(instr)` => Triton lowering is not OK.
  //
  // In order to make sure that the call succeeds in both cases, the user must
  // pass valid output tile sizes for the tested instruction/computation
  // as an additional parameter.
  //
  // In some cases, the Triton lowering is not handled gracefully by the
  // lowering code, and the lowering fails with a crash. In such cases, the
  // user can set `skip_failure_branch_to_avoid_crash` to `true` to skip the
  // lowering test when `IsTritonSupportedInstruction` returns `false`.
  void RunSupportTest(TestedInstruction ti,
                      std::vector<int64_t> output_tile_sizes,
                      bool skip_failure_branch_to_avoid_crash = false) {
    BlockLevelParameters block_level_parameters =
        FromOutputTileSizes(std::move(output_tile_sizes));
    if (IsTritonSupportedInstruction(ti.Instruction(),
                                     GetCudaComputeCapability())) {
      TF_EXPECT_OK(CreateTritonIrAndFileCheck(ti.TritonComputation(),
                                              block_level_parameters,
                                              "CHECK: tt.func @triton_fn"));
    } else {
      if (!skip_failure_branch_to_avoid_crash) {
        const se::DeviceDescription dev_info =
            TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
        EXPECT_THAT(
            TritonWrapper("test_fn", &ti.TritonFusion(),
                          GetCudaComputeCapability(), dev_info,
                          block_level_parameters, &llvm_module_, mlir_context_),
            Not(IsOk()));
      }
    }
  }
};

class TritonSupportTestWithParam : public TritonSupportTest,
                                   public ::testing::WithParamInterface<
                                       std::tuple<PrimitiveType, HloOpcode>> {};

using BitcastOrReshapeTest = TritonSupportTestWithParam;

TEST_P(BitcastOrReshapeTest, IsTritonSupportedBitcastOrReshape) {
  auto [data_type, opcode] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[1,16,4]{2,1,0} parameter(0)
  ROOT bitcast_or_reshape = $0[64]{0} $1(parameter_0)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16});
}

INSTANTIATE_TEST_SUITE_P(
    BitcastOrReshapeTestSuite, BitcastOrReshapeTest,
    ::testing::Combine(AllXlaDataTypes(),
                       ::testing::Values(HloOpcode::kBitcast,
                                         HloOpcode::kReshape)),
    TritonSupportTestParamsToString);

using UnaryElementwiseTest = TritonSupportTestWithParam;

// TODO(b/331636835): updates elementwise op tests to directly emit single op
// instead of relying on triton gemm kernel.
TEST_P(UnaryElementwiseTest, IsTritonSupportedUnaryElementwise) {
  auto [data_type, opcode] = GetParam();
  if (data_type == BF16 && SkipBF16Tests()) {
    GTEST_SKIP();
  }

  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  unary = $0[33,68]{1,0} $1(parameter_0)
  ROOT convert = f32[33,68]{1,0} convert(unary)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32});
}

INSTANTIATE_TEST_SUITE_P(
    UnaryElementwiseTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kConvert, HloOpcode::kAbs,
                                         HloOpcode::kNegate)),
    TritonSupportTestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    UnaryPREDTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED),
                       ::testing::Values(HloOpcode::kConvert, HloOpcode::kNot)),
    TritonSupportTestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    UnaryMathTestSuite, UnaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kCeil, HloOpcode::kCos,
                                         HloOpcode::kExp, HloOpcode::kExpm1,
                                         HloOpcode::kFloor, HloOpcode::kLog,
                                         HloOpcode::kLog1p, HloOpcode::kRsqrt,
                                         HloOpcode::kSin, HloOpcode::kSqrt,
                                         HloOpcode::kCbrt, HloOpcode::kTan,
                                         HloOpcode::kTanh, HloOpcode::kErf)),
    TritonSupportTestParamsToString);

using BinaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(BinaryElementwiseTest, IsTritonSupportedBinaryElementwise) {
  auto [data_type, opcode] = GetParam();
  if (data_type == BF16 && SkipBF16Tests()) {
    GTEST_SKIP();
  }

  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT binary = $0[11,63]{1,0} $1(parameter_0, parameter_1)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));

  bool skip_failure_branch_to_avoid_crash = false;
  if (data_type == F16 && opcode == HloOpcode::kDivide) {
    skip_failure_branch_to_avoid_crash = true;
  }
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32},
                 /*skip_failure_branch_to_avoid_crash=*/
                 skip_failure_branch_to_avoid_crash);
}

INSTANTIATE_TEST_SUITE_P(
    BinaryElementwiseTestSuite, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kAdd, HloOpcode::kMultiply,
                                         HloOpcode::kMaximum,
                                         HloOpcode::kMinimum,
                                         HloOpcode::kSubtract)),
    TritonSupportTestParamsToString);

INSTANTIATE_TEST_SUITE_P(BinaryPREDTestSuite, BinaryElementwiseTest,
                         ::testing::Combine(::testing::Values(PRED),
                                            ::testing::Values(HloOpcode::kAnd,
                                                              HloOpcode::kOr,
                                                              HloOpcode::kXor)),
                         TritonSupportTestParamsToString);
INSTANTIATE_TEST_SUITE_P(
    BinaryMathTestSuite, BinaryElementwiseTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kAtan2, HloOpcode::kDivide,
                                         HloOpcode::kPower)),
    TritonSupportTestParamsToString);

using CompareTest = TritonSupportTestWithParam;

TEST_P(CompareTest, IsTritonSupportedCompare) {
  auto [data_type, opcode] = GetParam();
  if (data_type == BF16 && SkipBF16Tests()) {
    GTEST_SKIP();
  }

  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  compare = pred[11,63]{1,0} $1(parameter_0, parameter_1), direction=GE
  ROOT convert = f32[11,63]{1,0} convert(compare)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32});
}

INSTANTIATE_TEST_SUITE_P(
    CompareTestSuite, CompareTest,
    ::testing::Combine(::testing::Values(PRED, S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kCompare)),
    TritonSupportTestParamsToString);

using TernaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(TernaryElementwiseTest, IsTritonSupportedTernaryElementwise) {
  auto [data_type, opcode] = GetParam();
  if (data_type == BF16 && SkipBF16Tests()) {
    GTEST_SKIP();
  }

  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[13,63]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = pred[13,63]{1,0} parameter(2)
  ternary = $0[13,63]{1,0} $1(parameter_2, parameter_0, parameter_1)
  ROOT convert = f32[13,63]{1,0} convert(ternary)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32});
}

INSTANTIATE_TEST_SUITE_P(
    TernaryElementwiseTestSuite, TernaryElementwiseTest,
    ::testing::Combine(::testing::Values(PRED, S8, S16, S32, F16, F32, BF16),
                       ::testing::Values(HloOpcode::kSelect)),
    TritonSupportTestParamsToString);

using ReduceConstTest = TritonSupportTestWithParam;

TEST_P(ReduceConstTest, IsTritonSupportedReduceWithConstInit) {
  auto [data_type, opcode] = GetParam();
  if (data_type == BF16 && SkipBF16Tests()) {
    GTEST_SKIP();
  }

  const std::string kHloTestTemplate = R"(
HloModule t
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  constant_0 = $0[] constant(0)
  ROOT reduce = $0[125]{0} $1(parameter_0, constant_0), dimensions={1}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1});
}

INSTANTIATE_TEST_SUITE_P(
    ReduceConstTestSuite, ReduceConstTest,
    ::testing::Combine(::testing::Values(F16, F32, BF16),
                       ::testing::Values(HloOpcode::kReduce)),
    TritonSupportTestParamsToString);

TEST_F(TritonSupportTest,
       SupportedReduceWithConvertConstantIsCodegenedSuccessfullyWithTriton) {
  if (SkipBF16Tests()) {
    GTEST_SKIP();
  }
  const std::string kHloTest = R"(
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = bf16[] constant(0)
  convert_0 = f32[] convert(constant_0)
  ROOT reduce = f32[125]{0} reduce(parameter_0, convert_0), dimensions={1}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kReduce));
  EXPECT_TRUE(
      IsTritonSupportedInstruction(ti.Instruction(), GetCudaComputeCapability())
          .CanFuse());
  TF_EXPECT_OK(ApplyFloatNormalization(ti.Module().get()));
  TF_EXPECT_OK(CreateTritonIrAndFileCheck(ti.TritonComputation(),
                                          FromOutputTileSizes({1}),
                                          "CHECK: tt.func @triton_fn"));
}

TEST_F(
    TritonSupportTestBase,
    UnsupportedReduceWithMoreThanOneReduceDimensionsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = f32[2,125,127]{2,1,0} parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[2]{0} reduce(parameter_0, constant_0), dimensions={1,2}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kReduce));
  EXPECT_THAT(
      IsTritonSupportedInstruction(ti.Instruction(), GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr(
          "Reduction is not a row-reduction of a single operand."));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, FromOutputTileSizes({1}), &llvm_module_,
                    mlir_context_),
      Not(IsOk()));
}

TEST_F(TritonSupportTest,
       UnsupportedReduceWithNonLastReduceDimensionFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[127]{0} reduce(parameter_0, constant_0), dimensions={0}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kReduce));
  EXPECT_THAT(
      IsTritonSupportedInstruction(ti.Instruction(), GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr(
          "Reduction is not a row-reduction of a single operand."));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, FromOutputTileSizes({1}), &llvm_module_,
                    mlir_context_),
      Not(IsOk()));
}

TEST_F(TritonSupportTest,
       UnsupportedReduceWithMoreThanOneOperandsFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
add {
  Arg_0 = f32[] parameter(0)
  Arg_2 = f32[] parameter(1)
  Arg_1 = f32[] parameter(2)
  Arg_3 = f32[] parameter(3)
  add_0 = f32[] add(Arg_0, Arg_2)
  add_1 = f32[] add(Arg_1, Arg_3)
  ROOT pair = (f32[], f32[]) tuple(add_0, add_1)
}

ENTRY triton_computation {
  parameter_0 = f32[125,127] parameter(0)
  constant_0 = f32[] constant(0)
  tuple_0 = (f32[125]{0}, f32[125]{0}) reduce(parameter_0, parameter_0, constant_0, constant_0), dimensions={1}, to_apply=add
  ROOT reduce = f32[125]{0} get-tuple-element(tuple_0), index=0
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kReduce));
  EXPECT_THAT(
      IsTritonSupportedInstruction(ti.Instruction(), GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr("Unsupported output data type"));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, FromOutputTileSizes({1}), &llvm_module_,
                    mlir_context_),
      Not(IsOk()));
}

TEST_F(TritonSupportTest,
       UnsupportedReduceWithNonConstReduceValueFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
add {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT add = f32[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  init = f32[] parameter(1)
  ROOT reduce = f32[125]{0} reduce(parameter_0, init), dimensions={1}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kReduce));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(
      IsTritonSupportedInstruction(ti.Instruction(), GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr("Reduction init value should be a constant "
                           "or a convert of a constant."));
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, FromOutputTileSizes({1}), &llvm_module_,
                    mlir_context_),
      tsl::testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr("operand->opcode() == HloOpcode::kConstant")));
}

TEST_F(TritonSupportTest,
       UnsupportedReductionComputationFailsGracefullyWithTriton) {
  const std::string kHloTest = R"(
custom_call {
  Arg_0 = f32[] parameter(0)
  Arg_1 = f32[] parameter(1)
  ROOT custom_call = f32[] custom-call(Arg_0, Arg_1), custom_call_target="foo"
}

ENTRY triton_computation {
  parameter_0 = f32[125,127]{1,0} parameter(0)
  constant_0 = f32[] constant(0)
  ROOT reduce = f32[125]{0} reduce(parameter_0, constant_0), dimensions={1}, to_apply=custom_call
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTest, /*data_type=*/{}, HloOpcode::kReduce));
  const se::DeviceDescription dev_info =
      TestGpuDeviceInfo::RTXA6000DeviceInfo(GetCudaComputeCapability());
  EXPECT_THAT(
      IsTritonSupportedInstruction(ti.Instruction(), GetCudaComputeCapability())
          .Explain(),
      ::testing::HasSubstr("Unsupported reduction computation by Triton."));
  EXPECT_THAT(
      TritonWrapper("test_fn", &ti.TritonFusion(), GetCudaComputeCapability(),
                    dev_info, FromOutputTileSizes({1}), &llvm_module_,
                    mlir_context_),
      tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("Unsupported operation")));
}
}  // namespace
}  // namespace gpu
}  // namespace xla

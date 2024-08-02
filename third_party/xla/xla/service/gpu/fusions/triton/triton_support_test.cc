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

#include "xla/service/gpu/fusions/triton/triton_support.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/fusions/triton/triton_fusion_emitter.h"
#include "xla/service/gpu/fusions/triton/triton_test_utils.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::Not;
using ::tsl::testing::IsOk;

std::vector<xla::PrimitiveType> AllXlaDataTypes() {
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
  return xla_data_types;
}

// Returns true if the given `opcode` supports the given `type` with respect to
// HLO semantics. This is completely independent of the what Triton supports or
// what the hardware supports.
//
// This function is used to decide what test combinations are generated. Without
// it we would need to generate all combinations and skip invalid ones. Because
// there are a lot of invalid combinations the test output will be very noisy.
// A slightly more robust alternative would be to call the HLO verifier
// directly, but that works at the level of an HLO instruction, which we don't
// have at the time we're deciding what tests to generate.
bool DoesOpSupportType(HloOpcode opcode, PrimitiveType type) {
  namespace pu = ::xla::primitive_util;

  switch (opcode) {
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kNot:
      return type == PRED || pu::IsIntegralType(type);
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kReal:
    case HloOpcode::kImag:
    case HloOpcode::kLogistic:
      return pu::IsFloatingPointType(type) || pu::IsComplexType(type);
    case HloOpcode::kErf:
    case HloOpcode::kFloor:
    case HloOpcode::kCeil:
    case HloOpcode::kIsFinite:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kReducePrecision:
      return pu::IsFloatingPointType(type);
    case HloOpcode::kClz:
    case HloOpcode::kPopulationCount:
      return pu::IsIntegralType(type);
    case HloOpcode::kAbs:
    case HloOpcode::kSign:
      return pu::IsSignedIntegralType(type) || pu::IsFloatingPointType(type) ||
             pu::IsComplexType(type);
    case HloOpcode::kPower:
    case HloOpcode::kAtan2:
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kNegate:
      return type != PRED;
    default:
      // Returning true by default ensures that newly added ops are not
      // skipped.
      return true;
  }
}

auto AllDevicesToTest() {
  using cc = se::GpuComputeCapability;
#ifdef TENSORFLOW_USE_ROCM
  se::RocmComputeCapability example_rocm_compute_capability =
      TestGpuDeviceInfo::AMDMI210DeviceInfo().rocm_compute_capability();
  return std::vector<cc>{cc(example_rocm_compute_capability)};
#else  // GOOGLE_CUDA
  return std::vector<cc>{cc(se::CudaComputeCapability::Ampere()),
                         cc(se::CudaComputeCapability::Hopper())};
#endif
}

// Generates all the possible test combinations for a given opcodes. A test
// combination is a tuple of the form (data_type, opcode, compute_capability).
auto AllTestCombinationsForOpcodes(std::vector<HloOpcode>&& opcodes) {
  std::vector<std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>>
      test_combinations;
  for (PrimitiveType data_type : AllXlaDataTypes()) {
    for (HloOpcode opcode : opcodes) {
      if (DoesOpSupportType(opcode, data_type)) {
        for (se::GpuComputeCapability cc : AllDevicesToTest()) {
          test_combinations.push_back({data_type, opcode, cc});
        }
      }
    }
  }
  return ::testing::ValuesIn(test_combinations);
};

class TritonSupportTest : public TritonSupportTestBase {
 public:
  // Runs a support test for the given `TestedInstruction` and the given
  // compute capability. The support test verifies that
  // `IsTritonSupportedInstruction` is in sync with the implemented Triton
  // emitter, i.e., given an instruction `instr`, either
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
                      se::GpuComputeCapability cc,
                      bool skip_failure_branch_to_avoid_crash = false) {
    BlockLevelParameters block_level_parameters =
        FromOutputTileSizes(std::move(output_tile_sizes));
    const se::DeviceDescription dev_info =
        std::holds_alternative<se::CudaComputeCapability>(cc)
            ? TestGpuDeviceInfo::RTXA6000DeviceInfo(cc)
            : TestGpuDeviceInfo::AMDMI210DeviceInfo();
    auto run_triton_codegen = [&]() {
      return TritonWrapper("test_fn", &ti.TritonFusion(), cc, dev_info,
                           block_level_parameters, &llvm_module_,
                           mlir_context_);
    };

    if (IsTritonSupportedInstruction(ti.Instruction(), cc)) {
      EXPECT_THAT(run_triton_codegen(), IsOk());
    } else {
      if (skip_failure_branch_to_avoid_crash) {
        EXPECT_DEATH(run_triton_codegen().IgnoreError(), "");

      } else {
        EXPECT_THAT(run_triton_codegen(), Not(IsOk()));
      }
    }
  }
};

class TritonSupportTestWithParam
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>> {};

using BitcastOrReshapeTest = TritonSupportTestWithParam;

TEST_P(BitcastOrReshapeTest, IsTritonSupportedBitcastOrReshape) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[1,16,4]{2,1,0} parameter(0)
  ROOT bitcast_or_reshape = $0[64]{0} $1(parameter_0)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16}, cc);
}

INSTANTIATE_TEST_SUITE_P(BitcastOrReshapeTestSuite, BitcastOrReshapeTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kBitcast,
                                                        HloOpcode::kReshape}),
                         TritonSupportTestTypeOpcodeAndDeviceToString);

using UnaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(UnaryElementwiseTest, IsTritonSupportedUnaryElementwise) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kDefaultHloTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  ROOT unary = $0[33,68]{1,0} $1(parameter_0)
})";

  // Used for elementwise ops that return f64 regardless of the input type (e.g.
  // Imag).
  const std::string kF64OutputTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  ROOT unary = f64[33,68]{1,0} $1(parameter_0)
})";

  // Used for elementwise ops that return pred regardless of the input type
  // (e.g. IsFinite).
  const std::string kPredOutputTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  ROOT unary = pred[33,68]{1,0} $1(parameter_0)
})";

  // Used for the ReducePrecision op, since it requires extra attributes.
  const std::string kReducePrecisionTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68]{1,0} parameter(0)
  ROOT unary = $0[33,68]{1,0} $1(parameter_0), exponent_bits=2, mantissa_bits=2
})";

  bool f64_output =
      opcode == HloOpcode::kReal || opcode == HloOpcode::kImag ||
      (opcode == HloOpcode::kAbs && primitive_util::IsComplexType(data_type));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(
          f64_output ? kF64OutputTemplate
                     : (opcode == HloOpcode::kIsFinite
                            ? kPredOutputTemplate
                            : (opcode == HloOpcode::kReducePrecision
                                   ? kReducePrecisionTemplate
                                   : kDefaultHloTemplate)),
          data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    UnaryElementwiseTestSuite, UnaryElementwiseTest,
    AllTestCombinationsForOpcodes({HloOpcode::kAbs,
                                   HloOpcode::kCbrt,
                                   HloOpcode::kCeil,
                                   HloOpcode::kClz,
                                   HloOpcode::kConvert,
                                   HloOpcode::kCos,
                                   HloOpcode::kErf,
                                   HloOpcode::kExp,
                                   HloOpcode::kExpm1,
                                   HloOpcode::kFloor,
                                   HloOpcode::kImag,
                                   HloOpcode::kIsFinite,
                                   HloOpcode::kLog,
                                   HloOpcode::kLog1p,
                                   HloOpcode::kLogistic,
                                   HloOpcode::kNegate,
                                   HloOpcode::kNot,
                                   HloOpcode::kPopulationCount,
                                   HloOpcode::kReal,
                                   HloOpcode::kReducePrecision,
                                   HloOpcode::kRoundNearestAfz,
                                   HloOpcode::kRoundNearestEven,
                                   HloOpcode::kRsqrt,
                                   HloOpcode::kSign,
                                   HloOpcode::kSin,
                                   HloOpcode::kSqrt,
                                   HloOpcode::kTan,
                                   HloOpcode::kTanh}),
    TritonSupportTestTypeOpcodeAndDeviceToString);

using BinaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(BinaryElementwiseTest, IsTritonSupportedBinaryElementwise) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  ROOT binary = $0[11,63]{1,0} $1(parameter_0, parameter_1)
})";

  // TODO(b/345763510): Investigate why the convert below is needed. If removed
  // the test fails because `pm.run(triton_module.get())` returns this error:
  //
  //    loc("compare"): error: 'tt.store' op failed to
  //    verify that value type matches ptr type
  const std::string kHloCompareTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[11,63]{1,0} parameter(0)
  parameter_1 = $0[11,63]{1,0} parameter(1)
  compare = pred[11,63]{1,0} $1(parameter_0, parameter_1), direction=GE
  ROOT convert = f32[11,63]{1,0} convert(compare)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(opcode == HloOpcode::kCompare
                                         ? kHloCompareTestTemplate
                                         : kHloTestTemplate,
                                     data_type, opcode));

  bool skip_failure_branch_to_avoid_crash =
      opcode == HloOpcode::kDivide &&
      (data_type == PrimitiveType::BF16 || data_type == PrimitiveType::F16 ||
       data_type == PrimitiveType::F8E5M2 ||
       data_type == PrimitiveType::F8E4M3FN);

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32}, cc,
                 skip_failure_branch_to_avoid_crash);
}

INSTANTIATE_TEST_SUITE_P(
    BinaryElementwiseTestSuite, BinaryElementwiseTest,
    AllTestCombinationsForOpcodes(
        {HloOpcode::kAnd, HloOpcode::kOr, HloOpcode::kXor, HloOpcode::kAdd,
         HloOpcode::kMultiply, HloOpcode::kMaximum, HloOpcode::kMinimum,
         HloOpcode::kSubtract, HloOpcode::kAtan2, HloOpcode::kDivide,
         HloOpcode::kRemainder, HloOpcode::kPower, HloOpcode::kShiftLeft,
         HloOpcode::kShiftRightArithmetic, HloOpcode::kShiftRightLogical,
         HloOpcode::kCompare}),
    TritonSupportTestTypeOpcodeAndDeviceToString);

using TernaryElementwiseTest = TritonSupportTestWithParam;

TEST_P(TernaryElementwiseTest, IsTritonSupportedTernaryElementwise) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $2[13,63]{1,0} parameter(0)
  parameter_1 = $0[13,63]{1,0} parameter(1)
  parameter_2 = $0[13,63]{1,0} parameter(2)
  ROOT ternary = $0[13,63]{1,0} $1(parameter_0, parameter_1, parameter_2)
})";

  auto type = primitive_util::LowercasePrimitiveTypeName(data_type);
  const std::string hlo_text =
      absl::Substitute(kHloTestTemplate, type, HloOpcodeString(opcode),
                       opcode == HloOpcode::kSelect ? "pred" : type);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32}, cc);
}

INSTANTIATE_TEST_SUITE_P(TernaryElementwiseTestSuite, TernaryElementwiseTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kSelect,
                                                        HloOpcode::kClamp}),
                         TritonSupportTestTypeOpcodeAndDeviceToString);

using ReduceTest = TritonSupportTestWithParam;

TEST_P(ReduceTest, IsTritonSupportedReduction) {
  GTEST_SKIP() << "TODO(b/348565795): this test is currently broken.";
  auto [data_type, opcode, cc] = GetParam();
  bool dtype_is_complex = data_type == C64 || data_type == C128;
  const std::string kHloTestTemplate =
      absl::Substitute(R"(
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  constant_0 = $0[] constant($1)
  ROOT reduce = $0[125]{0} reduce(parameter_0, constant_0),
    dimensions={1}, to_apply=add
})",
                       "$0", dtype_is_complex ? "(0, 0)" : "0");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_F(ReduceTest, IsTritonSupportedReductionWithMultidimensionalTile) {
  const std::string kHloTestTemplate = R"(
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $0[3,125,127]{2,1,0} parameter(0)
  constant_0 = $0[] constant(0)
  ROOT reduce = $0[3,125]{1,0} reduce(parameter_0, constant_0),
    dimensions={2}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(kHloTestTemplate, F32,
                                                         HloOpcode::kReduce));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{3, 4},
                 se::CudaComputeCapability::Ampere());
}

TEST_P(
    ReduceTest,
    UnsupportedReduceWithMoreThanOneReduceDimensionsFailsGracefullyWithTriton) {
  auto [data_type, opcode, cc] = GetParam();
  bool dtype_is_complex = data_type == C64 || data_type == C128;
  const std::string kHloTestTemplate =
      absl::Substitute(R"(
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $0[2,125,127]{2,1,0} parameter(0)
  constant_0 = $0[] constant($1)
  ROOT reduce = $0[2]{0} reduce(parameter_0, constant_0),
    dimensions={1,2}, to_apply=add
})",
                       "$0", dtype_is_complex ? "(0, 0)" : "0");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  EXPECT_FALSE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_P(ReduceTest, IsTritonSupportedReduceWithNonLastReduceDimension) {
  GTEST_SKIP() << "TODO(b/348565795): this test is currently broken.";
  auto [data_type, opcode, cc] = GetParam();
  bool dtype_is_complex = data_type == C64 || data_type == C128;
  const std::string kHloTestTemplate =
      absl::Substitute(R"(
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  constant_0 = $0[] constant($1)
  ROOT reduce = $0[127]{0} reduce(parameter_0, constant_0), dimensions={0}, to_apply=add
})",
                       "$0", dtype_is_complex ? "(0, 0)" : "0");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_P(ReduceTest,
       UnsupportedReduceWithMoreThanOneOperandsFailsGracefullyWithTriton) {
  auto [data_type, opcode, cc] = GetParam();
  bool dtype_is_complex = data_type == C64 || data_type == C128;
  const std::string kHloTestTemplate =
      absl::Substitute(R"(
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  Arg_2 = $0[] parameter(2)
  Arg_3 = $0[] parameter(3)
  add_0 = $0[] add(Arg_0, Arg_2)
  add_1 = $0[] add(Arg_1, Arg_3)
  ROOT pair = ($0[], $0[]) tuple(add_0, add_1)
}

ENTRY triton_computation {
  parameter_0 = $0[125,127] parameter(0)
  constant_0 = $0[] constant($1)
  tuple = ($0[125]{0}, $0[125]{0}) reduce(
    parameter_0, parameter_0, constant_0, constant_0),
      dimensions={1}, to_apply=add
  ROOT reduce = $0[125]{0} get-tuple-element(tuple), index=0
})",
                       "$0", dtype_is_complex ? "(0, 0)" : "0");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  EXPECT_FALSE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_F(ReduceTest, ReduceWithNonConstReduceValueIsSupportedWithTriton) {
  const se::GpuComputeCapability cc = se::CudaComputeCapability::Ampere();
  const std::string kHloTestTemplate = R"(
add {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT add = $0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  init = $0[] parameter(1)
  ROOT reduce = $0[125]{0} reduce(parameter_0, init), dimensions={1}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(kHloTestTemplate, F32,
                                                         HloOpcode::kReduce));
  EXPECT_TRUE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2}, cc);
}

TEST_P(ReduceTest, UnsupportedReductionComputationFailsGracefullyWithTriton) {
  auto [data_type, opcode, cc] = GetParam();
  bool dtype_is_complex = data_type == C64 || data_type == C128;
  const std::string kHloTestTemplate =
      absl::Substitute(R"(
custom_call {
  Arg_0 = $0[] parameter(0)
  Arg_1 = $0[] parameter(1)
  ROOT custom_call = $0[] custom-call(Arg_0, Arg_1), custom_call_target="foo"
}

ENTRY triton_computation {
  parameter_0 = $0[125,127]{1,0} parameter(0)
  constant_0 = $0[] constant($1)
  ROOT reduce = $0[125]{0} reduce(parameter_0, constant_0),
    dimensions={1}, to_apply=custom_call
})",
                       "$0", dtype_is_complex ? "(0, 0)" : "0");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  EXPECT_FALSE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(ReduceTestSuite, ReduceTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kReduce}),
                         TritonSupportTestTypeOpcodeAndDeviceToString);

using CollectiveTest = TritonSupportTestWithParam;

TEST_P(CollectiveTest, UnsupportedCollectivesFailGracefullyWithTriton) {
  auto [data_type, opcode, cc] = GetParam();
  absl::flat_hash_map<HloOpcode, std::string> kHloCollectiveTestTemplates = {
      {
          HloOpcode::kAllGather,
          R"(
            ENTRY triton_computation {
              input = $0[128,32]{0,1} parameter(0)
              ROOT all-gather = $0[128,128]{0,1} all-gather(input),
              replica_groups={}, dimensions={1}
            }
          )",
      },
      {
          HloOpcode::kAllReduce,
          R"(
            apply_op {
              x = $0[] parameter(0)
              y = $0[] parameter(1)
              ROOT apply_op = $0[] add(x, y)
            }

            ENTRY triton_computation {
              input = $0[128,32] parameter(0)
              ROOT all-reduce = $0[128,32] all-reduce(input), replica_groups={}, to_apply=apply_op
            }
          )",
      },
      {
          HloOpcode::kAllToAll,
          R"(
             ENTRY triton_computation {
               input = f32[128,32]{0,1} parameter(0)
               ROOT a2a = (f32[128,32]{0,1}) all-to-all(input), replica_groups={}
             }
          )",
      },
      {HloOpcode::kCollectivePermute,
       R"(
          ENTRY triton_computation {
            a = $0[] parameter(0)
            ROOT collective-permute = $0[] collective-permute(a), source_target_pairs={{1,0}, {0,1}, {2,2}}
          }
        )"},
      {HloOpcode::kReduceScatter,
       R"(
          apply_op {
            lhs = $0[] parameter(0)
            rhs = $0[] parameter(1)
            ROOT apply_op = $0[] add(lhs, rhs)
          }

          ENTRY triton_computation {
            input = $0[8] parameter(0)
            ROOT result = $0[4] reduce-scatter(input), replica_groups={},
                              dimensions={0}, to_apply=apply_op
          }
        )"}};

  std::string hlo_template = kHloCollectiveTestTemplates.at(opcode);
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_template, data_type, opcode));
  EXPECT_FALSE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    CollectiveTestSuite, CollectiveTest,
    AllTestCombinationsForOpcodes({HloOpcode::kAllGather, HloOpcode::kAllReduce,
                                   HloOpcode::kAllToAll,
                                   HloOpcode::kCollectivePermute,
                                   HloOpcode::kReduceScatter}),
    TritonSupportTestTypeOpcodeAndDeviceToString);

}  // namespace
}  // namespace gpu
}  // namespace xla

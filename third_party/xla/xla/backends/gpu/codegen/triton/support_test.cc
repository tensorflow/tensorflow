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

#include "xla/backends/gpu/codegen/triton/support.h"

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

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
    case HloOpcode::kAtan2:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kReal:
    case HloOpcode::kImag:
    case HloOpcode::kLogistic:
      return pu::IsFloatingPointType(type) || pu::IsComplexType(type);
    case HloOpcode::kCbrt:
    case HloOpcode::kErf:
    case HloOpcode::kFloor:
    case HloOpcode::kCeil:
    case HloOpcode::kIsFinite:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kReducePrecision:
      return pu::IsFloatingPointType(type);
    case HloOpcode::kClz:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kPopulationCount:
      return pu::IsIntegralType(type);
    case HloOpcode::kAbs:
    case HloOpcode::kSign:
      return pu::IsSignedIntegralType(type) || pu::IsFloatingPointType(type) ||
             pu::IsComplexType(type);
    case HloOpcode::kPower:
    case HloOpcode::kDivide:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kNegate:
    case HloOpcode::kIota:
      return type != PRED;
    case HloOpcode::kRng:
      return !pu::IsComplexType(type);
    case HloOpcode::kComplex:
      return type == F32 || type == F64;
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
auto AllTestCombinationsForOpcodes(absl::Span<const HloOpcode> opcodes) {
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
    // Ensure that the caller provided the right number of output tile sizes.
    // If that is not the case, codegen could fail for that reason---which
    // wouldn't give any valuable signal here.  We skip the check for non-array
    // output shapes, since we have no meaningful way of providing tile sizes
    // for them at the moment.
    if (ti.Instruction().shape().IsArray()) {
      ASSERT_EQ(output_tile_sizes.size(),
                ti.Instruction().shape().dimensions_size());
    }
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
        EXPECT_DEATH(
            // We need to catch exceptions and abort(), because in OSS there
            // seem to be cases where exceptions are used instead of terminating
            // the program.
            try { run_triton_codegen().IgnoreError(); } catch (...) {
              abort();
            },
            // It's not possible to find stable matching patterns for all
            // aborting code paths that occur here, so we at least make sure
            // that we don't interpret sanitizer errors as success.
            ::testing::Not(::testing::HasSubstr("Sanitizer:")));

      } else {
        EXPECT_THAT(run_triton_codegen(), Not(IsOk()));
      }
    }
  }
};

class TritonSupportTestWithTypeAndOpcodeAndDeviceParam
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, HloOpcode, se::GpuComputeCapability>> {};

using BitcastOrReshapeTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(BitcastOrReshapeTest, IsTritonSupportedBitcastOrReshape) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[1,16,4] parameter(0)
  ROOT bitcast_or_reshape = $0[64] $1(parameter_0)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16}, cc);
}

TEST_P(BitcastOrReshapeTest, IsTritonSupported0DBitcastOrReshape) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[1,1,1] parameter(0)
  ROOT bitcast_or_reshape = $0[] $1(parameter_0)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

constexpr std::array kTestedOpsBitcastReshape = {HloOpcode::kBitcast,
                                                 HloOpcode::kReshape};

INSTANTIATE_TEST_SUITE_P(
    BitcastOrReshapeTestSuite, BitcastOrReshapeTest,
    AllTestCombinationsForOpcodes(kTestedOpsBitcastReshape),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using UnaryElementwiseTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(UnaryElementwiseTest, IsTritonSupportedUnaryElementwise) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kDefaultHloTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68] parameter(0)
  ROOT unary = $0[33,68] $1(parameter_0)
})";

  // Used for elementwise ops that return f64 regardless of the input type (e.g.
  // Imag).
  const std::string kF64OutputTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68] parameter(0)
  ROOT unary = f64[33,68] $1(parameter_0)
})";

  // Used for elementwise ops that return pred regardless of the input type
  // (e.g. IsFinite).
  const std::string kPredOutputTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68] parameter(0)
  ROOT unary = pred[33,68] $1(parameter_0)
})";

  // Used for the ReducePrecision op, since it requires extra attributes.
  const std::string kReducePrecisionTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68] parameter(0)
  ROOT unary = $0[33,68] $1(parameter_0), exponent_bits=2, mantissa_bits=2
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

constexpr std::array kTestedOpsUnaryElementwise = {
    // clang-format off
    // go/keep-sorted start
    HloOpcode::kAbs,
    HloOpcode::kCbrt,
    HloOpcode::kCeil,
    HloOpcode::kClz,
    HloOpcode::kCopy,
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
    HloOpcode::kTanh
    // go/keep-sorted end
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(
    UnaryElementwiseTestSuite, UnaryElementwiseTest,
    AllTestCombinationsForOpcodes(kTestedOpsUnaryElementwise),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

class ConvertTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, se::GpuComputeCapability>> {
};

TEST_P(ConvertTest, Convert) {
  auto [data_type_in, data_type_out, cc] = GetParam();

  const std::string hlo_text = absl::Substitute(
      R"(
ENTRY triton_computation {
  parameter_0 = $0[33,68] parameter(0)
  ROOT convert = $1[33,68] convert(parameter_0)
})",
      primitive_util::LowercasePrimitiveTypeName(data_type_in),
      primitive_util::LowercasePrimitiveTypeName(data_type_out));

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(
          hlo_text, data_type_in,  // The type provided here is irrelevant.
          HloOpcode::kConvert));

  bool skip_failure_branch_to_avoid_crash = false;

  // The two variables below are only needed prior to C++20 as capturing
  // structured bindings is not supported.
  // TODO(b/328238952): remove this indirection after XLA moves to C++20.
  PrimitiveType captured_in = data_type_in;
  PrimitiveType captured_out = data_type_out;

  auto any_is = [=](PrimitiveType compare) {
    return captured_in == compare || captured_out == compare;
  };

  if (data_type_in != data_type_out && any_is(PrimitiveType::F8E4M3FN) &&
      std::holds_alternative<se::CudaComputeCapability>(cc) &&
      !std::get<se::CudaComputeCapability>(cc).IsAtLeastHopper()) {
    skip_failure_branch_to_avoid_crash |=
        any_is(F16) || any_is(BF16) || any_is(F32);

    // Crashes due to unsupported/unspecified rounding mode.
    skip_failure_branch_to_avoid_crash |=
        (data_type_in == PrimitiveType::F8E4M3FN &&
         data_type_out == PrimitiveType::F64);
  }

  // Crashes due to unsupported/unspecified rounding mode.
  skip_failure_branch_to_avoid_crash |=
      (any_is(PrimitiveType::F8E4M3FN) && any_is(PrimitiveType::F8E5M2)) ||
      (data_type_in == PrimitiveType::F64 &&
       (data_type_out == PrimitiveType::F8E4M3FN ||
        data_type_out == PrimitiveType::F8E5M2));

  // Crashes due to unsupported conversion.
  skip_failure_branch_to_avoid_crash |=
      (data_type_out == PrimitiveType::F64 &&
       (data_type_in == PrimitiveType::F8E4M3FN ||
        data_type_in == PrimitiveType::F8E5M2));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32}, cc,
                 skip_failure_branch_to_avoid_crash);
}

constexpr std::array kTestedOpsConvert = {HloOpcode::kConvert};

INSTANTIATE_TEST_SUITE_P(
    ConvertTestSuite, ConvertTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTwoTypesAndDeviceToString);

using BinaryElementwiseTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(BinaryElementwiseTest, IsTritonSupportedBinaryElementwise) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[11,63] parameter(0)
  parameter_1 = $0[11,63] parameter(1)
  ROOT binary = $0[11,63] $1(parameter_0, parameter_1)
})";

  const std::string kHloCompareTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[11,63] parameter(0)
  parameter_1 = $0[11,63] parameter(1)
  ROOT compare = pred[11,63] $1(parameter_0, parameter_1), direction=GE
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

TEST_P(BinaryElementwiseTest, IsTritonSupportedBinaryElementwise0D) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[] parameter(0)
  parameter_1 = $0[] parameter(1)
  ROOT binary = $0[] $1(parameter_0, parameter_1)
})";

  const std::string kHloCompareTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[] parameter(0)
  parameter_1 = $0[] parameter(1)
  ROOT compare = pred[] $1(parameter_0, parameter_1), direction=GE
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

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc,
                 skip_failure_branch_to_avoid_crash);
}

constexpr std::array kTestedOpsBinaryElementwise = {
    // clang-format off
    // go/keep-sorted start
    HloOpcode::kAdd,
    HloOpcode::kAnd,
    HloOpcode::kAtan2,
    HloOpcode::kCompare,
    HloOpcode::kDivide,
    HloOpcode::kMaximum,
    HloOpcode::kMinimum,
    HloOpcode::kMultiply,
    HloOpcode::kOr,
    HloOpcode::kPower,
    HloOpcode::kRemainder,
    HloOpcode::kShiftLeft,
    HloOpcode::kShiftRightArithmetic,
    HloOpcode::kShiftRightLogical,
    HloOpcode::kSubtract,
    HloOpcode::kXor,
    // go/keep-sorted end
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(
    BinaryElementwiseTestSuite, BinaryElementwiseTest,
    AllTestCombinationsForOpcodes(kTestedOpsBinaryElementwise),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using TernaryElementwiseTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(TernaryElementwiseTest, IsTritonSupportedTernaryElementwise) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $2[13,63] parameter(0)
  parameter_1 = $0[13,63] parameter(1)
  parameter_2 = $0[13,63] parameter(2)
  ROOT ternary = $0[13,63] $1(parameter_0, parameter_1, parameter_2)
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

constexpr std::array kTestedOpsTernaryElementwise = {HloOpcode::kSelect,
                                                     HloOpcode::kClamp};

INSTANTIATE_TEST_SUITE_P(
    TernaryElementwiseTestSuite, TernaryElementwiseTest,
    AllTestCombinationsForOpcodes(kTestedOpsTernaryElementwise),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using ReduceTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

static absl::string_view init_value(PrimitiveType dtype) {
  if (dtype == C64 || dtype == C128) {
    return "(0, 0)";
  } else if (dtype == F8E8M0FNU) {
    return "1e-40";
  } else {
    return "0";
  }
}

TEST_P(ReduceTest, IsTritonSupportedReduction) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(R"(
add {
  Arg_0 = $$0[] parameter(0)
  Arg_1 = $$0[] parameter(1)
  ROOT add = $$0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $$0[125,127] parameter(0)
  constant_0 = $$0[] constant($0)
  ROOT reduce = $$0[125] reduce(parameter_0, constant_0),
    dimensions={1}, to_apply=add
})",
                                                        init_value(data_type));
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
  parameter_0 = $0[3,125,127] parameter(0)
  constant_0 = $0[] constant(0)
  ROOT reduce = $0[3,125] reduce(parameter_0, constant_0),
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
  const std::string kHloTestTemplate = absl::Substitute(R"(
add {
  Arg_0 = $$0[] parameter(0)
  Arg_1 = $$0[] parameter(1)
  ROOT add = $$0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $$0[2,125,127] parameter(0)
  constant_0 = $$0[] constant($0)
  ROOT reduce = $$0[2] reduce(parameter_0, constant_0),
    dimensions={1,2}, to_apply=add
})",
                                                        init_value(data_type));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_P(ReduceTest, IsTritonSupportedReduceWithNonLastReduceDimension) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(R"(
add {
  Arg_0 = $$0[] parameter(0)
  Arg_1 = $$0[] parameter(1)
  ROOT add = $$0[] add(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $$0[125,127] parameter(0)
  constant_0 = $$0[] constant($0)
  ROOT reduce = $$0[127] reduce(parameter_0, constant_0), dimensions={0}, to_apply=add
})",
                                                        init_value(data_type));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_P(ReduceTest,
       UnsupportedReduceWithMoreThanOneOperandsFailsGracefullyWithTriton) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(R"(
add {
  Arg_0 = $$0[] parameter(0)
  Arg_1 = $$0[] parameter(1)
  Arg_2 = $$0[] parameter(2)
  Arg_3 = $$0[] parameter(3)
  add_0 = $$0[] add(Arg_0, Arg_2)
  add_1 = $$0[] add(Arg_1, Arg_3)
  ROOT pair = ($$0[], $$0[]) tuple(add_0, add_1)
}

ENTRY triton_computation {
  parameter_0 = $$0[125,127] parameter(0)
  constant_0 = $$0[] constant($0)
  tuple = ($$0[125], $$0[125]) reduce(
    parameter_0, parameter_0, constant_0, constant_0),
      dimensions={1}, to_apply=add
  ROOT reduce = $$0[125] get-tuple-element(tuple), index=0
})",
                                                        init_value(data_type));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
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
  parameter_0 = $0[125,127] parameter(0)
  init = $0[] parameter(1)
  ROOT reduce = $0[125] reduce(parameter_0, init), dimensions={1}, to_apply=add
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(kHloTestTemplate, F32,
                                                         HloOpcode::kReduce));
  EXPECT_TRUE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2}, cc);
}

TEST_P(ReduceTest, UnsupportedReductionComputationFailsGracefullyWithTriton) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(R"(
custom_call {
  Arg_0 = $$0[] parameter(0)
  Arg_1 = $$0[] parameter(1)
  ROOT custom_call = $$0[] custom-call(Arg_0, Arg_1), custom_call_target="foo"
}

ENTRY triton_computation {
  parameter_0 = $$0[125,127] parameter(0)
  constant_0 = $$0[] constant($0)
  ROOT reduce = $$0[125] reduce(parameter_0, constant_0),
    dimensions={1}, to_apply=custom_call
})",
                                                        init_value(data_type));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

constexpr std::array kTestedOpsReduction = {HloOpcode::kReduce};

INSTANTIATE_TEST_SUITE_P(ReduceTestSuite, ReduceTest,
                         AllTestCombinationsForOpcodes(kTestedOpsReduction),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using ReductionComputationTest =
    TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

// The test below tests what kind of binary element-wise operations are
// supported within a reduction's computation.
//
// Note that there is a difference in what is supported inside the reduction
// computation and in regular HLO. See triton_support.cc for more details.
TEST_P(ReductionComputationTest, DifferentBinaryOps) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(
      R"(
reduce_computation {
  Arg_0 = $$0[] parameter(0)
  Arg_1 = $$0[] parameter(1)
  ROOT output = $$0[] $0(Arg_0, Arg_1)
}

ENTRY triton_computation {
  parameter_0 = $$0[125,127] parameter(0)
  constant_0 = $$0[] constant($1)
  ROOT reduce = $$0[125] reduce(parameter_0, constant_0),
    dimensions={1}, to_apply=reduce_computation
})",
      HloOpcodeString(opcode), init_value(data_type));

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTestTemplate, data_type, HloOpcode::kReduce));

  // TODO(b/361526623): Reduce the cases where setting
  // skip_failure_branch_to_avoid_crash is needed.
  bool skip_failure_branch_to_avoid_crash =
      opcode == HloOpcode::kDivide &&
      (data_type == BF16 || data_type == F16 || data_type == F8E4M3FN ||
       data_type == F8E5M2);

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc,
                 skip_failure_branch_to_avoid_crash);
}

std::vector<HloOpcode> ExcludeOps(absl::Span<const HloOpcode> all_ops,
                                  absl::Span<const HloOpcode> ops_to_exclude) {
  std::vector<HloOpcode> ret;
  for (HloOpcode op : all_ops) {
    if (!absl::c_linear_search(ops_to_exclude, op)) {
      ret.push_back(op);
    }
  }
  return ret;
}

INSTANTIATE_TEST_SUITE_P(
    ReductionComputationTestSuite, ReductionComputationTest,
    AllTestCombinationsForOpcodes(ExcludeOps(kTestedOpsBinaryElementwise,
                                             {HloOpcode::kCompare})),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using TransposeTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(TransposeTest, LoadTranspose3D) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter_0 = $0[125,127,37] parameter(0)
  ROOT transpose = $0[127,37,125] $1(parameter_0), dimensions={1,2,0}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32, 16}, cc);
}

constexpr std::array kTestedOpsTranspose = {HloOpcode::kTranspose};

INSTANTIATE_TEST_SUITE_P(TransposeTestSuite, TransposeTest,
                         AllTestCombinationsForOpcodes(kTestedOpsTranspose),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

class TritonSupportTestWithTypeAndDeviceParam
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, se::GpuComputeCapability>> {};

using SliceTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(SliceTest, ContinuousSlice) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = (R"(
ENTRY triton_computation {
  p = $0[128,32] parameter(0)
  ROOT slice = $0[12,5] $1(p), slice={[116:128], [20:25]}
})");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{8, 4}, cc);
}

TEST_P(SliceTest, NonContinuousSliceWhereStrideDividesOffsetEvenly) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = (R"(
ENTRY triton_computation {
  p = f32[16,16,32] parameter(0)
  ROOT slice = f32[4,4,8] slice(p), slice={[2:10:2], [2:6], [3:11]}
})");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2, 2}, cc);
}

TEST_P(SliceTest, NonContinuousSliceWhereStrideDoesNotDivideOffsetEvenly) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = (R"(
ENTRY triton_computation {
  p = f32[16,16,32] parameter(0)
  ROOT slice = f32[4,4,8] slice(p), slice={[3:11:2], [2:6], [3:11]}
})");
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2, 2}, cc);
}

constexpr std::array kTestedOpsSlice = {HloOpcode::kSlice};

INSTANTIATE_TEST_SUITE_P(SliceTestSuite, SliceTest,
                         AllTestCombinationsForOpcodes(kTestedOpsSlice),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using CollectiveTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(CollectiveTest, UnsupportedAllGatherFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  ROOT all-gather = $0[128,128] all-gather(input),
    replica_groups={}, dimensions={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kAllGather));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedAllGatherStartFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  ROOT all-gather-start = ($0[128,32], $0[256,32]) all-gather-start(input),
    replica_groups={{0,1}}, dimensions={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kAllGatherStart));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedAllGatherDoneFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = ($0[128,32], $0[128,32]) parameter(0)
  ROOT all-gather-done = $0[128,32] all-gather-done(input)
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kAllGatherDone));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedAllReduceFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
apply_op {
  x = $0[] parameter(0)
  y = $0[] parameter(1)
  ROOT apply_op = $0[] add(x, y)
}

ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  ROOT all-reduce = $0[128,32] all-reduce(input), replica_groups={},
      to_apply=apply_op
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kAllReduce));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest,
       UnsupportedAllReduceStartAndDoneFailGracefullyWithTriton) {
  // 'all-reduce-start' and 'all-reduce-done' need to be tested together, since
  // the HLO verifier relies on one directly consuming the other.
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
apply_op {
  x = $0[] parameter(0)
  y = $0[] parameter(1)
  ROOT apply_op = $0[] add(x, y)
}

ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  all-reduce-start = $0[128,32] all-reduce-start(input), replica_groups={},
      to_apply=apply_op
  ROOT all-reduce-done = $0[128,32] all-reduce-done(all-reduce-start)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_start,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kAllReduceStart));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_done,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kAllReduceDone));
  RunSupportTest(std::move(ti_start), /*output_tile_sizes=*/{2, 2}, cc);
  RunSupportTest(std::move(ti_done), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedAllToAllFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  ROOT a2a = ($0[128,32]) all-to-all(input), replica_groups={}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kAllToAll));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedCollectivePermuteFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  a = $0[128,32] parameter(0)
  ROOT collective-permute = $0[128,32] collective-permute(a),
      source_target_pairs={{1,0}, {0,1}, {2,2}}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kCollectivePermute));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest,
       UnsupportedCollectivePermuteStartAndDoneFailGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  // 'collective-permute-start' and 'collective-permute-done' need to be tested
  // together, since the HLO verifier relies on one directly consuming the
  // other.
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  a = $0[128,32] parameter(0)
  start = ($0[128,32], $0[128,32]) collective-permute-start(a),
      source_target_pairs={{1,0}, {0,1}, {2,2}}
  ROOT done = $0[128,32] collective-permute-done(start)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_start,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kCollectivePermuteStart));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_done,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kCollectivePermuteDone));

  RunSupportTest(std::move(ti_start), /*output_tile_sizes=*/{2, 2}, cc);
  RunSupportTest(std::move(ti_done), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedReduceScatterFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
apply_op {
  lhs = $0[] parameter(0)
  rhs = $0[] parameter(1)
  ROOT apply_op = $0[] add(lhs, rhs)
}

ENTRY triton_computation {
  input = $0[8] parameter(0)
  ROOT result = $0[4] reduce-scatter(input), replica_groups={},
      dimensions={0}, to_apply=apply_op
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kReduceScatter));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

TEST_P(CollectiveTest,
       UnsupportedAsyncStartAndUpdateAndDoneFailGracefullyWithTriton) {
  // 'async-start', 'async-update', and 'async-done' need to be tested together,
  // since the HLO verifier requires 'async-start' and 'async-done' to always
  // appear together within a module.
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
async_computation {
  ROOT p0 = $0[10] parameter(0)
}

ENTRY triton_computation {
  input = $0[10] parameter(0)
  async-start = (($0[10]), $0[10]) async-start(input),
    calls=async_computation
  async-update = (($0[10]), $0[10]) async-update(async-start),
    calls=async_computation
  ROOT async-done = $0[10] async-done(async-update), calls=async_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_start,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kAsyncStart));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_update,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kAsyncUpdate));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_done,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kAsyncDone));
  RunSupportTest(std::move(ti_start), /*output_tile_sizes=*/{1}, cc);
  RunSupportTest(std::move(ti_update), /*output_tile_sizes=*/{1}, cc);
  RunSupportTest(std::move(ti_done), /*output_tile_sizes=*/{1}, cc);
}

TEST_P(CollectiveTest,
       UnsupportedCollectiveBroadcastFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  ROOT result = $0[128,32] collective-broadcast(input), replica_groups={}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kCollectiveBroadcast));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

TEST_P(CollectiveTest, UnsupportedReplicaIdFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  ROOT replica_id = u32[] replica-id()
})";

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kReplicaId));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

TEST_P(CollectiveTest, UnsupportedPartitionIdFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  ROOT partition_id = u32[] partition-id()
})";

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kPartitionId));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

TEST_P(CollectiveTest, UnsupportedRaggedAllToAllFailsGracefullyWithTriton) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[128,32] parameter(0)
  output = $0[128,32] parameter(1)
  input_offsets = s32[1] parameter(2)
  send_sizes = s32[1] parameter(3)
  output_offsets = s32[1] parameter(4)
  recv_sizes = s32[1] parameter(5)
  ROOT root = $0[128,32] ragged-all-to-all(input, output, input_offsets, send_sizes, output_offsets, recv_sizes), replica_groups={}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kRaggedAllToAll));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

constexpr std::array kTestedOpsCollectives = {
    // clang-format off
    // go/keep-sorted start
    HloOpcode::kAllGather,
    HloOpcode::kAllGatherDone,
    HloOpcode::kAllGatherStart,
    HloOpcode::kAllReduce,
    HloOpcode::kAllReduceDone,
    HloOpcode::kAllReduceStart,
    HloOpcode::kAllToAll,
    HloOpcode::kAsyncDone,
    HloOpcode::kAsyncStart,
    HloOpcode::kAsyncUpdate,
    HloOpcode::kCollectiveBroadcast,
    HloOpcode::kCollectivePermute,
    HloOpcode::kCollectivePermuteDone,
    HloOpcode::kCollectivePermuteStart,
    HloOpcode::kPartitionId,
    HloOpcode::kRaggedAllToAll,
    HloOpcode::kReduceScatter,
    HloOpcode::kReplicaId
    // go/keep-sorted end
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P(
    CollectiveTestSuite, CollectiveTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using BroadcastTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(BroadcastTest, Broadcast) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[35,131] parameter(0)
  ROOT bcast = $0[3,35,131,12] broadcast(input), dimensions={1,2}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kBroadcast));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 16, 32, 8}, cc);
}

constexpr std::array kTestedOpsBroadcast = {HloOpcode::kBroadcast};

INSTANTIATE_TEST_SUITE_P(
    BroadcastTestSuite, BroadcastTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using ParameterTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(ParameterTest, Parameter) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = $0[35,131] parameter(0)
  // TODO(b/363961478) remove the line below once parameters can be ROOT.
  ROOT noop = $0[35,131] convert(input)
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kParameter));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc);
}

constexpr std::array kTestedOpsParameter = {HloOpcode::kParameter};

INSTANTIATE_TEST_SUITE_P(
    ParameterTestSuite, ParameterTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using ConstantTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(ConstantTest, ConstantEffectiveScalar) {
  // The IsTritonSupportedReduction effectively tests the scalar constant
  // support.
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(R"(
ENTRY triton_computation {
  ROOT const = $$0[1,1] constant({{$0}})
})",
                                                        init_value(data_type));

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kConstant));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 1}, cc);
}

TEST_P(ConstantTest, Constant2D) {
  // The IsTritonSupportedReduction effectively tests the scalar constant
  // support.
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(R"(
ENTRY triton_computation {
  ROOT const = $$0[3,3] constant({{$0,$0,$0},{$0,$0,$0},{$0,$0,$0}})
})",
                                                        init_value(data_type));

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kConstant));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

constexpr std::array kTestedOpsConstant = {HloOpcode::kConstant};

INSTANTIATE_TEST_SUITE_P(
    ConstantTestSuite, ConstantTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using IotaTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(IotaTest, Iota2D) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  ROOT input = $0[35,131] iota(), iota_dimension=0
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc);
}

constexpr std::array kTestedOpsIota = {HloOpcode::kIota};

INSTANTIATE_TEST_SUITE_P(IotaTestSuite, IotaTest,
                         AllTestCombinationsForOpcodes(kTestedOpsIota),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using RngTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(RngTest, Rng) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  low = $0[] parameter(0)
  high = $0[] parameter(1)
  ROOT root = $0[33,77] rng(low, high), distribution=rng_uniform
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc);
}

constexpr std::array kTestedOpsRng = {HloOpcode::kRng};

INSTANTIATE_TEST_SUITE_P(RngTestSuite, RngTest,
                         AllTestCombinationsForOpcodes(kTestedOpsRng),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using RngBitGeneratorTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(RngBitGeneratorTest, RngBitGenerator) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  state = u64[2] parameter(0)
  ROOT root = (u64[2], $0[33,77]) rng-bit-generator(state), algorithm=rng_three_fry
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc);
}

constexpr std::array kTestedOpsRngBitGenerator = {HloOpcode::kRngBitGenerator};

INSTANTIATE_TEST_SUITE_P(
    RngBitGeneratorTestSuite, RngBitGeneratorTest,
    AllTestCombinationsForOpcodes(kTestedOpsRngBitGenerator),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

class TritonSupportTestWithDeviceParam
    : public TritonSupportTest,
      public ::testing::WithParamInterface<se::GpuComputeCapability> {};

using RngGetAndUpdateStateTest = TritonSupportTestWithDeviceParam;

TEST_P(RngGetAndUpdateStateTest, RngGetAndUpdateState) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
  ENTRY triton_computation {
    ROOT root = u64[2]{0} rng-get-and-update-state(), delta=4096
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate,
                                     F16,  // data_type doesn't matter here
                                     HloOpcode::kRngGetAndUpdateState));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}
constexpr std::array kTestedOpsRngGetAndUpdateState = {
    HloOpcode::kRngGetAndUpdateState};

INSTANTIATE_TEST_SUITE_P(RngGetAndUpdateStateTestSuite,
                         RngGetAndUpdateStateTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using ComplexTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(ComplexTest, Complex) {
  auto [data_type, opcode, cc] = GetParam();

  const std::string kF32HloTestTemplate = R"(
ENTRY triton_computation {
  real = $0[33,77] parameter(0)
  imag = $0[33,77] parameter(1)
  ROOT root = c64[33,77] complex(real, imag)
})";
  const std::string kF64HloTestTemplate = R"(
ENTRY triton_computation {
  real = $0[33,77] parameter(0)
  imag = $0[33,77] parameter(1)
  ROOT root = c128[33,77] complex(real, imag)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(
          data_type == F32 ? kF32HloTestTemplate : kF64HloTestTemplate,
          data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc);
}

constexpr std::array kTestedOpsComplex = {HloOpcode::kComplex};

INSTANTIATE_TEST_SUITE_P(ComplexTestSuite, ComplexTest,
                         AllTestCombinationsForOpcodes(kTestedOpsComplex),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

constexpr std::array kUnsupportedOps = {
    // clang-format off
    // go/keep-sorted start
    HloOpcode::kAddDependency,
    HloOpcode::kAfterAll,
    HloOpcode::kBatchNormGrad,
    HloOpcode::kBatchNormInference,
    HloOpcode::kBatchNormTraining,
    HloOpcode::kBitcastConvert,
    HloOpcode::kCall,
    HloOpcode::kCholesky,
    HloOpcode::kConcatenate,
    HloOpcode::kConditional,
    HloOpcode::kConvolution,
    HloOpcode::kCopyDone,
    HloOpcode::kCopyStart,
    HloOpcode::kCustomCall,
    HloOpcode::kDomain,
    HloOpcode::kDot,
    HloOpcode::kDynamicReshape,
    HloOpcode::kDynamicSlice,
    HloOpcode::kDynamicUpdateSlice,
    HloOpcode::kFft,
    HloOpcode::kFusion,
    HloOpcode::kGather,
    HloOpcode::kGetDimensionSize,
    HloOpcode::kGetTupleElement,
    HloOpcode::kInfeed,
    HloOpcode::kMap,
    HloOpcode::kOptimizationBarrier,
    HloOpcode::kOutfeed,
    HloOpcode::kPad,
    HloOpcode::kRaggedDot,
    HloOpcode::kRecv,
    HloOpcode::kRecvDone,
    HloOpcode::kReduceWindow,
    HloOpcode::kReverse,
    HloOpcode::kScatter,
    HloOpcode::kSelectAndScatter,
    HloOpcode::kSend,
    HloOpcode::kSendDone,
    HloOpcode::kSetDimensionSize,
    HloOpcode::kSort,
    HloOpcode::kStochasticConvert,
    HloOpcode::kTopK,
    HloOpcode::kTriangularSolve,
    HloOpcode::kTuple,
    HloOpcode::kWhile
    // go/keep-sorted end
    // clang-format on
};

absl::flat_hash_set<HloOpcode> AllTestedOpcodes() {
  absl::flat_hash_set<HloOpcode> ret;
  ret.insert(kTestedOpsBitcastReshape.begin(), kTestedOpsBitcastReshape.end());
  ret.insert(kTestedOpsUnaryElementwise.begin(),
             kTestedOpsUnaryElementwise.end());
  ret.insert(kTestedOpsConvert.begin(), kTestedOpsConvert.end());
  ret.insert(kTestedOpsBinaryElementwise.begin(),
             kTestedOpsBinaryElementwise.end());
  ret.insert(kTestedOpsTernaryElementwise.begin(),
             kTestedOpsTernaryElementwise.end());
  ret.insert(kTestedOpsReduction.begin(), kTestedOpsReduction.end());
  ret.insert(kTestedOpsSlice.begin(), kTestedOpsSlice.end());
  ret.insert(kTestedOpsTranspose.begin(), kTestedOpsTranspose.end());
  ret.insert(kTestedOpsCollectives.begin(), kTestedOpsCollectives.end());
  ret.insert(kTestedOpsBroadcast.begin(), kTestedOpsBroadcast.end());
  ret.insert(kTestedOpsParameter.begin(), kTestedOpsParameter.end());
  ret.insert(kTestedOpsConstant.begin(), kTestedOpsConstant.end());
  ret.insert(kTestedOpsIota.begin(), kTestedOpsIota.end());
  ret.insert(kTestedOpsRng.begin(), kTestedOpsRng.end());
  ret.insert(kTestedOpsRngBitGenerator.begin(),
             kTestedOpsRngBitGenerator.end());
  ret.insert(kTestedOpsRngGetAndUpdateState.begin(),
             kTestedOpsRngGetAndUpdateState.end());
  ret.insert(kTestedOpsComplex.begin(), kTestedOpsComplex.end());

  ret.insert(kUnsupportedOps.begin(), kUnsupportedOps.end());
  return ret;
}

TEST(OpCoverage, UnsupportedOpcodes) {
  for (HloOpcode opcode : kUnsupportedOps) {
    EXPECT_TRUE(internal::IsTritonUnsupportedOpcode(opcode));
  }
}

TEST(OpCoverage, AllOpcodesAreTested) {
  absl::flat_hash_set<HloOpcode> tested_opcodes = AllTestedOpcodes();
  for (int opcode_index = 0; opcode_index < HloOpcodeCount(); ++opcode_index) {
    auto opcode = static_cast<HloOpcode>(opcode_index);
    EXPECT_TRUE(tested_opcodes.contains(opcode))
        << "Opcode `" << HloOpcodeString(opcode) << "` is not tested.";
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

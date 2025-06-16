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
#include <iterator>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/test_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::Not;
using ::tsl::testing::IsOk;

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
    case HloOpcode::kCholesky:
    case HloOpcode::kTriangularSolve:
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
    case HloOpcode::kDot:
      return type != PRED;
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kStochasticConvert:
      return pu::IsFloatingPointType(type);
    default:
      // Returning true by default ensures that newly added ops are not
      // skipped.
      return true;
  }
}

std::vector<xla::PrimitiveType> AllOpSupportedTypes(HloOpcode opcode) {
  std::vector<xla::PrimitiveType> result;
  absl::c_copy_if(AllXlaDataTypes(), std::back_inserter(result),
                  [&](PrimitiveType data_type) {
                    return DoesOpSupportType(opcode, data_type);
                  });
  return result;
}

template <typename Predicate>
std::vector<xla::PrimitiveType> XlaDataTypesMatching(Predicate predicate) {
  std::vector<xla::PrimitiveType> result;
  absl::c_copy_if(AllXlaDataTypes(), std::back_inserter(result), predicate);
  return result;
}

std::vector<PrecisionConfig::Algorithm> AllPrecisionAlgorithms() {
  std::vector<PrecisionConfig::Algorithm> algorithms;
  const tsl::protobuf::EnumDescriptor* algorithm_descriptor =
      tsl::protobuf::GetEnumDescriptor<PrecisionConfig::Algorithm>();
  for (int enum_ix = 0; enum_ix < algorithm_descriptor->value_count();
       ++enum_ix) {
    algorithms.push_back(static_cast<PrecisionConfig::Algorithm>(
        algorithm_descriptor->value(enum_ix)->number()));
  }
  return algorithms;
}

auto AllDevicesToTest() {
  using cc = se::GpuComputeCapability;
#ifdef TENSORFLOW_USE_ROCM
  se::RocmComputeCapability example_rocm_compute_capability =
      TestGpuDeviceInfo::AMDMI210DeviceInfo().rocm_compute_capability();
  return std::vector<cc>{cc(example_rocm_compute_capability)};
#else  // GOOGLE_CUDA
  return std::vector<cc>{cc(se::CudaComputeCapability::Ampere()),
                         cc(se::CudaComputeCapability::Hopper()),
                         cc(se::CudaComputeCapability::Blackwell())};
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

// Expected failure mode of the Triton lowering.
enum class ExpectedFailMode {
  // Denotes a graceful failure, e.g. a verifier failure, or an absl::Status.
  kFail,
  // Denotes a crash. That is typically the case when encountering a bug in
  // the Triton compiler itself.
  kCrash,
  // Use only in cases when the failure mode depends on the compilation mode
  // (i.e. when the failure is caused by a CHECK).
  kFailOrCrash,
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
                      ExpectedFailMode failure_mode = ExpectedFailMode::kFail) {
    // output_tile_sizes is embedded in a vector of 1 element to share the logic
    // with the multiple output tiles case.
    RunSupportTestMultipleOutputTiles(
        std::move(ti), {std::move(output_tile_sizes)}, cc, failure_mode);
  }

  void RunSupportTestMultipleOutputTiles(
      TestedInstruction ti, std::vector<std::vector<int64_t>> output_tile_sizes,
      se::GpuComputeCapability cc,
      ExpectedFailMode failure_mode = ExpectedFailMode::kFail) {
    // Ensure that the caller provided the right number of output tile sizes.
    // If that is not the case, codegen could fail for that reason---which
    // wouldn't give any valuable signal here. The check is only done for array
    // and tuple shapes (only one layer of nesting is supported for tuples).
    const auto& root_instruction = ti.TritonComputation().root_instruction();
    if (root_instruction->shape().IsArray()) {
      ASSERT_EQ(output_tile_sizes.size(), 1);
      ASSERT_EQ(output_tile_sizes[0].size(),
                root_instruction->shape().dimensions().size());
    } else if (root_instruction->shape().IsTuple()) {
      ASSERT_EQ(output_tile_sizes.size(),
                root_instruction->shape().tuple_shapes().size());
      for (int64_t i = 0; i < output_tile_sizes.size(); ++i) {
        const auto& shape = root_instruction->shape().tuple_shapes(i);
        if (shape.IsTuple() || shape.IsToken()) {
          continue;  // No validation for nested tuples, as there is no way to
                     // specify output tile sizes for them.
        }
        ASSERT_TRUE(shape.IsArray());
        ASSERT_EQ(shape.dimensions().size(), output_tile_sizes[i].size());
      }
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
      EXPECT_THAT(run_triton_codegen(), IsOk()) << ti.Module()->ToString();
      return;
    }
    if (failure_mode == ExpectedFailMode::kFail) {
      EXPECT_THAT(run_triton_codegen(), Not(IsOk()));
      return;
    }
    EXPECT_DEATH(
        // We need to catch exceptions and abort(), because in OSS there
        // seem to be cases where exceptions are used instead of terminating
        // the program.
        try {
          absl::StatusOr<TritonWrapperResult> s = run_triton_codegen();
          if (!s.ok() && failure_mode == ExpectedFailMode::kFailOrCrash) {
            // Force a crash if failure is also acceptable.
            abort();
          }
        } catch (...) { abort(); },
        // It's not possible to find stable matching patterns for all
        // aborting code paths that occur here, so we at least make sure
        // that we don't interpret sanitizer errors as success.
        ::testing::Not(::testing::HasSubstr("Sanitizer:")));
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

using PadTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(PadTest, IsTritonSupportedHighPad) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = $0[4, 4] parameter(0)
  p1 = $0[] parameter(1)
  ROOT pad = $0[32, 16] $1(p0, p1), padding=0_28_0x0_12_0
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode,
                                     /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4, 4}, cc);
}

TEST_P(PadTest, IsTritonSupportedInteriorPad) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = $0[4] parameter(0)
  p1 = $0[] parameter(1)
  ROOT pad = $0[7] $1(p0, p1), padding=0_0_1
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode,
                                     /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4}, cc);
}

TEST_P(PadTest, IsTritonSupportedLowPad) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = $0[4] parameter(0)
  p1 = $0[] parameter(1)
  ROOT pad = $0[8] $1(p0, p1), padding=4_0_0
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode,
                                     /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4}, cc);
}

constexpr std::array kTestedOpsPad = {HloOpcode::kPad};

INSTANTIATE_TEST_SUITE_P(PadTestSuite, PadTest,
                         AllTestCombinationsForOpcodes(kTestedOpsPad),
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

class DynamicSliceTest : public TritonSupportTest,
                         public ::testing::WithParamInterface<PrimitiveType> {};

TEST_P(DynamicSliceTest, IsTritonSupportedDynamicSliceBufferType) {
  auto data_type = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = $0[64,128] parameter(0)
  off0 = s32[] parameter(1)
  off1 = s32[] parameter(2)
  ROOT r = $0[8, 16] dynamic-slice(p0, off0, off1), dynamic_slice_sizes={8, 16}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kDynamicSlice,
                                     /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4, 8},
                 se::CudaComputeCapability::Ampere());
}

INSTANTIATE_TEST_SUITE_P(
    DynamicSliceTestSuite, DynamicSliceTest,
    ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kDynamicSlice)),
    TritonSupportTestTypeToString);

using DynamicSliceOffsetsTest = DynamicSliceTest;

TEST_P(DynamicSliceOffsetsTest, IsTritonSupportedDynamicSliceOffsetType) {
  auto data_type = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = f32[64,128] parameter(0)
  off0 = $0[] parameter(1)
  off1 = $0[] parameter(2)
  ROOT r = f32[8, 16] dynamic-slice(p0, off0, off1), dynamic_slice_sizes={8, 16}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kDynamicSlice,
                                     /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4, 8},
                 se::CudaComputeCapability::Ampere());
}

INSTANTIATE_TEST_SUITE_P(DynamicSliceOffsetTestSuite, DynamicSliceOffsetsTest,
                         ::testing::ValuesIn(XlaDataTypesMatching(
                             ::xla::primitive_util::IsIntegralType)),
                         TritonSupportTestTypeToString);

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
    HloOpcode::kOptimizationBarrier,
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

  // The two variables below are only needed prior to C++20 as capturing
  // structured bindings is not supported.
  // TODO(b/328238952): remove this indirection after XLA moves to C++20.
  PrimitiveType captured_in = data_type_in;
  PrimitiveType captured_out = data_type_out;

  auto any_is = [=](PrimitiveType compare) {
    return captured_in == compare || captured_out == compare;
  };

  bool crashes_on_failure = false;
  if (data_type_in != data_type_out && any_is(PrimitiveType::F8E4M3FN) &&
      std::holds_alternative<se::CudaComputeCapability>(cc) &&
      !std::get<se::CudaComputeCapability>(cc).IsAtLeastHopper()) {
    crashes_on_failure |= any_is(F16) || any_is(BF16) || any_is(F32);

    // Crashes due to unsupported/unspecified rounding mode.
    crashes_on_failure |= (data_type_in == PrimitiveType::F8E4M3FN &&
                           data_type_out == PrimitiveType::F64);

    crashes_on_failure |=
        any_is(PrimitiveType::F8E4M3FN) && any_is(PrimitiveType::F8E5M2);
  }

  // Crashes due to unsupported/unspecified rounding mode.
  crashes_on_failure |= (data_type_in == PrimitiveType::F64 &&
                         (data_type_out == PrimitiveType::F8E4M3FN ||
                          data_type_out == PrimitiveType::F8E5M2));

  // Crashes due to unsupported conversion.
  crashes_on_failure |= (data_type_out == PrimitiveType::F64 &&
                         (data_type_in == PrimitiveType::F8E4M3FN ||
                          data_type_in == PrimitiveType::F8E5M2));

  RunSupportTest(
      std::move(ti), /*output_tile_sizes=*/{1, 32}, cc,
      crashes_on_failure ? ExpectedFailMode::kCrash : ExpectedFailMode::kFail);
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

  ExpectedFailMode fail_mode = ExpectedFailMode::kFail;
  if (opcode == HloOpcode::kDivide &&
      (data_type == PrimitiveType::BF16 || data_type == PrimitiveType::F16 ||
       data_type == PrimitiveType::F8E5M2 ||
       data_type == PrimitiveType::F8E4M3FN)) {
    fail_mode = ExpectedFailMode::kCrash;
  };

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32}, cc, fail_mode);
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

  ExpectedFailMode fail_mode = ExpectedFailMode::kFail;
  if (opcode == HloOpcode::kDivide &&
      (data_type == PrimitiveType::BF16 || data_type == PrimitiveType::F16 ||
       data_type == PrimitiveType::F8E5M2 ||
       data_type == PrimitiveType::F8E4M3FN)) {
    fail_mode = ExpectedFailMode::kCrash;
  }

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc, fail_mode);
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
  bool crashes_on_failure = data_type == PrimitiveType::F8E4M3FN ||
                            data_type == PrimitiveType::F8E5M2;
  RunSupportTest(
      std::move(ti), /*output_tile_sizes=*/{1}, cc,
      crashes_on_failure ? ExpectedFailMode::kCrash : ExpectedFailMode::kFail);
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

  bool crashes_on_failure = data_type == PrimitiveType::F8E4M3FN ||
                            data_type == PrimitiveType::F8E5M2;
  RunSupportTest(
      std::move(ti), /*output_tile_sizes=*/{1}, cc,
      crashes_on_failure ? ExpectedFailMode::kCrash : ExpectedFailMode::kFail);
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
  ROOT reduce = ($$0[125], $$0[125]) reduce(
    parameter_0, parameter_0, constant_0, constant_0),
      dimensions={1}, to_apply=add
})",
                                                        init_value(data_type));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTestMultipleOutputTiles(std::move(ti),
                                    /*output_tile_sizes=*/{{1}, {1}}, cc);
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

  // TODO(b/361526623): Reduce the cases where emitter crashes.
  ExpectedFailMode fail_mode = ExpectedFailMode::kFail;
  if (opcode == HloOpcode::kDivide && (data_type == BF16 || data_type == F16)) {
    fail_mode = ExpectedFailMode::kCrash;
  }
  if (data_type == F8E4M3FN || data_type == F8E5M2) {
    fail_mode = ExpectedFailMode::kFailOrCrash;
  }
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc, fail_mode);
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

class TritonSupportTestWithDeviceParam
    : public TritonSupportTest,
      public ::testing::WithParamInterface<se::GpuComputeCapability> {};

using ConcatenateDeviceTest = TritonSupportTestWithDeviceParam;

TEST_P(ConcatenateDeviceTest,
       TritonDoesNotSupportConcatenateOfUnnestedParameters) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = $0[18,128,20] parameter(0)
  p1 = $0[18,128,20] parameter(1)
  p2 = $0[18,128,20] parameter(2)
  ROOT concatenate = $0[18,384,20] concatenate(p0, p1, p2), dimensions={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTestTemplate, F32, HloOpcode::kConcatenate,
                              /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 64, 1}, cc);
}

INSTANTIATE_TEST_SUITE_P(ConcatenateTestSuite, ConcatenateDeviceTest,
                         ::testing::ValuesIn(AllDevicesToTest()));

// TODO(b/393299275): remove the boolean parameter once the migration is
// complete.
class TritonSupportTestWithTypeAndDeviceAndBoolParam
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, se::GpuComputeCapability, bool>> {
 public:
  static std::string ParamToString(
      const ::testing::TestParamInfo<ParamType>& info) {
    auto [data_type, cc, use_nested_gemm_fusions] = info.param;
    return absl::StrCat(PrimitiveType_Name(data_type), "_",
                        ComputeCapabilityToString(cc), "_",
                        use_nested_gemm_fusions ? "nested_gemm_fusions"
                                                : "no_nested_gemm_fusions");
  }
};

using ConcatenateTest = TritonSupportTestWithTypeAndDeviceAndBoolParam;

TEST_P(ConcatenateTest, IsTritonSupportedConcatenate) {
  auto [data_type, cc, use_nested_gemm_fusions] = GetParam();
  const std::string kHloTestTemplate = R"(
nest0 {
  ROOT p0 = $0[128] parameter(0)
}

nest1 {
  ROOT p0 = $0[128] parameter(0)
}

nest2 {
  ROOT p0 = $0[128] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128] parameter(0)
  p1 = $0[128] parameter(1)
  p2 = $0[128] parameter(2)

  fusion0 = $0[128] fusion(p0), kind=kCustom, calls=nest0, backend_config={
    "fusion_backend_config":{"kind":"__triton_nested_gemm_fusion",
    "block_level_fusion_config":{"output_tiles":[{"sizes":["64"]}]}}}
  fusion1 = $0[128] fusion(p1), kind=kCustom, calls=nest1, backend_config={
    "fusion_backend_config":{"kind":"__triton_nested_gemm_fusion",
    "block_level_fusion_config":{"output_tiles":[{"sizes":["64"]}]}}}
  fusion2 = $0[128] fusion(p2), kind=kCustom, calls=nest2, backend_config={
    "fusion_backend_config":{"kind":"__triton_nested_gemm_fusion",
    "block_level_fusion_config":{"output_tiles":[{"sizes":["64"]}]}}}
  ROOT result = $0[384] concatenate(fusion0, fusion1, fusion2), dimensions={0}
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kConcatenate,
                                                    use_nested_gemm_fusions));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{64}, cc);
}

constexpr std::array kTestedOpsConcatenate = {HloOpcode::kConcatenate};

INSTANTIATE_TEST_SUITE_P(
    ConcatenateTestSuite, ConcatenateTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest()),
                       ::testing::Bool()),
    ConcatenateTest::ParamToString);

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
  RunSupportTestMultipleOutputTiles(std::move(ti),
                                    /*output_tile_sizes=*/{{2, 2}, {2, 2}}, cc);
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
  std::string hlo_test_template =
      R"(
ENTRY triton_computation {
  ROOT root = $0[35,131] parameter(0)
})";
  if (data_type == S4) {  // S4 is not a valid output, convert it to S8.
    hlo_test_template = R"(
ENTRY triton_computation {
  input = $0[35,131] parameter(0)
  ROOT noop = s8[35,131] convert(input)
})";
  }
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_test_template, data_type,
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
  RunSupportTestMultipleOutputTiles(std::move(ti),
                                    /*output_tile_sizes=*/{{1}, {16, 32}}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    RngBitGeneratorTestSuite, RngBitGeneratorTest,
    AllTestCombinationsForOpcodes({HloOpcode::kRngBitGenerator}),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using RngGetAndUpdateStateTest = TritonSupportTestWithDeviceParam;

TEST_P(RngGetAndUpdateStateTest, RngGetAndUpdateState) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  ROOT root = u64[2]{0} rng-get-and-update-state(), delta=4096
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kRngGetAndUpdateState));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

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

INSTANTIATE_TEST_SUITE_P(ComplexTestSuite, ComplexTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kComplex}),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using ConditionalTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(ConditionalTest, Conditional) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
true_branch {
  p_true = $0[10] parameter(0)
  ROOT add = $0[10] add(p_true, p_true)
}
false_branch {
  p_false = $0[10] parameter(0)
  ROOT mul = $0[10] multiply(p_false, p_false)
}
ENTRY triton_computation {
  cond = pred[] parameter(0)
  operand = $0[10] parameter(1)
  ROOT conditional_op = $0[10] conditional(cond, operand, operand),
                              true_computation=true_branch,
                              false_computation=false_branch
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    ConditionalTestSuite, ConditionalTest,
    AllTestCombinationsForOpcodes({HloOpcode::kConditional}),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using WhileTest = TritonSupportTestWithDeviceParam;
// TODO: b/363981282 - Add tests for more data types.
TEST_P(WhileTest, While) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
body {
  constant = s32[] constant(1)
  prev = s32[] parameter(0)
  ROOT add = s32[] add(constant, prev)
}
condition {
  constant = s32[] constant(5)
  prev = s32[] parameter(0)
  ROOT greater-than = pred[] compare(constant, prev), direction=GT
}
ENTRY triton_computation {
  constant = s32[] constant(0)
  ROOT while = s32[] while(constant), condition=condition, body=body
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kWhile));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

INSTANTIATE_TEST_SUITE_P(WhileTestSuite, WhileTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using CallTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(CallTest, Call) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
called_computation {
  p = $0[10] parameter(0)
  ROOT add = $0[10] add(p, p)
}

ENTRY triton_computation {
  operand = $0[10] parameter(0)
  ROOT call_op = $0[10] call(operand), to_apply=called_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(CallTestSuite, CallTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kCall}),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using BatchNormInferenceTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(BatchNormInferenceTest, BatchNormInference) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  operand = $0[4,8,16,32] parameter(0)
  scale = $0[32] parameter(1)
  offset = $0[32] parameter(2)
  mean = $0[32] parameter(3)
  variance = $0[32] parameter(4)
  ROOT bn_inf = $0[4,8,16,32] batch-norm-inference(operand, scale, offset, mean, variance),
    epsilon=0.001, feature_index=3
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 1, 4, 8}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    BatchNormInferenceSuite, BatchNormInferenceTest,
    AllTestCombinationsForOpcodes({HloOpcode::kBatchNormInference}),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using BatchNormTrainingTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(BatchNormTrainingTest, BatchNormTraining) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  operand = $0[4,8,16,32] parameter(0)
  scale = $0[32] parameter(1)
  offset = $0[32] parameter(2)
  ROOT bn_train = ($0[4,8,16,32], $0[32], $0[32]) batch-norm-training(operand, scale, offset),
    epsilon=0.001, feature_index=3
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTestMultipleOutputTiles(
      std::move(ti), /*output_tile_sizes=*/{{1, 1, 4, 8}, {1}, {1}}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    BatchNormTrainingSuite, BatchNormTrainingTest,
    AllTestCombinationsForOpcodes({HloOpcode::kBatchNormTraining}),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using BatchNormGradTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(BatchNormGradTest, BatchNormGrad) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  operand = $0[4,8,16,32] parameter(0)
  scale = $0[32] parameter(1)
  mean = $0[32] parameter(2)
  variance = $0[32] parameter(3)
  grad_output = $0[4,8,16,32] parameter(4)
  ROOT bn_grad = ($0[4,8,16,32], $0[32], $0[32]) batch-norm-grad(operand, scale, mean, variance, grad_output),
    epsilon=0.001, feature_index=3
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTestMultipleOutputTiles(
      std::move(ti), /*output_tile_sizes=*/{{1, 1, 4, 8}, {1}, {1}}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    BatchNormGradSuite, BatchNormGradTest,
    AllTestCombinationsForOpcodes({HloOpcode::kBatchNormGrad}),
    TritonSupportTestTypeAndOpcodeAndDeviceToString);

using DomainTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(DomainTest, Domain) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  operand = $0[] parameter(0)
  ROOT domain_op = $0[] domain(operand), domain={kind="sharding", entry={maximal device=0}, exit={maximal device=1}}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

INSTANTIATE_TEST_SUITE_P(DomainSuite, DomainTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kDomain}),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using GetDimensionSizeTest = TritonSupportTestWithDeviceParam;

TEST_P(GetDimensionSizeTest, GetDimensionSize) {
  const auto cc = GetParam();

  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  operand = s32[16, 32] parameter(0)
  ROOT get_dim_size = s32[] get-dimension-size(operand), dimensions={1}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kGetDimensionSize));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

INSTANTIATE_TEST_SUITE_P(GetDimensionSizeSuite, GetDimensionSizeTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using ReverseTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(ReverseTest, Reverse) {
  auto [data_type, opcode, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  operand = $0[16,32] parameter(0)
  ROOT reverse_op = $0[16,32] reverse(operand), dimensions={0, 1}
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4, 8}, cc);
}

INSTANTIATE_TEST_SUITE_P(ReverseSuite, ReverseTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kReverse}),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

using DotTest = TritonSupportTest;

class DotTypesTest
    : public DotTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, se::GpuComputeCapability>> {
 public:
  static std::string ParamToString(
      const ::testing::TestParamInfo<DotTypesTest::ParamType>& data) {
    auto [result_type, input_type, cc] = data.param;
    return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(result_type),
                        "_",
                        primitive_util::LowercasePrimitiveTypeName(input_type),
                        "_", ComputeCapabilityToString(cc));
  };
};

TEST_P(DotTypesTest, Dot) {
  // Testing B[] = dot(A[], A[]).
  auto [result_type, input_type, cc] = GetParam();

  ExpectedFailMode fail_mode = ExpectedFailMode::kFail;
  if (input_type == F8E4M3FN || result_type == F8E4M3FN) {
    if (auto* cuda_cc = std::get_if<se::CudaComputeCapability>(&cc);
        cuda_cc && !cuda_cc->IsAtLeastHopper()) {
      // Hits llvm::report_fatal_error during Triton compilation.
      fail_mode = ExpectedFailMode::kFailOrCrash;
    }
  }

  std::string hlo_text = R"(
flhs {
  ROOT result = $0[128,256] parameter(0)
}

frhs {
  ROOT result = $0[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,256] parameter(0)
  p1 = $0[256,512] parameter(1)
  lhs = $0[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  rhs = $0[256,512]{1,0} fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "32"]}]
      }
    }
  }
  ROOT result = $1[128,512]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  hlo_text = absl::Substitute(
      hlo_text, primitive_util::LowercasePrimitiveTypeName(input_type),
      primitive_util::LowercasePrimitiveTypeName(result_type));

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              hlo_text, PRIMITIVE_TYPE_INVALID, HloOpcode::kDot,
                              /*use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc, fail_mode);
}

INSTANTIATE_TEST_SUITE_P(
    DotTestSuite, DotTypesTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kDot)),
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kDot)),
        ::testing::ValuesIn(AllDevicesToTest())),
    DotTypesTest::ParamToString);

TEST_F(DotTest, NonFusionRhs) {
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[128,256] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,256] parameter(0)
  p1 = $0[256,512] parameter(1)
  lhs = $0[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(lhs, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, NonFusionLhs) {
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,256] parameter(0)
  p1 = $0[256,512] parameter(1)
  rhs = $0[256,512] fusion(p1), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(p0, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, SingleBatchDim) {
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[16,128,256] parameter(0)
}

frhs {
  ROOT result = $0[16,256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[16,128,256] parameter(0)
  p1 = $0[16,256,512] parameter(1)
  lhs = $0[16,128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "16", "64"]}]
      }
    }
  }
  rhs = $0[16,256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "64", "32"]}]
      }
    }
  }
  ROOT result = $0[16,128,512] dot(lhs, rhs),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 16, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, MultipleNonContractingDimensions) {
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[16,128,256] parameter(0)
}

frhs {
  ROOT result = $0[16,256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[16,128,256] parameter(0)
  p1 = $0[16,256,512] parameter(1)
  lhs = $0[16,128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "16", "64"]}]
      }
    }
  }
  rhs = $0[16,256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["1", "64", "32"]}]
      }
    }
  }
  ROOT result = $0[16,128,16,512] dot(lhs, rhs),
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 16, 1, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, MultipleContractingDimensions) {
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[128,16,256] parameter(0)
}

frhs {
  ROOT result = $0[16,256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,16,256] parameter(0)
  lhs = $0[128,16,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "4", "64"]}]
      }
    }
  }
  p1 = $0[16,256,512] parameter(1)
  rhs = $0[16,256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "4", "32"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(lhs, rhs),
    lhs_contracting_dims={1, 2},
    rhs_contracting_dims={0, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, NonDefaultDimensionOrder_kmkn) {
  // Multiplying as [k, m] x [k, n] = [m, n].
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[256,128] parameter(0)
}

frhs {
  ROOT result = $0[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[256,128] parameter(0)
  p1 = $0[256,512] parameter(1)
  lhs = $0[256,128] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "16"]}]
      }
    }
  }
  rhs = $0[256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "32"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(lhs, rhs),
    lhs_contracting_dims={0},
    rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, NonDefaultDimensionOrder_mknk) {
  // Muliplying as [m, k] x [n, k] = [m, n].
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[128,256] parameter(0)
}

frhs {
  ROOT result = $0[512,256] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,256] parameter(0)
  p1 = $0[512,256] parameter(1)
  lhs = $0[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  rhs = $0[512,256] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32", "64"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(lhs, rhs),
    lhs_contracting_dims={1},
    rhs_contracting_dims={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32},
                 se::CudaComputeCapability::Ampere());
}

TEST_F(DotTest, SparsityConfiguration) {
  // Note that support rejects this HLO as u16 is not supported.
  const std::string kHloTestTemplate = R"(
flhs {
  ROOT result = $0[128,128] parameter(0)
}

frhs {
  ROOT result = $0[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,128] parameter(0)
  p1 = $0[256,512] parameter(1)
  lhs = $0[128,128] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  rhs = $0[256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "32"]}]
      }
    }
  }
  meta = u16[128,16] parameter(2)
  ROOT result = $0[128,512] dot(lhs, rhs, meta),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    sparsity=L.1@2:4
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32},
                 se::CudaComputeCapability::Ampere());
}

class DotPrecisionTest
    : public DotTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrecisionConfig::Precision,
                     PrecisionConfig::Precision, se::GpuComputeCapability>> {};

std::string OperandPrecisionTestName(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, PrecisionConfig::Precision,
                   PrecisionConfig::Precision, se::GpuComputeCapability>>&
        data) {
  auto [type, lhs_precision, rhs_precision, cc] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(type), "_",
                      PrecisionToString(lhs_precision), "_",
                      PrecisionToString(rhs_precision), "_",
                      ComputeCapabilityToString(cc));
}

TEST_P(DotPrecisionTest, OperandPrecision) {
  auto [data_type, lhs_precision, rhs_precision, cc] = GetParam();
  std::string hlo_text = absl::Substitute(
      R"(
flhs {
  ROOT result = $0[128,256] parameter(0)
}

frhs {
  ROOT result = $0[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,256] parameter(0)
  p1 = $0[256,512] parameter(1)
  lhs = $0[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  rhs = $0[256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "32"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(lhs, rhs),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    operand_precision={$1, $2}
}
)",
      primitive_util::LowercasePrimitiveTypeName(data_type),
      PrecisionToString(lhs_precision), PrecisionToString(rhs_precision));

  ExpectedFailMode fail_mode = ExpectedFailMode::kFail;
  if (absl::c_linear_search(std::vector{F8E5M2, F8E4M3FN, S8}, data_type)) {
    fail_mode = ExpectedFailMode::kFailOrCrash;
  }
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(
          hlo_text, PrimitiveType::PRIMITIVE_TYPE_INVALID, HloOpcode::kDot,
          /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc, fail_mode);
}

constexpr std::array kOperandPrecisions = {
    // All precisions except PACKED_NIBBLE.
    PrecisionConfig::DEFAULT,
    PrecisionConfig::HIGH,
    PrecisionConfig::HIGHEST,
};

INSTANTIATE_TEST_SUITE_P(
    DotPrecisionTestSuite, DotPrecisionTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kDot)),
        ::testing::ValuesIn(kOperandPrecisions),
        ::testing::ValuesIn(kOperandPrecisions),
        ::testing::ValuesIn(AllDevicesToTest())),
    OperandPrecisionTestName);

class DotPrecisionAlgorithmTest
    : public DotTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrecisionConfig::Algorithm,
                     se::GpuComputeCapability>> {};

std::string DotPrecisionAlgorithmTestName(
    const ::testing::TestParamInfo<std::tuple<
        PrimitiveType, PrecisionConfig::Algorithm, se::GpuComputeCapability>>&
        data) {
  auto [type, algorigthm, cc] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(type), "_",
                      AlgorithmToString(algorigthm), "_",
                      ComputeCapabilityToString(cc));
}

TEST_P(DotPrecisionAlgorithmTest, Algorithm) {
  auto [data_type, algorithm, cc] = GetParam();
  std::string hlo_text =
      absl::Substitute(R"(
flhs {
  ROOT result = $0[128,256] parameter(0)
}

frhs {
  ROOT result = $0[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = $0[128,256] parameter(0)
  p1 = $0[256,512] parameter(1)
  lhs = $0[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["16", "64"]}]
      }
    }
  }
  rhs = $0[256,512] fusion(p1), kind=kCustom, calls=frhs, backend_config={
    "fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
        "output_tiles":[{"sizes":["64", "32"]}]
      }
    }
  }
  ROOT result = $0[128,512] dot(lhs, rhs),
    lhs_contracting_dims={1},
    rhs_contracting_dims={0},
    algorithm=$1
}
)",
                       primitive_util::LowercasePrimitiveTypeName(data_type),
                       AlgorithmToString(algorithm));
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, F32, HloOpcode::kDot,
                                     /* use_nested_gemm_fusions=*/true));

  ExpectedFailMode fail_mode = ExpectedFailMode::kFail;
  if (absl::c_linear_search(std::vector{F8E5M2, F8E4M3FN, F8E4M3, S8},
                            data_type)) {
    fail_mode = ExpectedFailMode::kFailOrCrash;
  }
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc, fail_mode);
}

INSTANTIATE_TEST_SUITE_P(
    DotPrecisionTestSuite, DotPrecisionAlgorithmTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kDot)),
        ::testing::ValuesIn(AllPrecisionAlgorithms()),
        ::testing::ValuesIn(AllDevicesToTest())),
    DotPrecisionAlgorithmTestName);

class FusionKindsTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<absl::string_view, se::GpuComputeCapability>> {};

TEST_P(FusionKindsTest, OperandOfDot) {
  auto [kind, cc] = GetParam();
  const std::string hlo_text = absl::Substitute(
      R"(
flhs {
  ROOT result = f32[128,256] parameter(0)
}

frhs {
  ROOT result = f32[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = f32[128,256] parameter(0)
  p1 = f32[256,512] parameter(1)
  lhs = f32[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{"kind":"$0", "block_level_fusion_config":{
    "output_tiles":[{"sizes":["16", "64"]}]}}}
  rhs = f32[256,512]{1,0} fusion(p1), kind=kCustom, calls=frhs,
    backend_config={ "fusion_backend_config":{ "kind":"$0",
    "block_level_fusion_config": {"output_tiles":[{"sizes":["64", "32"]}]}}}
  ROOT result = f32[128,512]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)",
      kind);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, F32, HloOpcode::kFusion,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{16, 32}, cc);
}

std::string FusionKindsTestName(
    const ::testing::TestParamInfo<
        std::tuple<absl::string_view, se::GpuComputeCapability>>& data) {
  auto [kind, cc] = data.param;
  return absl::StrCat(kind, "_", ComputeCapabilityToString(cc));
}

TEST_P(FusionKindsTest, OperandOfConcatenate) {
  auto [kind, cc] = GetParam();
  const std::string hlo_text = absl::Substitute(
      R"(
nest0 {
  ROOT p0 = f32[128] parameter(0)
}

nest1 {
  ROOT p0 = f32[128] parameter(0)
}

ENTRY triton_computation {
  p0 = f32[128] parameter(0)
  p1 = f32[128] parameter(1)

  fusion0 = f32[128] fusion(p0), kind=kCustom, calls=nest0, backend_config={
    "fusion_backend_config":{"kind":"$0",
    "block_level_fusion_config":{"output_tiles":[{"sizes":["64"]}]}}}
  fusion1 = f32[128] fusion(p1), kind=kCustom, calls=nest1, backend_config={
    "fusion_backend_config":{"kind":"$0",
    "block_level_fusion_config":{"output_tiles":[{"sizes":["64"]}]}}}
  ROOT result = f32[256] concatenate(fusion0, fusion1), dimensions={0}
}
)",
      kind);
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, F32, HloOpcode::kFusion,
                                     /* use_nested_gemm_fusions=*/true));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{64}, cc);
}

std::vector<absl::string_view> FusionKindsForTest() {
  return {kTritonFusionKind, kTritonNestedGemmFusionKind, "__invalid"};
}

INSTANTIATE_TEST_SUITE_P(
    FusionTestSuite, FusionKindsTest,
    ::testing::Combine(::testing::ValuesIn(FusionKindsForTest()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    FusionKindsTestName);

using FusionTest = TritonSupportTest;

TEST_F(FusionTest, FusionComputationIsCheckedRecursively) {
  // We expect test for fail as `flhs` is not a supported computation as
  // fusion there is not an operand of a dot or a concatenate.
  absl::string_view hlo_text = R"(
identity {
  ROOT result = f32[128,256] parameter(0)
}

flhs {
  p0 = f32[128,256] parameter(0)
  ROOT result = f32[128,256] fusion(p0), kind=kCustom, calls=identity, backend_config={
    "fusion_backend_config":{"kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
    "output_tiles":[{"sizes":["16", "64"]}]}}}
}

frhs {
  ROOT result = f32[256,512] parameter(0)
}

ENTRY triton_computation {
  p0 = f32[128,256] parameter(0)
  p1 = f32[256,512] parameter(1)
  lhs = f32[128,256] fusion(p0), kind=kCustom, calls=flhs, backend_config={
    "fusion_backend_config":{"kind":"__triton_nested_gemm_fusion", "block_level_fusion_config":{
    "output_tiles":[{"sizes":["16", "64"]}]}}}
  rhs = f32[256,512]{1,0} fusion(p1), kind=kCustom, calls=frhs,
    backend_config={ "fusion_backend_config":{ "kind":"__triton_nested_gemm_fusion",
    "block_level_fusion_config": {"output_tiles":[{"sizes":["64", "32"]}]}}}
  ROOT result = f32[128,512]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, F32, HloOpcode::kFusion,
                                     /*use_nested_gemm_fusions=*/true));
  se::GpuComputeCapability cc = se::CudaComputeCapability::Ampere();
  ASSERT_FALSE(IsTritonSupportedInstruction(ti.Instruction(), cc));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{64, 32}, cc);
}

class BitcastConvertTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, se::GpuComputeCapability>> {
};

TEST_P(BitcastConvertTest, BitcastConvert) {
  auto [data_type_in, data_type_out, cc] = GetParam();

  if (primitive_util::IsComplexType(data_type_in) !=
      primitive_util::IsComplexType(data_type_out)) {
    GTEST_SKIP()
        << "BitcastConvert does not support complex <-> real conversion.";
  }

  std::string hlo_text;
  std::vector<int64_t> output_tile_sizes = {1, 32};

  const int bit_width_in = primitive_util::BitWidth(data_type_in);
  const int bit_width_out = primitive_util::BitWidth(data_type_out);
  const std::string data_type_in_str =
      primitive_util::LowercasePrimitiveTypeName(data_type_in);
  const std::string data_type_out_str =
      primitive_util::LowercasePrimitiveTypeName(data_type_out);

  if (bit_width_in == bit_width_out) {
    hlo_text = absl::Substitute(
        R"(
ENTRY triton_computation {
  parameter = $0[33,68] parameter(0)
  ROOT bc_convert = $1[33,68] bitcast-convert(parameter)
})",
        data_type_in_str, data_type_out_str);
  } else if (bit_width_in > bit_width_out) {
    hlo_text = absl::Substitute(
        R"(
ENTRY triton_computation {
  parameter = $0[33] parameter(0)
  ROOT bc_convert = $1[33, $2] bitcast-convert(parameter)
})",
        data_type_in_str, data_type_out_str, bit_width_in / bit_width_out);
  } else {  // bit_width_in < bit_width_out
    hlo_text = absl::Substitute(
        R"(
ENTRY triton_computation {
  parameter = $0[33, $1] parameter(0)
  ROOT bc_convert = $2[33] bitcast-convert(parameter)
})",
        data_type_in_str, bit_width_out / bit_width_in, data_type_out_str);
    output_tile_sizes = {1};
  }

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type_in,
                                     HloOpcode::kBitcastConvert));

  RunSupportTest(std::move(ti), output_tile_sizes, cc);
}

INSTANTIATE_TEST_SUITE_P(
    BitcastConvertSuite, BitcastConvertTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTwoTypesAndDeviceToString);

using AddDependencyTest = TritonSupportTestWithDeviceParam;

TEST_P(AddDependencyTest, AddDependency) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  param = f32[10] parameter(0)
  token0 = token[] after-all()
  ROOT add_dep = f32[10] add-dependency(param, token0)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kAddDependency));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(AddDependencySuite, AddDependencyTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using AfterAllTest = TritonSupportTestWithDeviceParam;

TEST_P(AfterAllTest, AfterAll) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  token0 = token[] after-all()
  token1 = token[] after-all()
  ROOT token2 = token[] after-all(token0, token1)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kAfterAll));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

INSTANTIATE_TEST_SUITE_P(AfterAllSuite, AfterAllTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using TupleTest = TritonSupportTestWithDeviceParam;

TEST_P(TupleTest, Tuple) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  p0 = f32[10] parameter(0)
  p1 = s32[5] parameter(1)
  ROOT tuple_op = (f32[10], s32[5]) tuple(p0, p1)
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kTuple));
  RunSupportTestMultipleOutputTiles(std::move(ti),
                                    /*output_tile_sizes=*/{{1}, {1}}, cc);
}

INSTANTIATE_TEST_SUITE_P(TupleSuite, TupleTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using GetTupleElementTest = TritonSupportTestWithDeviceParam;

TEST_P(GetTupleElementTest, GetTupleElement) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  tuple_op = (f32[10], s32[5]) parameter(0)
  ROOT gte = f32[10] get-tuple-element(tuple_op), index=0
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kGetTupleElement));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(GetTupleElementSuite, GetTupleElementTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

using CustomCallTest = TritonSupportTestWithDeviceParam;

TEST_P(CustomCallTest, CustomCall) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter = f32[10] parameter(0)
  ROOT custom_call_op = f32[10] custom-call(parameter), custom_call_target="SomeTarget"
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kCustomCall));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1}, cc);
}

INSTANTIATE_TEST_SUITE_P(CustomCallSuite, CustomCallTest,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

class CholeskyTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          // The bool parameter is used to parametrize the lower=?.
          std::tuple<PrimitiveType, se::GpuComputeCapability, bool>> {};

TEST_P(CholeskyTest, Cholesky) {
  auto [data_type, cc, lower] = GetParam();

  const std::string kHloTestTemplate = absl::Substitute(
      R"(
ENTRY triton_computation {
  parameter = $0[4,4] parameter(0)
  ROOT cholesky_op = $0[4,4] cholesky(parameter), lower=$1
})",
      primitive_util::LowercasePrimitiveTypeName(data_type), lower);

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kCholesky));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{2, 2}, cc);
}

std::string CholeskyTestName(
    const ::testing::TestParamInfo<
        std::tuple<PrimitiveType, se::GpuComputeCapability, bool>>& data) {
  const auto [data_type, cc, lower] = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_", ComputeCapabilityToString(cc), "_", lower);
}

INSTANTIATE_TEST_SUITE_P(
    CholeskySuite, CholeskyTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kCholesky)),
        ::testing::ValuesIn(AllDevicesToTest()), ::testing::Bool()),
    CholeskyTestName);

class TriangularSolveParamTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, se::GpuComputeCapability, bool /*lower*/,
                     bool /*unit_diagonal*/,
                     TriangularSolveOptions::Transpose /*transpose_a*/>> {
 public:
  static std::string ParamToString(
      const ::testing::TestParamInfo<ParamType>& info) {
    auto [data_type, cc, lower, unit_diagonal, transpose_a] = info.param;
    return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                        "_", ComputeCapabilityToString(cc), "_lower", lower,
                        "_unitdiag", unit_diagonal, "_",
                        TriangularSolveOptions::Transpose_Name(transpose_a));
  }
};

TEST_P(TriangularSolveParamTest, TriangularSolveLeftSideTrue) {
  auto [data_type, cc, lower, unit_diagonal, transpose_a] = GetParam();

  const std::string hlo_text = absl::Substitute(
      R"(
ENTRY triton_computation {
  a = $0[2,4,4] parameter(0)
  b = $0[2,4,2] parameter(1)
  ROOT result = $0[2,4,2] triangular-solve(a, b),
    left_side=true, lower=$1, unit_diagonal=$2, transpose_a=$3
})",
      primitive_util::LowercasePrimitiveTypeName(data_type),
      lower ? "true" : "false", unit_diagonal ? "true" : "false",
      TriangularSolveOptions::Transpose_Name(transpose_a));

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type,
                                     HloOpcode::kTriangularSolve));
  RunSupportTest(std::move(ti), {1, 2, 1}, cc);
}

TEST_P(TriangularSolveParamTest, TriangularSolveLeftSideFalse) {
  auto [data_type, cc, lower, unit_diagonal, transpose_a] = GetParam();

  const std::string hlo_text = absl::Substitute(
      R"(
ENTRY triton_computation {
  a = $0[2,4,4] parameter(0)
  b = $0[2,2,4] parameter(1)
  ROOT result = $0[2,2,4] triangular-solve(a, b),
    left_side=false, lower=$1, unit_diagonal=$2, transpose_a=$3
})",
      primitive_util::LowercasePrimitiveTypeName(data_type),
      lower ? "true" : "false", unit_diagonal ? "true" : "false",
      TriangularSolveOptions::Transpose_Name(transpose_a));

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type,
                                     HloOpcode::kTriangularSolve));
  RunSupportTest(std::move(ti), {1, 1, 2}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    TriangularSolveSuite, TriangularSolveParamTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kTriangularSolve)),
        ::testing::ValuesIn(AllDevicesToTest()),
        ::testing::Bool(),  // lower
        ::testing::Bool(),  // unit_diagonal
        ::testing::ValuesIn({TriangularSolveOptions::NO_TRANSPOSE,
                             TriangularSolveOptions::TRANSPOSE,
                             TriangularSolveOptions::ADJOINT})),
    TriangularSolveParamTest::ParamToString);

class FftTest : public TritonSupportTest,
                public ::testing::WithParamInterface<
                    std::tuple<PrimitiveType, se::GpuComputeCapability>> {};

TEST_P(FftTest, FFT) {
  auto [data_type, cc] = GetParam();

  const std::string hlo_text = R"(
ENTRY triton_computation {
  parameter = $0[16,16] parameter(0)
  ROOT fft_op = $0[16,16] fft(parameter), fft_type=FFT, fft_length={16}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type, HloOpcode::kFft));

  RunSupportTest(std::move(ti), {4, 4}, cc);
}

TEST_P(FftTest, IFFT) {
  auto [data_type, cc] = GetParam();

  const std::string hlo_text = R"(
ENTRY triton_computation {
  parameter = $0[16,16] parameter(0)
  ROOT fft_op = $0[16,16] fft(parameter), fft_type=IFFT, fft_length={16}
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type, HloOpcode::kFft));

  RunSupportTest(std::move(ti), {4, 4}, cc);
}

TEST_P(FftTest, RFFT) {
  auto [data_type, cc] = GetParam();
  const std::string complex_data_type_str =
      primitive_util::LowercasePrimitiveTypeName(data_type);
  // Real type matching the complex type for real -> complex conversion.
  const std::string real_data_type_str =
      primitive_util::LowercasePrimitiveTypeName(
          primitive_util::ComplexComponentType(data_type));

  const std::string hlo_text = absl::Substitute(
      R"(
ENTRY triton_computation {
  parameter = $0[16,16,32] parameter(0)
  ROOT fft_op = $1[16,16,17] fft(parameter), fft_type=RFFT, fft_length={16,32}
})",
      real_data_type_str, complex_data_type_str);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type, HloOpcode::kFft));

  RunSupportTest(std::move(ti), {4, 4, 4}, cc);
}

TEST_P(FftTest, IRFFT) {
  auto [data_type, cc] = GetParam();
  const std::string complex_data_type_str =
      primitive_util::LowercasePrimitiveTypeName(data_type);
  // Real type matching the complex type for complex -> real conversion.
  const std::string real_data_type_str =
      primitive_util::LowercasePrimitiveTypeName(
          primitive_util::ComplexComponentType(data_type));

  const std::string hlo_text = absl::Substitute(
      R"(
ENTRY triton_computation {
  parameter = $0[16,16,32,33] parameter(0)
  ROOT fft_op = $1[16,16,32,64] fft(parameter), fft_type=IRFFT, fft_length={16,32,64}
})",
      complex_data_type_str, real_data_type_str);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, data_type, HloOpcode::kFft));

  RunSupportTest(std::move(ti), {4, 4, 4, 4}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    FftTestSuite, FftTest,
    // FFT takes a complex type either as input, output or both. When there is a
    // complex <-> real conversion, the real type can be directly inferred from
    // the complex type (C64 <-> F32, C128 <-> F64).
    ::testing::Combine(::testing::ValuesIn({C64, C128}),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using CopyStartDoneTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(CopyStartDoneTest, CopyStartDone) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  parameter = $0[10,10,10] parameter(0)
  cp_start = ($0[10,10,10], $0[10,10,10], u32[]) copy-start(parameter)
  ROOT cp_done = $0[10,10,10] copy-done(cp_start)
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_start,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kCopyStart));
  RunSupportTest(std::move(ti_start), /*output_tile_sizes=*/{1, 1, 1}, cc);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_done,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kCopyDone));
  RunSupportTest(std::move(ti_done), /*output_tile_sizes=*/{1, 1, 1}, cc);
}
constexpr std::array kTestedOpsCopy = {HloOpcode::kCopyStart,
                                       HloOpcode::kCopyDone};

INSTANTIATE_TEST_SUITE_P(
    CopyStartDoneSuite, CopyStartDoneTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using InfeedTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(InfeedTest, Infeed) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  token0 = token[] after-all()
  ROOT infeed_op = ($0[10], token[]) infeed(token0)
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTestTemplate, data_type, HloOpcode::kInfeed));
  RunSupportTestMultipleOutputTiles(std::move(ti),
                                    /*output_tile_sizes=*/{{1}, {}}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    InfeedSuite, InfeedTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using OutfeedTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(OutfeedTest, Outfeed) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  data = $0[10] parameter(0)
  token0 = token[] after-all()
  ROOT outfeed_op = token[] outfeed(data, token0)
})";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kOutfeed));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{}, cc);
}

using MapTest = TritonSupportTestWithTypeAndOpcodeAndDeviceParam;

TEST_P(MapTest, Map) {
  auto [data_type, opcode, cc] = GetParam();

  // Note: the test is only relevant for datatypes supported by kAdd op.
  const std::string kHloTestTemplate = R"(
map_computation {
  p = $0[] parameter(0)
  ROOT add = $0[] add(p, p)
}

ENTRY triton_computation {
  parameter = $0[10, 20] parameter(0)
  ROOT map_op = $0[10, 20] map(parameter), dimensions={0, 1}, to_apply=map_computation
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type, opcode));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{4, 8}, cc);
}

INSTANTIATE_TEST_SUITE_P(MapSuite, MapTest,
                         AllTestCombinationsForOpcodes({HloOpcode::kMap}),
                         TritonSupportTestTypeAndOpcodeAndDeviceToString);

INSTANTIATE_TEST_SUITE_P(
    OutfeedSuite, OutfeedTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using RecvOpsTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(RecvOpsTest, RecvAndRecvDone) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
  ENTRY triton_computation {
    token0 = token[] after-all()
    recv_op = ($0[10,20], u32[], token[]) recv(token0), channel_id=15
    recv_done_op = ($0[10,20], token[]) recv-done(recv_op), channel_id=15
    ROOT result = $0[10,20] get-tuple-element(recv_done_op), index=0
  })";
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti_recv,
                          ParseTemplateAndGetInstruction(
                              kHloTestTemplate, data_type, HloOpcode::kRecv));
  RunSupportTest(std::move(ti_recv), /*output_tile_sizes=*/{1, 1}, cc);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_recv_done,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kRecvDone));
  RunSupportTest(std::move(ti_recv_done), /*output_tile_sizes=*/{1, 1}, cc);
}

constexpr std::array kTestedOpsRecv = {HloOpcode::kRecv, HloOpcode::kRecvDone};

INSTANTIATE_TEST_SUITE_P(
    RecvOpsSuite, RecvOpsTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

using SendOpsTest = TritonSupportTestWithTypeAndDeviceParam;

TEST_P(SendOpsTest, SendAndSendDone) {
  auto [data_type, cc] = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  data = $0[10] parameter(0)
  token0 = token[] after-all()
  send_op = ($0[10], u32[], token[]) send(data, token0), channel_id=77
  ROOT send_done_op = token[] send-done(send_op), channel_id=77
})";

  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti_send,
                          ParseTemplateAndGetInstruction(
                              kHloTestTemplate, data_type, HloOpcode::kSend));
  RunSupportTest(std::move(ti_send), /*output_tile_sizes=*/{}, cc);

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti_send_done,
      ParseTemplateAndGetInstruction(kHloTestTemplate, data_type,
                                     HloOpcode::kSendDone));
  RunSupportTest(std::move(ti_send_done), /*output_tile_sizes=*/{}, cc);
}

constexpr std::array kTestedOpsSend = {HloOpcode::kSend, HloOpcode::kSendDone};

INSTANTIATE_TEST_SUITE_P(
    SendOpsSuite, SendOpsTest,
    ::testing::Combine(::testing::ValuesIn(AllXlaDataTypes()),
                       ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTypeAndDeviceToString);

class StochasticConvertTest
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, se::GpuComputeCapability>> {
};

TEST_P(StochasticConvertTest, StochasticConvert) {
  auto [operand_type, new_element_type, cc] = GetParam();

  PrimitiveType random_type = primitive_util::UnsignedIntegralTypeForBitWidth(
      primitive_util::BitWidth(operand_type));

  ASSERT_NE(random_type, PRIMITIVE_TYPE_INVALID)
      << "Could not determine a valid random_type for operand_type: "
      << PrimitiveType_Name(operand_type);

  const std::string hlo_text = absl::Substitute(
      R"(
ENTRY triton_computation {
  operand = $0[33,68] parameter(0)
  random = $1[33,68] parameter(1)
  ROOT result = $2[33,68] stochastic-convert(operand, random)
})",
      primitive_util::LowercasePrimitiveTypeName(operand_type),
      primitive_util::LowercasePrimitiveTypeName(random_type),
      primitive_util::LowercasePrimitiveTypeName(new_element_type));

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(hlo_text, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kStochasticConvert));

  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 32}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    StochasticConvertTestSuite, StochasticConvertTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(
            HloOpcode::kStochasticConvert)),     // Operand type.
        ::testing::ValuesIn(AllXlaDataTypes()),  // New element type.
        ::testing::ValuesIn(AllDevicesToTest())),
    TritonSupportTestTwoTypesAndDeviceToString);

using TopKTest = TritonSupportTestWithTypeAndDeviceAndBoolParam;

std::string ParamToStringTopK(
    const ::testing::TestParamInfo<TopKTest::ParamType>& info) {
  auto [data_type, cc, largest] = info.param;
  return absl::StrCat(PrimitiveType_Name(data_type), "_",
                      ComputeCapabilityToString(cc), "_",
                      largest ? "largest" : "smallest");
}

TEST_P(TopKTest, TopK) {
  auto [data_type, cc, largest] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(
      R"(
ENTRY triton_computation {
  operand = $$0[11,33,77] parameter(0)
  ROOT topk_op = ($$0[11,33,10], s32[11,33,10]) topk(operand), k=10, largest=$0
})",
      largest);
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti,
                          ParseTemplateAndGetInstruction(
                              kHloTestTemplate, data_type, HloOpcode::kTopK));
  RunSupportTestMultipleOutputTiles(
      std::move(ti),
      /*output_tile_sizes=*/{{2, 2, 1}, {2, 2, 1}}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    TopKSuite, TopKTest,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kTopK)),
        ::testing::ValuesIn(AllDevicesToTest()),
        ::testing::Bool()),  // largest?
    ParamToStringTopK);

class ConvolutionTestFullParametrization
    : public TritonSupportTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrecisionConfig::Precision,
                     PrecisionConfig::Precision, se::GpuComputeCapability>> {};

// Test a basic convolution (NHWC layout)
TEST_P(ConvolutionTestFullParametrization, IsTritonSupportedConvNHWC) {
  auto [data_type, input_precision, kernel_precision, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(
      R"(
ENTRY triton_computation {
  input = $0[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = $0[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = $0[1,5,6,3] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f,
    operand_precision={$1, $2}
})",
      primitive_util::LowercasePrimitiveTypeName(data_type),
      PrecisionToString(input_precision), PrecisionToString(kernel_precision));
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 1}, cc);
}

// Test a convolution with NCHW layout
TEST_P(ConvolutionTestFullParametrization, IsTritonSupportedConvNCHW) {
  auto [data_type, input_precision, kernel_precision, cc] = GetParam();
  const std::string kHloTestTemplate = absl::Substitute(
      R"(
ENTRY triton_computation {
  input = $0[1,2,5,6] parameter(0)  // N=1, C_in=2, H=5, W=6
  kernel = $0[3,2,3,3] parameter(1) // C_out=3, C_in=2, H=3, W=3
  ROOT conv = $0[1,3,5,6] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_oi01->bf01,
    operand_precision={$1, $2}
  })",
      primitive_util::LowercasePrimitiveTypeName(data_type),
      PrecisionToString(input_precision), PrecisionToString(kernel_precision));
  TF_ASSERT_OK_AND_ASSIGN(TestedInstruction ti, ParseTemplateAndGetInstruction(
                                                    kHloTestTemplate, data_type,
                                                    HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 1, 2, 2}, cc);
}

INSTANTIATE_TEST_SUITE_P(
    ConvolutionTestSuiteFull, ConvolutionTestFullParametrization,
    ::testing::Combine(
        ::testing::ValuesIn(AllOpSupportedTypes(HloOpcode::kConvolution)),
        ::testing::ValuesIn(kOperandPrecisions),
        ::testing::ValuesIn(kOperandPrecisions),
        ::testing::ValuesIn(AllDevicesToTest())),
    OperandPrecisionTestName);

using ConvolutionTestCcOnly = TritonSupportTestWithDeviceParam;

// Test a convolution with stride > 1
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvStrided) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = f16[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = f16[1,2,2,3] convolution(input, kernel),
    window={size=3x3 stride=2x2}, dim_labels=b01f_01io->b01f
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 1, 1, 1}, cc);
}

// Test a convolution with kernel dilation > 1
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvDilated) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = f16[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = f16[1,1,2,3] convolution(input, kernel),
    window={size=3x3 rhs_dilate=2x2}, dim_labels=b01f_01io->b01f
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 1, 1, 1}, cc);
}

// Test a depthwise convolution
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvDepthwise) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)    // N=1, H=5, W=6, C_in=1
  kernel = f16[3,3,1,2] parameter(1)   // H=3, W=3, C_in=1, C_out=2
  ROOT conv = f16[1,5,6,2] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f,
    feature_group_count=2
})";

  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 1}, cc);
}

// Test a convolution with batch_group_count > 1
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvBatchGrouped) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[4,5,6,2] parameter(0)  // N=4, H=5, W=6, C_in=2
  kernel = f16[3,3,2,6] parameter(1) // H=3, W=3, C_in=2, C_out=6
  ROOT conv = f16[2,5,6,6] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f,
    batch_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 2}, cc);
}

// Test a convolution with lhs_dilation > 1
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvLhsDilation) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = f16[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = f16[1,7,9,3] convolution(input, kernel),
    window={size=3x3 lhs_dilate=2x2}, dim_labels=b01f_01io->b01f
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 1}, cc);
}

// Test a convolution with asymmetric padding
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvAsymmetricPadding) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = f16[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = f16[1,5,7,3] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_2}, dim_labels=b01f_01io->b01f
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 1}, cc);
}

// Test a general grouped convolution (1 < feature_group_count < Cin)
TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvGeneralGrouped) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,4] parameter(0)  // N=1, H=5, W=6, C_in=4
  kernel = f16[3,3,2,6] parameter(1) // H=3, W=3, C_in_per_group=2, C_out_total=6
  ROOT conv = f16[1,5,6,6] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f,
    feature_group_count=2
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 2}, cc);
}

// Test a convolution where kernel spatial dims are larger than input spatial
// dims, requiring padding.
TEST_P(ConvolutionTestCcOnly,
       IsTritonSupportedConvKernelLargerThanInputPadded) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,2,2,2] parameter(0)  // N=1, H=2, W=2, C_in=2
  kernel = f16[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = f16[1,2,2,3] convolution(input, kernel),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 1, 1, 1}, cc);
}

TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvDifferentWindowPadding) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = f16[3,3,2,3] parameter(1) // H=3, W=3, C_in=2, C_out=3
  ROOT conv = f16[1,5,6,3] convolution(input, kernel),
    window={size=3x3 pad=2_0x0_2}, dim_labels=b01f_01io->b01f
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 1}, cc);
}

TEST_P(ConvolutionTestCcOnly, IsTritonSupportedConvDifferentWindowSize) {
  auto cc = GetParam();
  const std::string kHloTestTemplate = R"(
ENTRY triton_computation {
  input = f16[1,5,6,2] parameter(0)  // N=1, H=5, W=6, C_in=2
  kernel = f16[2,2,2,3] parameter(1) // H=2, W=2, C_in=2, C_out=3
  ROOT conv = f16[1,5,6,3] convolution(input, kernel),
    window={size=2x2 pad=1_0x0_1}, dim_labels=b01f_01io->b01f
})";
  TF_ASSERT_OK_AND_ASSIGN(
      TestedInstruction ti,
      ParseTemplateAndGetInstruction(kHloTestTemplate, PRIMITIVE_TYPE_INVALID,
                                     HloOpcode::kConvolution));
  RunSupportTest(std::move(ti), /*output_tile_sizes=*/{1, 2, 2, 1}, cc);
}

INSTANTIATE_TEST_SUITE_P(ConvolutionTestSuiteCcOnly, ConvolutionTestCcOnly,
                         ::testing::ValuesIn(AllDevicesToTest()),
                         TritonSupportTestDeviceToString);

constexpr std::array kUnsupportedOps = {
    // clang-format off
    // go/keep-sorted start
    HloOpcode::kDynamicReshape,
    HloOpcode::kDynamicUpdateSlice,
    HloOpcode::kGather,
    HloOpcode::kRaggedDot,
    HloOpcode::kReduceWindow,
    HloOpcode::kScatter,
    HloOpcode::kSelectAndScatter,
    HloOpcode::kSetDimensionSize,
    HloOpcode::kSort,
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
  ret.insert(kTestedOpsConcatenate.begin(), kTestedOpsConcatenate.end());
  ret.insert(kTestedOpsTranspose.begin(), kTestedOpsTranspose.end());
  ret.insert(kTestedOpsCollectives.begin(), kTestedOpsCollectives.end());
  ret.insert(kTestedOpsBroadcast.begin(), kTestedOpsBroadcast.end());
  ret.insert(kTestedOpsParameter.begin(), kTestedOpsParameter.end());
  ret.insert(kTestedOpsConstant.begin(), kTestedOpsConstant.end());
  ret.insert(kTestedOpsIota.begin(), kTestedOpsIota.end());
  ret.insert(kTestedOpsPad.begin(), kTestedOpsPad.end());
  ret.insert(kTestedOpsRng.begin(), kTestedOpsRng.end());
  ret.insert(kTestedOpsCopy.begin(), kTestedOpsCopy.end());
  ret.insert(kTestedOpsRecv.begin(), kTestedOpsRecv.end());
  ret.insert(kTestedOpsSend.begin(), kTestedOpsSend.end());

  // go/keep-sorted start
  ret.emplace(HloOpcode::kAddDependency);
  ret.emplace(HloOpcode::kAfterAll);
  ret.emplace(HloOpcode::kBatchNormGrad);
  ret.emplace(HloOpcode::kBatchNormInference);
  ret.emplace(HloOpcode::kBatchNormTraining);
  ret.emplace(HloOpcode::kBitcastConvert);
  ret.emplace(HloOpcode::kCall);
  ret.emplace(HloOpcode::kCholesky);
  ret.emplace(HloOpcode::kComplex);
  ret.emplace(HloOpcode::kConditional);
  ret.emplace(HloOpcode::kConvolution);
  ret.emplace(HloOpcode::kCustomCall);
  ret.emplace(HloOpcode::kDomain);
  ret.emplace(HloOpcode::kDot);
  ret.emplace(HloOpcode::kDynamicSlice);  // TODO(b/417172838): add tests.
  ret.emplace(HloOpcode::kFft);
  ret.emplace(HloOpcode::kFusion);
  ret.emplace(HloOpcode::kGetDimensionSize);
  ret.emplace(HloOpcode::kGetTupleElement);
  ret.emplace(HloOpcode::kInfeed);
  ret.emplace(HloOpcode::kMap);
  ret.emplace(HloOpcode::kOutfeed);
  ret.emplace(HloOpcode::kReverse);
  ret.emplace(HloOpcode::kRngBitGenerator);
  ret.emplace(HloOpcode::kRngGetAndUpdateState);
  ret.emplace(HloOpcode::kStochasticConvert);
  ret.emplace(HloOpcode::kTopK);
  ret.emplace(HloOpcode::kTriangularSolve);
  ret.emplace(HloOpcode::kTuple);
  ret.emplace(HloOpcode::kWhile);
  // go/keep-sorted end
  ret.insert(kUnsupportedOps.begin(), kUnsupportedOps.end());

  return ret;
}

TEST(OpCoverage, UnsupportedOpcodes) {
  for (HloOpcode opcode : kUnsupportedOps) {
    EXPECT_TRUE(internal::IsTritonUnsupportedOpcode(opcode))
        << "Opcode `" << HloOpcodeString(opcode)
        << "` is not expected to be supported.";
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

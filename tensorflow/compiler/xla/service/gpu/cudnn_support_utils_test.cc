/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_support_utils.h"

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/dynamic_parameter_binding.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class CudnnSupportUtilsTest : public HloTestBase {
 public:
  // Gets the custom call with `target` from the `module`. Expects that there is
  // one and only one matching call.
  StatusOr<HloCustomCallInstruction*> GetCustomCall(
      xla::VerifiedHloModule* module, absl::string_view target) {
    HloCustomCallInstruction* call = nullptr;
    for (HloComputation* comp : module->MakeNonfusionComputations()) {
      for (HloInstruction* inst : comp->instructions()) {
        if (inst->IsCustomCall(target)) {
          VLOG(1) << inst->ToString();
          if (call != nullptr) {
            return tsl::errors::FailedPrecondition(
                "Found more than one custom call.");
          }
          call = Cast<HloCustomCallInstruction>(inst);
        }
      }
    }
    if (call == nullptr) {
      return tsl::errors::FailedPrecondition(
          "Did not find any matching custom call.");
    }
    return call;
  }
};

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedIntegerConvolutionCheckVectorSize) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[8,10,10,128] parameter(0)
    filter = s8[2,2,128,128] parameter(1)
    ROOT result = (s8[8,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();

  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(conv,
                          GetCustomCall(module.get(), "__cudnn$convForward"));

  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(true));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(true));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 7),
              IsOkAndHolds(false));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 1),
              IsOkAndHolds(false));  // 1 is not considered a vector size
}

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedIntegerConvolutionCheckComputeCapability) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[8,10,10,128] parameter(0)
    filter = s8[2,2,128,128] parameter(1)
    ROOT result = (s8[8,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();

  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(conv,
                          GetCustomCall(module.get(), "__cudnn$convForward"));

  // cc6.1 allows for int8x4
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({6, 0}, *conv, 4),
              IsOkAndHolds(false));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({6, 1}, *conv, 4),
              IsOkAndHolds(true));

  // cc7.5+ allows for int8x32
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 4}, *conv, 32),
              IsOkAndHolds(false));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(true));
}

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedIntegerConvolutionCheckKind) {
  auto moduleFwd = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,10,10,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    ROOT result = (s8[32,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                       .value();

  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(
      conv, GetCustomCall(moduleFwd.get(), "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(true));

  auto moduleBwdFilter = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,41] parameter(0)
    output = f16[10,20,30,40] parameter(1)
    result = (f16[2,2,41,40], u8[0]) custom-call(input, output),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardFilter"
    ROOT gte = f16[2,2,41,40] get-tuple-element(result), index=0
  })")
                             .value();

  TF_ASSERT_OK_AND_ASSIGN(
      conv, GetCustomCall(moduleBwdFilter.get(), "__cudnn$convBackwardFilter"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));

  auto moduleBwdInput = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    output = f16[10,20,30,40] parameter(0)
    filter = f16[2,2,41,40] parameter(1)
    result = (f16[10,20,30,41], u8[0]) custom-call(output, filter),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardInput"
    ROOT gte = f16[10,20,30,41] get-tuple-element(result), index=0
  })")
                            .value();

  TF_ASSERT_OK_AND_ASSIGN(
      conv, GetCustomCall(moduleBwdInput.get(), "__cudnn$convBackwardInput"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));
}

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedVectorizedIntegerConvolutionCheckTypes) {
  auto moduleS8InOut = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,10,10,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    ROOT result = (s8[32,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                           .value();
  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(
      conv, GetCustomCall(moduleS8InOut.get(), "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(true));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(true));

  auto moduleS8InF32Out = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,10,10,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    ROOT result = (f32[32,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                              .value();
  TF_ASSERT_OK_AND_ASSIGN(
      conv, GetCustomCall(moduleS8InF32Out.get(), "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(true));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));  // imma output must also be int8_t

  auto moduleF32InF32Out = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f32[32,10,10,64] parameter(0)
    filter = f32[2,2,64,128] parameter(1)
    ROOT result = (f32[32,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                               .value();
  TF_ASSERT_OK_AND_ASSIGN(
      conv, GetCustomCall(moduleF32InF32Out.get(), "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(false));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));
}

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedVectorizedIntegerConvolutionCheckDims) {
  // This 3d conv should be rejected
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,10,10,10,64] parameter(0)
    filter = s8[2,2,2,64,128] parameter(1)
    ROOT result = (s8[32,10,10,10,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b012f_012io->b012f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(conv,
                          GetCustomCall(module.get(), "__cudnn$convForward"));

  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(false));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));
}

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedVectorizedIntegerConvolutionCheckDilation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,10,10,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    ROOT result = (s8[32,20,20,128], u8[0]) custom-call(input, filter),
                  window={size=2x2 rhs_dilate=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(conv,
                          GetCustomCall(module.get(), "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(false));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));
}

TEST_F(CudnnSupportUtilsTest,
       CudnnSupportsOptimizedVectorizedIntegerConvolutionCheckAlgo1Dims) {
  auto moduleFilterCoversInput = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,2,2,64] parameter(0)
    filter = s8[3,3,64,128] parameter(1)
    ROOT result = (s8[32,2,2,128], u8[0]) custom-call(input, filter),
                  window={size=3x3}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                                     .value();
  HloCustomCallInstruction* conv;
  TF_ASSERT_OK_AND_ASSIGN(conv, GetCustomCall(moduleFilterCoversInput.get(),
                                              "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(true));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(false));

  auto moduleFilterAlmostCoversInput = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[32,3,3,64] parameter(0)
    filter = s8[3,3,64,128] parameter(1)
    ROOT result = (s8[32,3,3,128], u8[0]) custom-call(input, filter),
                  window={size=3x3}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                                           .value();
  TF_ASSERT_OK_AND_ASSIGN(conv,
                          GetCustomCall(moduleFilterAlmostCoversInput.get(),
                                        "__cudnn$convForward"));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 4),
              IsOkAndHolds(true));
  EXPECT_THAT(CudnnSupportsOptimizedIntegerConvolution({7, 5}, *conv, 32),
              IsOkAndHolds(true));
}

// Verify that convolutions with any filter dimension configuration are
// correctly reordered.
class ReorderFilterRank4Test : public ::testing::TestWithParam<std::string> {};

TEST_P(ReorderFilterRank4Test, InferTransposeRank4) {
  auto input_dims = GetParam();

  // Create convolution instruction from the test input dimensions.
  size_t dI = input_dims.find('i');
  size_t dO = input_dims.find('o');
  size_t dH = input_dims.find('0');
  size_t dW = input_dims.find('1');

  ConvolutionDimensionNumbers dnums;
  dnums.set_kernel_input_feature_dimension(dI);
  dnums.set_kernel_output_feature_dimension(dO);
  dnums.add_kernel_spatial_dimensions(dH);
  dnums.add_kernel_spatial_dimensions(dW);

  int64_t shape_dims[4] = {0, 0, 0, 0};
  shape_dims[dI] = 224;
  shape_dims[dO] = 96;
  shape_dims[dH] = 5;
  shape_dims[dW] = 3;

  Shape shape = ShapeUtil::MakeShape(U8, absl::MakeSpan(shape_dims));
  auto input = HloInstruction::CreateParameter(0, shape, "input");
  auto filter = HloInstruction::CreateParameter(1, shape, "filter");

  // Infer transpose from convolution filter.
  TF_ASSERT_OK_AND_ASSIGN(CudnnReorderTransposeConfig inferred_config,
                          CudnnInferTransposeForFilterReordering(shape, dnums));

  // Result shape: [O, I/32, H, W, 32]
  EXPECT_THAT(inferred_config.result_shape.dimensions(),
              ::testing::ElementsAre(96, 7, 5, 3, 32));

  // Transpose shape after the permutation: [I/32, H, W, O/8, 2, 8, 4, 4]
  Shape reshaped = ShapeUtil::PermuteDimensions(
      inferred_config.permutation, inferred_config.transpose_shape);
  EXPECT_THAT(reshaped.dimensions(),
              ::testing::ElementsAre(7, 5, 3, 12, 2, 8, 4, 4));

  // Additionally verify that 4-size dimensions are not swapped around.
  // O(4) should precede O(2), and I(4) should follow I(8).
  EXPECT_EQ(inferred_config.permutation[6], inferred_config.permutation[4] - 1);
  EXPECT_EQ(inferred_config.permutation[7], inferred_config.permutation[5] + 1);
}

std::vector<std::string> GeneratePermutations(std::string input_dims) {
  std::sort(input_dims.begin(), input_dims.end());
  std::vector<std::string> permutations;
  do {
    permutations.push_back(input_dims);
  } while (std::next_permutation(input_dims.begin(), input_dims.end()));
  return permutations;
}

INSTANTIATE_TEST_SUITE_P(ReorderTestSuite, ReorderFilterRank4Test,
                         ::testing::ValuesIn(GeneratePermutations("01io")));

// Verify that already vectorized convolutions (I/4) with any filter dimension
// configuration are correctly reordered.
class ReorderFilterRank5Test
    : public ::testing::TestWithParam<std::tuple<std::string, int>> {};

TEST_P(ReorderFilterRank5Test, InferTransposeRank5) {
  auto [input_dims, vsize] = GetParam();

  // Create convolution instruction from the test input dimensions.
  size_t dI = input_dims.find('i');
  size_t dO = input_dims.find('o');
  size_t dH = input_dims.find('0');
  size_t dW = input_dims.find('1');

  ConvolutionDimensionNumbers dnums;
  dnums.set_kernel_input_feature_dimension(dI);
  dnums.set_kernel_output_feature_dimension(dO);
  dnums.add_kernel_spatial_dimensions(dH);
  dnums.add_kernel_spatial_dimensions(dW);

  int64_t shape_dims[5] = {vsize, vsize, vsize, vsize, vsize};
  shape_dims[dI] = 224 / vsize;
  shape_dims[dO] = 96;
  shape_dims[dH] = 5;
  shape_dims[dW] = 3;

  Shape shape = ShapeUtil::MakeShape(U8, absl::MakeSpan(shape_dims));
  auto input = HloInstruction::CreateParameter(0, shape, "input");
  auto filter = HloInstruction::CreateParameter(1, shape, "filter");

  // Infer transpose from convolution filter.
  TF_ASSERT_OK_AND_ASSIGN(CudnnReorderTransposeConfig inferred_config,
                          CudnnInferTransposeForFilterReordering(shape, dnums));

  // Result shape: [O, I/32, H, W, 32]
  EXPECT_THAT(inferred_config.result_shape.dimensions(),
              ::testing::ElementsAre(96, 7, 5, 3, 32));

  // Transpose shape after the permutation: [I/32, H, W, O/8, 2, 8, 4, 4]
  Shape reshaped = ShapeUtil::PermuteDimensions(
      inferred_config.permutation, inferred_config.transpose_shape);
  EXPECT_THAT(reshaped.dimensions(),
              ::testing::ElementsAre(7, 5, 3, 12, 2, 8, 4, 4));

  // Additionally verify that 4-size dimensions are not swapped around.
  // O(4) should precede O(2), and I(4) correctness is implied.
  EXPECT_EQ(inferred_config.permutation[6], inferred_config.permutation[4] - 1);
}

INSTANTIATE_TEST_SUITE_P(
    ReorderTestSuite, ReorderFilterRank5Test,
    ::testing::Combine(::testing::ValuesIn(GeneratePermutations("01?io")),
                       ::testing::Values(4, 32)));

// Verify that convolutions with bias are correctly reordered.
class ReorderBiasTest : public ::testing::Test {};

TEST_F(ReorderBiasTest, InferTranspose) {
  Shape shape = ShapeUtil::MakeShape(U8, {96});
  auto bias = HloInstruction::CreateParameter(2, shape, "bias");

  Shape unused = ShapeUtil::MakeNil();
  auto input = HloInstruction::CreateParameter(0, unused, "input");
  auto filter = HloInstruction::CreateParameter(1, unused, "filter");

  // Infer transpose from convolution filter.
  TF_ASSERT_OK_AND_ASSIGN(CudnnReorderTransposeConfig inferred_config,
                          CudnnInferTransposeForBiasReordering(shape));

  // Transpose shape after the permutation: [O/32, 2, 4, 4]
  Shape reshaped = ShapeUtil::PermuteDimensions(
      inferred_config.permutation, inferred_config.transpose_shape);
  EXPECT_THAT(reshaped.dimensions(), ::testing::ElementsAre(3, 2, 4, 4));

  // Additionally verify that 4-size dimensions are not swapped around.
  EXPECT_EQ(inferred_config.permutation[2], 1);
  EXPECT_EQ(inferred_config.permutation[3], 3);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

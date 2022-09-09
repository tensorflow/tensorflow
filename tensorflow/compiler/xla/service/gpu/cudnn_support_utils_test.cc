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

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/service/dynamic_parameter_binding.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
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
            return tensorflow::errors::FailedPrecondition(
                "Found more than one custom call.");
          }
          call = Cast<HloCustomCallInstruction>(inst);
        }
      }
    }
    if (call == nullptr) {
      return tensorflow::errors::FailedPrecondition(
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

}  // namespace
}  // namespace gpu
}  // namespace xla

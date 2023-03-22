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

#include "tensorflow/compiler/xla/service/gpu/cudnn_vectorize_convolutions.h"

#include <vector>

#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class CudnnVectorizeConvolutionsTest : public HloTestBase {
 protected:
  // Runs this pass and some cleanup to make pattern-matching easier.
  StatusOr<bool> Run(std::pair<int, int> compute_capability,
                     HloModule* module) {
    CudnnVectorizeConvolutions pass(
        se::CudaComputeCapability{compute_capability.first,
                                  compute_capability.second},
        se::dnn::VersionInfo(8, 3, 0));
    TF_ASSIGN_OR_RETURN(bool changed, RunHloPass(&pass, module));

    CallInliner inliner;
    TF_RETURN_IF_ERROR(RunHloPass(&inliner, module).status());

    return changed;
  }
};

TEST_F(CudnnVectorizeConvolutionsTest, VectorizeTo4) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,40] parameter(0)
    filter = s8[2,2,40,44] parameter(1)
    ROOT result = (s8[10,20,30,44], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward",
                  backend_config="{bar: 0}"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, {kCudnnConvForwardCallTarget},
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 20, 30, 10, 4}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {2, 2, 10, 4, 44})))
                         .WithShape(S8, {10, 20, 30, 11, 4})),
          m::Op())));

  EXPECT_EQ(conv->raw_backend_config_string(), "{bar: 0}");

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.input_feature_dimension(), 3);

  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 0);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 1);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 2);
  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 4);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.output_feature_dimension(), 3);
}

TEST_F(CudnnVectorizeConvolutionsTest, NoVectorizeTo4UnsupportedFilterType) {
  // This test checks that the vectorize pass correctly calls
  // CudnnSupportsOptimizedIntegerConvolution() which should reject this
  // convolution because its filter type is f32.
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,40] parameter(0)
    filter = f32[2,2,40,44] parameter(1)
    ROOT result = (s8[10,20,30,44], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward",
                  backend_config="{bar: 0}"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnVectorizeConvolutionsTest, VectorizeTo4NCHW) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,48,20,30] parameter(0)
    filter = s8[48,44,2,2] parameter(1)
    ROOT result = (s8[10,44,20,30], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=bf01_io01->bf01,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, {kCudnnConvForwardCallTarget},
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 12, 4, 20, 30}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {12, 4, 44, 2, 2})))
                         .WithShape(S8, {10, 11, 4, 20, 30})),
          m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_feature_dimension(), 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 4);

  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 4);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_feature_dimension(), 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 4);
}

TEST_F(CudnnVectorizeConvolutionsTest, IncrementAllDnums) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[16,16,16,16] parameter(0)
    filter = s8[16,16,3,3] parameter(1)
    ROOT result = (s8[16,16,16,16], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=fb01_i01o->fb01,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, {kCudnnConvForwardCallTarget},
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {4, 4, 16, 16, 16}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {4, 4, 16, 3, 3})))
                         .WithShape(S8, {4, 4, 16, 16, 16})),
          m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_feature_dimension(), 0);
  EXPECT_EQ(dnums.input_batch_dimension(), 2);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 4);

  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 3);
  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 4);

  EXPECT_EQ(dnums.output_feature_dimension(), 0);
  EXPECT_EQ(dnums.output_batch_dimension(), 2);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 4);
}

TEST_F(CudnnVectorizeConvolutionsTest, FilterDnums) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[1,20,9,9] parameter(0)
    filter = s8[3,3,20,32] parameter(1)
    ROOT result = (s8[1,32,9,9], u8[0]) custom-call(s8[1,20,9,9] input, s8[3,3,20,32] filter),
                  window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, {kCudnnConvForwardCallTarget},
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {1, 5, 4, 9, 9}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {3, 3, 5, 4, 32})))
                         .WithShape(S8, {1, 8, 4, 9, 9})),
          m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_feature_dimension(), 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 4);

  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 0);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 1);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 2);
  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 4);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_feature_dimension(), 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 3);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 4);
}

TEST_F(CudnnVectorizeConvolutionsTest, NoVectorizeTo4) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,41] parameter(0)
    filter = s8[2,2,41,44] parameter(1)
    ROOT result = (s8[10,20,30,44], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  CudnnVectorizeConvolutions pass(
      /*compute_capability=*/{7, 5},
      /*cudnn_version=*/se::dnn::VersionInfo{8, 3, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

// Don't vectorize int8_t -> int32_t into int8x4 or int8x32; this is not
// supported in cudnn.
TEST_F(CudnnVectorizeConvolutionsTest, NoVectorizeTo4IfOutputIsS32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,41] parameter(0)
    filter = s8[2,2,41,44] parameter(1)
    ROOT result = (s32[10,20,30,44], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

// Don't vectorize int8_t -> float into int8x4 or int8x32.  Vectorizing to
// int8x4 *is* allowed by cudnn, but we don't do it at the moment.
TEST_F(CudnnVectorizeConvolutionsTest, NoVectorizeTo4IfOutputIsF32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,41] parameter(0)
    filter = s8[2,2,41,44] parameter(1)
    ROOT result = (f32[10,20,30,44], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(CudnnVectorizeConvolutionsTest, VectorizeTo32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    ROOT result = (s8[10,20,30,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(
              m::GetTupleElement(
                  m::CustomCall(
                      &conv, {kCudnnConvForwardCallTarget},
                      m::Reshape(m::Parameter(0))
                          .WithShape(S8, {10, 20, 30, 2, 32}),
                      m::Reshape(
                          m::Transpose(
                              m::Reshape(m::Parameter(1))
                                  .WithShape(S8, {2, 2, 2, 8, 4, 16, 4, 2}))
                              .WithShape(S8, {2, 2, 2, 16, 2, 8, 4, 4})
                              .WithPredicate([](const HloInstruction* instr) {
                                return absl::c_equal(
                                    instr->dimensions(),
                                    std::vector<int64_t>{2, 0, 1, 5, 7, 3, 6,
                                                         4});
                              }))
                          .WithShape(S8, {128, 2, 2, 2, 32})))
                  .WithShape(S8, {10, 20, 30, 4, 32})),
          m::Op())));

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, BiasAndSideInput) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    bias = f32[128] parameter(2)
    side_input = s8[10,20,30,64] parameter(3)

    ROOT result = (s8[10,20,30,128], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(
              m::GetTupleElement(
                  m::CustomCall(
                      &conv, {kCudnnConvForwardCallTarget},
                      m::Reshape(m::Parameter(0))
                          .WithShape(S8, {10, 20, 30, 2, 32}),
                      m::Reshape(m::Transpose(m::Reshape(m::Parameter(1))))
                          .WithShape(S8, {128, 2, 2, 2, 32}),
                      m::Reshape(
                          m::Transpose(m::Reshape(m::Parameter(2))
                                           .WithShape(F32, {4, 4, 2, 4}))
                              .WithShape(F32, {4, 2, 4, 4})
                              .WithPredicate([](const HloInstruction* instr) {
                                return absl::c_equal(
                                    instr->dimensions(),
                                    std::vector<int64_t>{0, 2, 1, 3});
                              }))
                          .WithShape(F32, {128}),
                      m::Reshape(m::Parameter(3))
                          .WithShape(S8, {10, 20, 30, 2, 32})))
                  .WithShape(S8, {10, 20, 30, 4, 32})),
          m::Op())));

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, NoVectorizeTo32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    ROOT result = (s8[10,20,30,128], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 0}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, {kCudnnConvForwardCallTarget},
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 20, 30, 16, 4}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {2, 2, 16, 4, 128})))
                         .WithShape(S8, {10, 20, 30, 32, 4})),
          m::Op())));

  EXPECT_FALSE(conv->backend_config<CudnnConvBackendConfig>()
                   ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, Vectorize4To32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,16,4] parameter(0)
    filter = s8[3,5,16,192,4] parameter(1)
    bias = f32[64] parameter(2)
    side_input = s8[10,20,30,16,4] parameter(3)
    ROOT result = (s8[10,20,30,48,4], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=3x5}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  auto conv_pat =
      m::GetTupleElement(
          m::CustomCall(
              &conv, {kCudnnConvForwardCallTarget},
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(0))
                                          .WithShape(S8, {10, 20, 30, 2, 8, 4}))
                             .WithShape(S8, {10, 20, 30, 2, 8, 4}))
                  .WithShape(S8, {10, 20, 30, 2, 32}),
              m::Reshape(
                  m::Transpose(m::Reshape(m::Parameter(1))
                                   .WithShape(S8, {3, 5, 2, 8, 24, 4, 2, 4}))
                      .WithShape(S8, {2, 3, 5, 24, 2, 8, 4, 4})
                      .WithPredicate([](const HloInstruction* instr) {
                        return absl::c_equal(
                            instr->dimensions(),
                            std::vector<int64_t>{2, 0, 1, 4, 6, 3, 5, 7});
                      }))
                  .WithShape(S8, {192, 2, 3, 5, 32}),
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(2)))),
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(3))
                                          .WithShape(S8, {10, 20, 30, 2, 8, 4}))
                             .WithShape(S8, {10, 20, 30, 2, 8, 4}))
                  .WithShape(S8, {10, 20, 30, 2, 32})))
          .WithShape(S8, {10, 20, 30, 6, 32});
  ASSERT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::Transpose(m::Reshape(conv_pat).WithShape(
                                                    S8, {10, 20, 30, 6, 8, 4}))
                                       .WithShape(S8, {10, 20, 30, 6, 8, 4}))
                            .WithShape(S8, {10, 20, 30, 48, 4}),
                        m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.input_feature_dimension(), 3);

  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 1);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 3);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.output_feature_dimension(), 3);

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, Vectorize4To32NCHW) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,16,20,30,4] parameter(0)
    filter = s8[16,128,2,2,4] parameter(1)
    bias = f32[64] parameter(2)
    side_input = s8[10,16,20,30,4] parameter(3)
    ROOT result = (s8[10,32,20,30,4], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=2x2}, dim_labels=bf01_io01->bf01,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  auto conv_pat =
      m::GetTupleElement(
          m::CustomCall(
              &conv, {kCudnnConvForwardCallTarget},
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(0))
                                          .WithShape(S8, {10, 2, 8, 20, 30, 4}))
                             .WithShape(S8, {10, 2, 20, 30, 8, 4}))
                  .WithShape(S8, {10, 2, 20, 30, 32}),
              m::Reshape(
                  m::Transpose(m::Reshape(m::Parameter(1))
                                   .WithShape(S8, {2, 8, 16, 4, 2, 2, 2, 4}))
                      .WithShape(S8, {2, 2, 2, 16, 2, 8, 4, 4})
                      .WithPredicate([](const HloInstruction* instr) {
                        return absl::c_equal(
                            instr->dimensions(),
                            std::vector<int64_t>{0, 5, 6, 2, 4, 1, 3, 7});
                      }))
                  .WithShape(S8, {128, 2, 2, 2, 32}),
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(2)))),
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(3))
                                          .WithShape(S8, {10, 2, 8, 20, 30, 4}))
                             .WithShape(S8, {10, 2, 20, 30, 8, 4}))
                  .WithShape(S8, {10, 2, 20, 30, 32})))
          .WithShape(S8, {10, 4, 20, 30, 32});
  ASSERT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::Transpose(m::Reshape(conv_pat).WithShape(
                                                    S8, {10, 4, 20, 30, 8, 4}))
                                       .WithShape(S8, {10, 4, 8, 20, 30, 4}))
                            .WithShape(S8, {10, 32, 20, 30, 4}),
                        m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_feature_dimension(), 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 3);

  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 1);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 3);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_feature_dimension(), 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 3);

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, Vectorize4To32VectorDimFirst) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[4,10,20,30,16] parameter(0)
    filter = s8[4,3,5,16,192] parameter(1)
    bias = f32[64] parameter(2)
    side_input = s8[4,10,20,30,16] parameter(3)
    ROOT result = (s8[4,10,20,30,48], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=3x5}, dim_labels=?b01f_?01io->?b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  auto conv_pat =
      m::GetTupleElement(
          m::CustomCall(
              &conv, {kCudnnConvForwardCallTarget},
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(0))
                                          .WithShape(S8, {4, 10, 20, 30, 2, 8}))
                             .WithShape(S8, {8, 4, 10, 20, 30, 2}))
                  .WithShape(S8, {32, 10, 20, 30, 2}),
              m::Reshape(
                  m::Transpose(m::Reshape(m::Parameter(1))
                                   .WithShape(S8, {4, 3, 5, 2, 8, 24, 4, 2}))
                      .WithShape(S8, {2, 3, 5, 24, 2, 8, 4, 4})
                      .WithPredicate([](const HloInstruction* instr) {
                        return absl::c_equal(
                            instr->dimensions(),
                            std::vector<int64_t>{3, 1, 2, 5, 7, 4, 6, 0});
                      }))
                  .WithShape(S8, {192, 2, 3, 5, 32}),
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(2)))),
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(3))
                                          .WithShape(S8, {4, 10, 20, 30, 2, 8}))
                             .WithShape(S8, {8, 4, 10, 20, 30, 2}))
                  .WithShape(S8, {32, 10, 20, 30, 2})))
          .WithShape(S8, {32, 10, 20, 30, 6});
  ASSERT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::Transpose(m::Reshape(conv_pat).WithShape(
                                                    S8, {8, 4, 10, 20, 30, 6}))
                                       .WithShape(S8, {4, 10, 20, 30, 6, 8}))
                            .WithShape(S8, {4, 10, 20, 30, 48}),
                        m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();
  ASSERT_EQ(dnums.input_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.kernel_spatial_dimensions().size(), 2);
  ASSERT_EQ(dnums.output_spatial_dimensions().size(), 2);

  EXPECT_EQ(dnums.input_batch_dimension(), 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 3);
  EXPECT_EQ(dnums.input_feature_dimension(), 4);

  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 1);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 3);

  EXPECT_EQ(dnums.output_batch_dimension(), 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 3);
  EXPECT_EQ(dnums.output_feature_dimension(), 4);

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, NoVectorize4To32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,16,4] parameter(0)
    filter = s8[2,2,16,128,4] parameter(1)
    bias = f32[10] parameter(2)
    side_input = s8[10,20,30,16,4] parameter(3)
    ROOT result = (s8[10,20,30,32,4], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 0}, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(CudnnVectorizeConvolutionsTest, Vectorize16To32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,4,16] parameter(0)
    filter = s8[3,5,4,192,16] parameter(1)
    ROOT result = (s8[10,20,30,12,16], u8[0]) custom-call(input, filter),
                  window={size=3x5}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  auto filter_pat =
      m::Reshape(
          m::Transpose(
              m::Reshape(m::Parameter(1)).WithShape(S8, {3, 5, 2, 2, 192, 16}))
              .WithShape(S8, {3, 5, 2, 192, 2, 16}))
          .WithShape(S8, {3, 5, 2, 192, 32});
  auto conv_pat =
      m::GetTupleElement(
          m::CustomCall(
              &conv, {kCudnnConvForwardCallTarget},
              m::Reshape(
                  m::Transpose(m::Reshape(m::Parameter(0))
                                   .WithShape(S8, {10, 20, 30, 2, 2, 16}))
                      .WithShape(S8, {10, 20, 30, 2, 2, 16}))
                  .WithShape(S8, {10, 20, 30, 2, 32}),
              m::Reshape(
                  m::Transpose(m::Reshape(filter_pat)
                                   .WithShape(S8, {3, 5, 2, 24, 4, 2, 8, 4}))
                      .WithShape(S8, {2, 3, 5, 24, 2, 8, 4, 4}))
                  .WithShape(S8, {192, 2, 3, 5, 32})))
          .WithShape(S8, {10, 20, 30, 6, 32});
  ASSERT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::Transpose(m::Reshape(conv_pat).WithShape(
                                                    S8, {10, 20, 30, 6, 2, 16}))
                                       .WithShape(S8, {10, 20, 30, 6, 2, 16}))
                            .WithShape(S8, {10, 20, 30, 12, 16}),
                        m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.input_feature_dimension(), 3);

  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 1);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 3);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.output_feature_dimension(), 3);

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

TEST_F(CudnnVectorizeConvolutionsTest, VectorizeMixedTo32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,8,8] parameter(0)
    filter = s8[3,5,2,192,32] parameter(1)
    ROOT result = (s8[10,20,30,96,2], u8[0]) custom-call(input, filter),
                  window={size=3x5}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  TF_ASSERT_OK_AND_ASSIGN(bool changed, Run({7, 5}, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  auto conv_pat =
      m::GetTupleElement(
          m::CustomCall(
              &conv, {kCudnnConvForwardCallTarget},
              m::Reshape(m::Transpose(m::Reshape(m::Parameter(0))
                                          .WithShape(S8, {10, 20, 30, 2, 4, 8}))
                             .WithShape(S8, {10, 20, 30, 2, 4, 8}))
                  .WithShape(S8, {10, 20, 30, 2, 32}),
              m::Reshape(
                  m::Transpose(m::Reshape(m::Parameter(1))
                                   .WithShape(S8, {3, 5, 2, 24, 4, 2, 8, 4}))
                      .WithShape(S8, {2, 3, 5, 24, 2, 8, 4, 4}))
                  .WithShape(S8, {192, 2, 3, 5, 32})))
          .WithShape(S8, {10, 20, 30, 6, 32});
  ASSERT_THAT(root, GmockMatch(m::Tuple(
                        m::Reshape(m::Transpose(m::Reshape(conv_pat).WithShape(
                                                    S8, {10, 20, 30, 6, 16, 2}))
                                       .WithShape(S8, {10, 20, 30, 6, 16, 2}))
                            .WithShape(S8, {10, 20, 30, 96, 2}),
                        m::Op())));

  const ConvolutionDimensionNumbers& dnums =
      conv->convolution_dimension_numbers();

  EXPECT_EQ(dnums.input_batch_dimension(), 0);
  EXPECT_EQ(dnums.input_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.input_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.input_feature_dimension(), 3);

  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 0);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 1);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 2);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 3);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.output_feature_dimension(), 3);

  EXPECT_TRUE(conv->backend_config<CudnnConvBackendConfig>()
                  ->reordered_int8_nchw_vect());
}

}  // namespace
}  // namespace gpu
}  // namespace xla

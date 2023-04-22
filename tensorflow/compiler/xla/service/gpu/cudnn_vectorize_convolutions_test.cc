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

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class CudnnVectorizeConvolutionsTest : public HloTestBase {};

TEST_F(CudnnVectorizeConvolutionsTest, VectorizeTo4) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,40] parameter(0)
    filter = s8[2,2,40,44] parameter(1)
    ROOT result = (s8[10,20,30,44], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass(se::CudaComputeCapability{7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, kCudnnConvForwardCallTarget,
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 20, 30, 10, 4}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {2, 2, 10, 4, 44})))
                         .WithShape(S8, {10, 20, 30, 11, 4})),
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

  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 0);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 1);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 2);
  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 4);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.output_feature_dimension(), 3);
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, kCudnnConvForwardCallTarget,
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, kCudnnConvForwardCallTarget,
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

// Don't vectorize int8 -> int32 into int8x4 or int8x32; this is not supported
// in cudnn.
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));

  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

// Don't vectorize int8 -> float into int8x4 or int8x32.  Vectorizing to int8x4
// *is* allowed by cudnn, but we don't do it at the moment.
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));

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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, kCudnnConvForwardCallTarget,
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 20, 30, 2, 32}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {2, 2, 2, 32, 128})))
                         .WithShape(S8, {10, 20, 30, 4, 32})),
          m::Op())));
}

TEST_F(CudnnVectorizeConvolutionsTest, BiasAndSideInput) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,64] parameter(0)
    filter = s8[2,2,64,128] parameter(1)
    bias = f32[10] parameter(2)
    side_input = s8[10,20,30,64] parameter(3)

    ROOT result = (s8[10,20,30,128], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, kCudnnConvForwardCallTarget,
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 20, 30, 2, 32}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {2, 2, 2, 32, 128}),
                                       m::Parameter(2),
                                       m::Reshape(m::Parameter(3))
                                           .WithShape(S8, {10, 20, 30, 2, 32})))
                         .WithShape(S8, {10, 20, 30, 4, 32})),
          m::Op())));
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(m::GetTupleElement(
                         m::CustomCall(&conv, kCudnnConvForwardCallTarget,
                                       m::Reshape(m::Parameter(0))
                                           .WithShape(S8, {10, 20, 30, 16, 4}),
                                       m::Reshape(m::Parameter(1))
                                           .WithShape(S8, {2, 2, 16, 4, 128})))
                         .WithShape(S8, {10, 20, 30, 32, 4})),
          m::Op())));
}

TEST_F(CudnnVectorizeConvolutionsTest, Vectorize4To32) {
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 5});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  SCOPED_TRACE(module->ToString());
  auto* root = module->entry_computation()->root_instruction();

  const HloInstruction* conv = nullptr;
  ASSERT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Reshape(
              m::Transpose(m::Reshape(
                  m::GetTupleElement(
                      m::CustomCall(
                          &conv, kCudnnConvForwardCallTarget,
                          m::Reshape().WithShape(S8, {10, 20, 30, 2, 32}),
                          m::Reshape().WithShape(S8, {2, 2, 2, 128, 32}),
                          m::Parameter(2),
                          m::Reshape().WithShape(S8, {10, 20, 30, 2, 32})))
                      .WithShape(S8, {10, 20, 30, 4, 32}))))
              .WithShape(S8, {10, 20, 30, 32, 4}),
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

  EXPECT_EQ(dnums.kernel_spatial_dimensions()[0], 0);
  EXPECT_EQ(dnums.kernel_spatial_dimensions()[1], 1);
  EXPECT_EQ(dnums.kernel_input_feature_dimension(), 2);
  EXPECT_EQ(dnums.kernel_output_feature_dimension(), 3);

  EXPECT_EQ(dnums.output_batch_dimension(), 0);
  EXPECT_EQ(dnums.output_spatial_dimensions()[0], 1);
  EXPECT_EQ(dnums.output_spatial_dimensions()[1], 2);
  EXPECT_EQ(dnums.output_feature_dimension(), 3);
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
                    .ValueOrDie();
  CudnnVectorizeConvolutions pass({7, 0});
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

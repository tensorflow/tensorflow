/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_layout_assignment.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using LayoutAssignmentTest = HloTestBase;

TEST_F(LayoutAssignmentTest, Elementwise) {
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  Shape ashape_in_row_major(ashape);
  Shape ashape_in_col_major(ashape);
  *ashape_in_row_major.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *ashape_in_col_major.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Enumerate all possible combinations of layouts.
  for (const Shape& lhs_shape_with_layout :
       {ashape_in_row_major, ashape_in_col_major}) {
    for (const Shape& rhs_shape_with_layout :
         {ashape_in_row_major, ashape_in_col_major}) {
      for (const Shape& result_shape_with_layout :
           {ashape_in_row_major, ashape_in_col_major}) {
        // GpuLayoutAssignment should assign the same layout to "add" and its
        // two operands.
        auto builder = HloComputation::Builder(TestName());
        auto x = builder.AddInstruction(
            HloInstruction::CreateParameter(0, ashape, "x"));
        auto y = builder.AddInstruction(
            HloInstruction::CreateParameter(1, ashape, "y"));
        auto add = builder.AddInstruction(
            HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, x, y));
        auto module = CreateNewModule();
        HloComputation* computation =
            module->AddEntryComputation(builder.Build(add));

        ComputationLayout computation_layout(
            computation->ComputeProgramShape());
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(lhs_shape_with_layout);
        *computation_layout.mutable_parameter_layout(1) =
            ShapeLayout(rhs_shape_with_layout);
        *computation_layout.mutable_result_layout() =
            ShapeLayout(result_shape_with_layout);

        GpuLayoutAssignment layout_assignment(&computation_layout);
        EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

        for (const HloInstruction* operand : add->operands()) {
          EXPECT_TRUE(LayoutUtil::Equal(add->shape().layout(),
                                        operand->shape().layout()));
        }
      }
    }
  }
}

// Returns a list shapes with all the possible layouts of this shape, including
// a shape with no layout.
std::vector<Shape> AllLayoutsOf(const Shape& s) {
  std::vector<int64> layout_vec(s.dimensions_size());
  std::iota(layout_vec.begin(), layout_vec.end(), 0);

  std::vector<Shape> shapes;
  shapes.push_back(s);
  shapes.back().clear_layout();

  do {
    shapes.push_back(s);
    *shapes.back().mutable_layout() = LayoutUtil::MakeLayout(layout_vec);
  } while (std::next_permutation(layout_vec.begin(), layout_vec.end()));

  return shapes;
}

TEST_F(LayoutAssignmentTest, BatchNormInference) {
  const int64 kFeatureIndex = 1;

  // The shape of the data operand to BatchNormInference and of the output of
  // the BatchNormInference call.
  Shape shape = ShapeUtil::MakeShape(F32, {42, 12, 1, 100});

  // The shape of the scale, offset, mean, and variance inputs to
  // BatchNormTraining.  These are rank 1, with as many elements are in the
  // kFeatureIndex dim of shape.
  Shape aux_shape =
      ShapeUtil::MakeShape(F32, {shape.dimensions(kFeatureIndex)});

  for (const Shape& input_shape : AllLayoutsOf(shape)) {
    for (const Shape& result_shape : AllLayoutsOf(shape)) {
      SCOPED_TRACE(tensorflow::strings::StrCat(
          "input_shape=", ShapeUtil::HumanStringWithLayout(input_shape),
          ", result_shape=", ShapeUtil::HumanStringWithLayout(result_shape)));

      auto builder = HloComputation::Builder(TestName());
      auto* operand = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "operand"));
      auto* scale = builder.AddInstruction(
          HloInstruction::CreateParameter(1, aux_shape, "scale"));
      auto* offset = builder.AddInstruction(
          HloInstruction::CreateParameter(2, aux_shape, "offset"));
      auto* mean = builder.AddInstruction(
          HloInstruction::CreateParameter(3, aux_shape, "mean"));
      auto* variance = builder.AddInstruction(
          HloInstruction::CreateParameter(4, aux_shape, "variance"));

      auto* epsilon = builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<float>(1)));
      auto* feature_index =
          builder.AddInstruction(HloInstruction::CreateConstant(
              Literal::CreateR0<int64>(kFeatureIndex)));

      auto* batchnorm = builder.AddInstruction(HloInstruction::CreateCustomCall(
          shape,
          {operand, scale, offset, mean, variance, epsilon, feature_index},
          kCudnnBatchNormForwardInferenceCallTarget));

      auto module = CreateNewModule();
      HloComputation* computation =
          module->AddEntryComputation(builder.Build(batchnorm));

      ComputationLayout computation_layout(computation->ComputeProgramShape());

      if (input_shape.has_layout()) {
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(input_shape);
      }

      if (result_shape.has_layout()) {
        *computation_layout.mutable_result_layout() = ShapeLayout(result_shape);
      }

      GpuLayoutAssignment layout_assignment(&computation_layout);
      EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

      // The first operand to batchnorm should have the same layout as the
      // result.
      EXPECT_TRUE(LayoutUtil::Equal(batchnorm->operand(0)->shape().layout(),
                                    batchnorm->shape().layout()))
          << batchnorm->ToString();
    }
  }
}

TEST_F(LayoutAssignmentTest, BatchNormTraining) {
  const int64 kFeatureIndex = 1;

  // The shape of the data operand to BatchNormTraining.
  Shape shape = ShapeUtil::MakeShape(F32, {42, 12, 1, 100});

  // The shape of the offset and scale inputs to BatchNormTraining.  These are
  // rank 1, with as many elements are in the kFeatureIndex dim of shape.
  Shape offset_scale_shape =
      ShapeUtil::MakeShape(F32, {shape.dimensions(kFeatureIndex)});

  // Shape of the output of our BatchNormTraining op.
  Shape batchnorm_shape = ShapeUtil::MakeTupleShape(
      {shape, offset_scale_shape, offset_scale_shape});

  // Enumerate all combinations of shapes.
  for (const Shape& input_shape : AllLayoutsOf(shape)) {
    for (const Shape& result_shape : AllLayoutsOf(shape)) {
      SCOPED_TRACE(tensorflow::strings::StrCat(
          "input_shape=", ShapeUtil::HumanStringWithLayout(input_shape),
          ", result_shape=", ShapeUtil::HumanStringWithLayout(result_shape)));

      auto builder = HloComputation::Builder(TestName());
      auto* operand = builder.AddInstruction(
          HloInstruction::CreateParameter(0, shape, "operand"));
      auto* scale = builder.AddInstruction(
          HloInstruction::CreateParameter(1, offset_scale_shape, "scale"));
      auto* offset = builder.AddInstruction(
          HloInstruction::CreateParameter(2, offset_scale_shape, "offset"));

      auto* epsilon = builder.AddInstruction(
          HloInstruction::CreateConstant(Literal::CreateR0<float>(1)));
      auto* feature_index =
          builder.AddInstruction(HloInstruction::CreateConstant(
              Literal::CreateR0<int64>(kFeatureIndex)));

      auto* batchnorm = builder.AddInstruction(HloInstruction::CreateCustomCall(
          batchnorm_shape, {operand, scale, offset, epsilon, feature_index},
          kCudnnBatchNormForwardTrainingCallTarget));

      auto module = CreateNewModule();
      HloComputation* computation =
          module->AddEntryComputation(builder.Build(batchnorm));

      ComputationLayout computation_layout(computation->ComputeProgramShape());

      if (input_shape.has_layout()) {
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(input_shape);
      }

      if (result_shape.has_layout()) {
        *computation_layout.mutable_result_layout() =
            ShapeLayout(ShapeUtil::MakeTupleShape(
                {result_shape, offset_scale_shape, offset_scale_shape}));
      }

      GpuLayoutAssignment layout_assignment(&computation_layout);
      EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

      // The first operand to batchnorm should have the same layout as the
      // first element of the result tuple.
      EXPECT_TRUE(
          LayoutUtil::Equal(batchnorm->operand(0)->shape().layout(),
                            batchnorm->shape().tuple_shapes(0).layout()))
          << batchnorm->ToString();
    }
  }
}

TEST_F(LayoutAssignmentTest, BatchNormGrad) {
  const int64 kFeatureIndex = 1;

  // The shape of the data operand to BatchNormTraining.
  Shape shape = ShapeUtil::MakeShape(F32, {42, 12, 1, 100});

  // The shape of the scale, mean, and variance inputs to BatchNormGrad.  These
  // are rank 1, with as many elements are in the kFeatureIndex dim of shape.
  Shape scale_shape =
      ShapeUtil::MakeShape(F32, {shape.dimensions(kFeatureIndex)});

  // Shape of the output of our BatchNormGrad op.
  Shape batchnorm_shape =
      ShapeUtil::MakeTupleShape({shape, scale_shape, scale_shape});

  // Enumerate all combinations of shapes plus whether we're constraining param
  // 0 or param 4.
  for (const Shape& input_shape : AllLayoutsOf(shape)) {
    for (const Shape& result_shape : AllLayoutsOf(shape)) {
      for (int constrained_param_no : {0, 4}) {
        SCOPED_TRACE(tensorflow::strings::StrCat(
            "input_shape=", ShapeUtil::HumanStringWithLayout(input_shape),
            ", result_shape=", ShapeUtil::HumanStringWithLayout(result_shape)));

        auto builder = HloComputation::Builder(TestName());
        auto* operand = builder.AddInstruction(
            HloInstruction::CreateParameter(0, shape, "operand"));
        auto* scale = builder.AddInstruction(
            HloInstruction::CreateParameter(1, scale_shape, "scale"));
        auto* mean = builder.AddInstruction(
            HloInstruction::CreateParameter(2, scale_shape, "mean"));
        auto* var = builder.AddInstruction(
            HloInstruction::CreateParameter(3, scale_shape, "var"));
        auto* grad_offset = builder.AddInstruction(
            HloInstruction::CreateParameter(4, shape, "var"));

        auto* epsilon = builder.AddInstruction(
            HloInstruction::CreateConstant(Literal::CreateR0<float>(1)));
        auto* feature_index =
            builder.AddInstruction(HloInstruction::CreateConstant(
                Literal::CreateR0<int64>(kFeatureIndex)));

        auto* batchnorm =
            builder.AddInstruction(HloInstruction::CreateCustomCall(
                batchnorm_shape,
                {operand, scale, mean, var, grad_offset, epsilon,
                 feature_index},
                kCudnnBatchNormBackwardCallTarget));

        auto module = CreateNewModule();
        HloComputation* computation =
            module->AddEntryComputation(builder.Build(batchnorm));

        ComputationLayout computation_layout(
            computation->ComputeProgramShape());

        if (input_shape.has_layout()) {
          *computation_layout.mutable_parameter_layout(constrained_param_no) =
              ShapeLayout(input_shape);
        }

        if (result_shape.has_layout()) {
          *computation_layout.mutable_result_layout() =
              ShapeLayout(ShapeUtil::MakeTupleShape(
                  {result_shape, scale_shape, scale_shape}));
        }

        GpuLayoutAssignment layout_assignment(&computation_layout);
        EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

        // The first and fourth operands to the batchnorm call should have the
        // same layout as the first element of the result tuple.
        EXPECT_TRUE(
            LayoutUtil::Equal(batchnorm->operand(0)->shape().layout(),
                              batchnorm->shape().tuple_shapes(0).layout()))
            << batchnorm->ToString();
        EXPECT_TRUE(
            LayoutUtil::Equal(batchnorm->operand(4)->shape().layout(),
                              batchnorm->shape().tuple_shapes(0).layout()))
            << batchnorm->ToString();
      }
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla

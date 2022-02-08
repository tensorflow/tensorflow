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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_layout.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;

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
        auto module = CreateNewVerifiedModule();
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

        GpuLayoutAssignment layout_assignment(
            &computation_layout, backend().default_stream_executor());
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
  std::vector<int64_t> layout_vec(s.dimensions_size());
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

TEST_F(LayoutAssignmentTest, DotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[8,8,256,64]{3,1,2,0} parameter(0)
    p1 = f32[8,8,256,64]{3,1,2,0} parameter(1)
    ROOT dot.1330.10585 = f32[8,8,256,256]{3,2,1,0} dot(p0, p1),
      lhs_batch_dims={0,1}, lhs_contracting_dims={3},
      rhs_batch_dims={0,1}, rhs_contracting_dims={3}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());
  EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

  Shape expected_shape =
      ShapeUtil::MakeShapeWithLayout(F32, {8, 8, 256, 64}, {3, 2, 1, 0});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::ShapeWithLayout(expected_shape),
                      op::ShapeWithLayout(expected_shape)));
}

TEST_F(LayoutAssignmentTest, DotLayoutS8) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY int8_t {
    p0 = s8[1024,65536] parameter(0)
    p1 = s8[65536,65536] parameter(1)
    ROOT out = s32[1024,65536] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());
  EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

  Shape expected_shape_p0 =
      ShapeUtil::MakeShapeWithLayout(S8, {1024, 65536}, {1, 0});
  Shape expected_shape_p1 =
      ShapeUtil::MakeShapeWithLayout(S8, {65536, 65536}, {0, 1});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::ShapeWithLayout(expected_shape_p0),
                      op::ShapeWithLayout(expected_shape_p1)));
}

TEST_F(LayoutAssignmentTest, SortLayout) {
  const char* hlo_text = R"(
  HloModule SortLayout

  compare {
    p.0.lhs = f32[] parameter(0)
    p.0.rhs = f32[] parameter(1)
    p.1.lhs = f32[] parameter(2)
    p.1.rhs = f32[] parameter(3)
    ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  ENTRY sort {
    keys = f32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = f32[2,3]{1,0} parameter(0)
    transpose = f32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (f32[3,2]{1,0}, f32[3,2]{1,0}) sort(keys, transpose),
      dimensions={1}, to_apply=compare
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());
  EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

  Shape expected_shape = ShapeUtil::MakeShapeWithLayout(F32, {3, 2}, {1, 0});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sort(op::ShapeWithLayout(expected_shape),
                       op::ShapeWithLayout(expected_shape)));
}

TEST_F(LayoutAssignmentTest, FftLayout) {
  const char* hlo_text = R"(
  HloModule Fft_module

  ENTRY Fft {
    input = c64[8,32]{0,1} parameter(0)
    fft = c64[8,32] fft(input), fft_type=FFT, fft_length={32}
    ROOT transpose = c64[32,8] transpose(fft), dimensions={1,0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());
  EXPECT_TRUE(layout_assignment.Run(module.get()).ValueOrDie());

  Shape expected_shape = ShapeUtil::MakeShapeWithLayout(C64, {8, 32}, {1, 0});
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::Transpose(op::ShapeWithLayout(expected_shape))));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Copy(op::Transpose(op::Fft(op::ShapeWithLayout(expected_shape)))));
}

}  // namespace
}  // namespace gpu
}  // namespace xla

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/batch_dot_simplification.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

class BatchDotSimplificationTest : public HloTestBase {};

TEST_F(BatchDotSimplificationTest,
       ElideSingleDegenerateBatchDotDim_VectorVector) {
  const std::string hlo_text = R"(
HloModule BatchDot

main {
  a = f32[1,3] parameter(0)
  b = f32[1,3] parameter(1)
  ROOT dot = f32[1] dot(a, b), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(op::Dot(
                  op::Reshape(op::Parameter(0)), op::Reshape(op::Parameter(1)),
                  /*lhs_contracting_dim=*/0, /*rhs_contracting_dim=*/0)));
}

TEST_F(BatchDotSimplificationTest,
       ElideSingleDegenerateBatchDotDim_MatrixVector) {
  const std::string hlo_text = R"(
HloModule BatchDot

main {
  a = f32[1,9,3] parameter(0)
  b = f32[1,3] parameter(1)
  ROOT dot = f32[1,9] dot(a, b), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(op::Dot(
                  op::Reshape(op::Parameter(0)), op::Reshape(op::Parameter(1)),
                  /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/0)));
}

TEST_F(BatchDotSimplificationTest,
       ElideSingleDegenerateBatchDotDim_MatrixMatrix) {
  const std::string hlo_text = R"(
HloModule BatchDot

main {
  a = f32[1,9,3] parameter(0)
  b = f32[1,3,7] parameter(1)
  ROOT dot = f32[1,9,7] dot(a, b), lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(op::Dot(
                  op::Reshape(op::Parameter(0)), op::Reshape(op::Parameter(1)),
                  /*lhs_contracting_dim=*/1, /*rhs_contracting_dim=*/0)));
}

TEST_F(BatchDotSimplificationTest,
       ElideMultipleDegenerateBatchDotDims_VectorVector) {
  const std::string hlo_text = R"(
HloModule BatchDot

main {
  a = f32[9,1,7,1,3] parameter(0)
  b = f32[9,1,7,1,3] parameter(1)
  ROOT dot = f32[9,1,7,1] dot(a, b), lhs_batch_dims={0,1,2,3}, rhs_batch_dims={0,1,2,3}, lhs_contracting_dims={4}, rhs_contracting_dims={4}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(op::Dot(
                  op::Reshape(op::Parameter(0)), op::Reshape(op::Parameter(1)),
                  /*lhs_contracting_dim=*/2, /*rhs_contracting_dim=*/2)));
}

TEST_F(BatchDotSimplificationTest,
       ElideMultipleDegenerateBatchDotDims_VectorMatrix) {
  const std::string hlo_text = R"(
HloModule BatchDot

main {
  a = f32[9,1,7,1,3] parameter(0)
  b = f32[9,1,7,1,20,3] parameter(1)
  ROOT dot = f32[9,1,7,1,20] dot(a, b), lhs_batch_dims={0,1,2,3}, rhs_batch_dims={0,1,2,3}, lhs_contracting_dims={4}, rhs_contracting_dims={5}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(op::Dot(
                  op::Reshape(op::Parameter(0)), op::Reshape(op::Parameter(1)),
                  /*lhs_contracting_dim=*/2, /*rhs_contracting_dim=*/3)));
}

TEST_F(BatchDotSimplificationTest,
       ElideMultipleDegenerateBatchDotDims_MatrixMatrix) {
  const std::string hlo_text = R"(
HloModule BatchDot

main {
  a = f32[9,1,7,1,19,3] parameter(0)
  b = f32[9,1,7,1,3,20] parameter(1)
  ROOT dot = f32[9,1,7,1,19,20] dot(a, b), lhs_batch_dims={0,1,2,3}, rhs_batch_dims={0,1,2,3}, lhs_contracting_dims={5}, rhs_contracting_dims={4}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_TRUE(pass.Run(m.get()).ValueOrDie());

  HloInstruction* root = m->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(op::Dot(
                  op::Reshape(op::Parameter(0)), op::Reshape(op::Parameter(1)),
                  /*lhs_contracting_dim=*/3, /*rhs_contracting_dim=*/2)));
}

TEST_F(BatchDotSimplificationTest,
       ElideMultipleDegenerateBatchDotDimsNonContracting) {
  const char* hlo_text = R"(
HloModule BatchDot

main {
  a = f32[1,101] parameter(0)
  b = f32[1,101] parameter(1)
  ROOT dot = f32[1,101,101] dot(a,b), lhs_batch_dims={0},
                                      lhs_contracting_dims={},
                                      rhs_batch_dims={0},
                                      rhs_contracting_dims={}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_FALSE(pass.Run(m.get()).ValueOrDie());
}

TEST_F(BatchDotSimplificationTest,
       ElideMultipleDegenerateBatchDotDimsMultipleContracting) {
  const char* hlo_text = R"(
HloModule BatchDot

main {
  lhs = f32[1,5,17,10,13] parameter(0)
  rhs = f32[1,9,10,13,6,5] parameter(1)
  ROOT dot = f32[10,1,17,9,6] dot(lhs,rhs), lhs_batch_dims={3,0},
                                            rhs_batch_dims={2,0},
                                            lhs_contracting_dims={1,4},
                                            rhs_contracting_dims={5,3}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  BatchDotSimplification pass;
  ASSERT_FALSE(pass.Run(m.get()).ValueOrDie());
}

}  // namespace
}  // namespace xla

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

#include "tensorflow/compiler/xla/service/indexed_array_analysis.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {
namespace {
class IndexedArrayAnalysisTest : public HloVerifiedTestBase {
 protected:
  void AssertArrayForRootExpressionIs(const string& hlo_text,
                                      const string& root_expression) {
    AssertArrayForRootExpressionIsImpl(hlo_text, root_expression,
                                       /*print_constants=*/false);
  }

  void AssertArrayWithConstantsForRootExpressionIs(
      const string& hlo_text, const string& root_expression) {
    AssertArrayForRootExpressionIsImpl(hlo_text, root_expression,
                                       /*print_constants=*/true);
  }

 private:
  void AssertArrayForRootExpressionIsImpl(const string& hlo_text,
                                          const string& root_expression,
                                          bool print_constants) {
    IndexedArrayAnalysis indexed_tensor_analysis;
    ParseAndVerifyModule(hlo_text);

    TF_ASSERT_OK_AND_ASSIGN(
        IndexedArrayAnalysis::Array* const array_result,
        indexed_tensor_analysis.GetArrayFor(
            module().entry_computation()->root_instruction()));
    string string_result =
        indexed_tensor_analysis.ToString(array_result, print_constants);
    LOG(INFO) << string_result;
    ASSERT_EQ(string_result, root_expression);
  }
};

TEST_F(IndexedArrayAnalysisTest, SimpleOneToOneGather) {
  string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[5] parameter(1)
  ROOT gather = s32[5,3] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,3}
}
)";

  AssertArrayForRootExpressionIs(hlo_text,
                                 "(scalar-indexed %operand %indices 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, SimpleOneToOneConstantGather) {
  string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] constant(s32[3,3]{{1,2,3},{1,2,3},{1,2,3}})
  indices = s32[5] parameter(0)
  ROOT gather = s32[5,3] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,3}
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text, "(scalar-indexed-const (constant s32[3,3]) %indices 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_OneToOne) {
  string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] constant(s32[3,3]{{1,2,3},{1,2,3},{1,2,3}})
  indices_a = s32[5] parameter(0)
  indices_b = s32[2] parameter(1)
  gather_a = s32[5,3] gather(operand, indices_a),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,3}
  ROOT gather_b = s32[2,3] gather(gather_a, indices_b),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,3}
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed-const (constant s32[3,3]) (scalar-indexed %indices_a "
      "%indices_b 0->[0]) 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_ManyToOneWithOneToOne) {
  string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,2] parameter(0)
  indices_a = s32[5,7] parameter(1)
  indices_b = s32[2] parameter(2)
  gather_a = s32[5,3,7] gather(operand, indices_a),
      output_window_dims={1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=2,
      window_bounds={3,1}
  ROOT gather_b = s32[5,3,2] gather(gather_a, indices_b),
      output_window_dims={0,1},
      elided_window_dims={2},
      gather_dims_to_operand_dims={2},
      index_vector_dim=1,
      window_bounds={5,3,1}
}
)";

  AssertArrayForRootExpressionIs(hlo_text,
                                 "(scalar-indexed %operand (scalar-indexed "
                                 "%indices_a %indices_b 1->[1]) 1->[0,2])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_OneToOneWithManyToOne) {
  string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,6] parameter(0)
  indices_a = s32[2] parameter(1)
  indices_b = s32[5,7] parameter(2)
  gather_a = s32[2,6] gather(operand, indices_a),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,6}
  ROOT gather_b = s32[5,6,7] gather(gather_a, indices_b),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=2,
      window_bounds={1,6}
}
)";

  AssertArrayForRootExpressionIs(hlo_text,
                                 "(scalar-indexed %operand (scalar-indexed "
                                 "%indices_a %indices_b 0->[0,1]) 0->[0,2])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_ManyToOneWithManyToOne) {
  string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,2] parameter(0)
  indices_a = s32[5,7] parameter(1)
  indices_b = s32[4,8] parameter(2)
  gather_a = s32[5,3,7] gather(operand, indices_a),
      output_window_dims={1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=2,
      window_bounds={3,1}
  ROOT gather_b = s32[4,5,3,8] gather(gather_a, indices_b),
      output_window_dims={1,2},
      elided_window_dims={2},
      gather_dims_to_operand_dims={2},
      index_vector_dim=2,
      window_bounds={5,3,1}
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed %operand (scalar-indexed %indices_a %indices_b "
      "1->[0,2]) 1->[0,1,3])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather0) {
  string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,2,3,4},{1,2,3,4}})
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT reshape = s32[5,2,2] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text, "(scalar-indexed-const (constant s32[3,2,2]) %indices 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather1) {
  string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,2,3,4},{1,2,3,4}})
  indices = s32[5,7] parameter(0)
  gather = s32[5,4,7] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=2,
      window_bounds={1,4}
  ROOT reshape = s32[5,2,2,7] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed-const (constant s32[3,2,2]) %indices 0->[0,3])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather2) {
  string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,2,6] constant(s32[3,2,6]{
      {{1,2,3,4,5,6},{1,2,3,4,5,6}},
      {{1,2,3,4,5,6},{1,2,3,4,5,6}},
      {{1,2,3,4,5,6},{1,2,3,4,5,6}}})
  indices = s32[5,7] parameter(0)
  gather = s32[5,2,6,7] gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=2,
      window_bounds={1,2,6}
  ROOT reshape = s32[5,3,4,7] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed-const (constant s32[3,3,4]) %indices 0->[0,3])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGatherNegative0) {
  string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,2,3,4},{1,2,3,4}})
  indices = s32[5,6] parameter(0)
  gather = s32[5,4,6] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=2,
      window_bounds={1,4}
  ROOT reshape = s32[5,2,2,2,3] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%reshape");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGatherNegative1) {
  string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,5,2] constant(s32[3,5,2]{
      {{1,2},{3,4},{5,6},{7,8},{9,10}},
      {{1,2},{3,4},{5,6},{7,8},{9,10}},
      {{1,2},{3,4},{5,6},{7,8},{9,10}}})
  indices = s32[7] parameter(0)
  gather = s32[3,2,7] gather(operand, indices),
      output_window_dims={0,1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=1,
      window_bounds={3,1,2}
  ROOT reshape = s32[6,7] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%reshape");
}

TEST_F(IndexedArrayAnalysisTest, UnaryOpOfGather) {
  string hlo_text = R"(
HloModule UnaryOpOfGather

ENTRY main {
  operand = f32[3,4] constant(f32[3,4]{{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  indices = s32[5] parameter(0)
  gather = f32[5,4] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT tanh = f32[5,4] tanh(gather)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, 1 + R"(
(scalar-indexed-const (constant f32[3,4] f32[3,4] {
  { 0.761594176, 0.964027584, 0.995054781, 0.999329329 },
  { 0.761594176, 0.995054781, 0.964027584, 0.999329329 },
  { 0.999329329, 0.995054781, 0.964027584, 0.761594176 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, AddBroadcastedScalarWithGather) {
  string hlo_text = R"(
HloModule AddBroadcastedScalarWithGather

ENTRY main {
  gather_operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant = s32[] constant(5)
  constant_broadcasted = s32[5,4] broadcast(constant), dimensions={}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT add = s32[5,4] add(gather, constant_broadcasted)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, 1 + R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { 6, 7, 8, 9 },
  { 6, 8, 7, 9 },
  { 9, 8, 7, 6 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest,
       SubtractBroadcastedScalarWithGather_GatherIsLhs) {
  string hlo_text = R"(
HloModule SubtractBroadcastedScalarWithGather

ENTRY main {
  gather_operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant = s32[] constant(5)
  constant_broadcasted = s32[5,4] broadcast(constant), dimensions={}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT sub = s32[5,4] subtract(gather, constant_broadcasted)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, 1 + R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { -4, -3, -2, -1 },
  { -4, -2, -3, -1 },
  { -1, -2, -3, -4 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest,
       SubtractBroadcastedScalarWithGather_GatherIsRhs) {
  string hlo_text = R"(
HloModule SubtractBroadcastedScalarWithGather

ENTRY main {
  gather_operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant = s32[] constant(5)
  constant_broadcasted = s32[5,4] broadcast(constant), dimensions={}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT sub = s32[5,4] subtract(constant_broadcasted, gather)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, 1 + R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { 4, 3, 2, 1 },
  { 4, 2, 3, 1 },
  { 1, 2, 3, 4 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, AddBroadcastedVectorWithGather) {
  string hlo_text = R"(
HloModule AddBroadcastedVectorWithGather

ENTRY main {
  gather_operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant_vect = s32[4] constant({10,11,12,13})
  constant_broadcasted = s32[5,4] broadcast(constant_vect), dimensions={1}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT add = s32[5,4] add(gather, constant_broadcasted)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, 1 + R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { 11, 13, 15, 17 },
  { 11, 14, 14, 17 },
  { 14, 14, 14, 14 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, AddBroadcastedVectorWithGather_Negative) {
  string hlo_text = R"(
HloModule AddBroadcastedVectorWithGather

ENTRY main {
  gather_operand = s32[3,4] constant(s32[3,4]{{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant_vect = s32[5] constant({10,11,12,13,14})
  constant_broadcasted = s32[5,4] broadcast(constant_vect), dimensions={0}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1,4}
  ROOT add = s32[5,4] add(gather, constant_broadcasted)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%add");
}

TEST_F(IndexedArrayAnalysisTest, RegularUnaryOp) {
  string hlo_text = R"(
HloModule RegularUnaryOp

ENTRY main {
  input = f32[100] parameter(0)
  ROOT tanh = f32[100] tanh(input)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%tanh");
}

TEST_F(IndexedArrayAnalysisTest, RegularBinaryOp) {
  string hlo_text = R"(
HloModule RegularUnaryOp

ENTRY main {
  input0 = f32[100] parameter(0)
  input1 = f32[100] parameter(1)
  ROOT add = f32[100] add(input0, input1)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%add");
}

}  // namespace
}  // namespace xla

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
    IndexedArrayAnalysis indexed_tensor_analysis;
    ParseAndVerifyModule(hlo_text);

    string result =
        indexed_tensor_analysis.ToString(indexed_tensor_analysis.GetArrayFor(
            module().entry_computation()->root_instruction()));
    LOG(INFO) << result;
    ASSERT_EQ(result, root_expression);
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
}  // namespace
}  // namespace xla

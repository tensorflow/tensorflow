/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/hlo/analysis/indexed_array_analysis.h"

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
class IndexedArrayAnalysisTest : public HloHardwareIndependentTestBase {
 protected:
  void AssertArrayForRootExpressionIs(const std::string& hlo_text,
                                      const std::string& root_expression) {
    AssertArrayForRootExpressionIsImpl(hlo_text, root_expression,
                                       /*print_constants=*/false);
  }

  void AssertArrayWithConstantsForRootExpressionIs(
      const std::string& hlo_text, const std::string& root_expression) {
    AssertArrayForRootExpressionIsImpl(hlo_text, root_expression,
                                       /*print_constants=*/true);
  }

 private:
  // Replaces sequences of whitespace with a single space.  This makes the
  // strings being matched against "whitespace insensitive" which lets us indent
  // them for readability.
  std::string CanonicalizeWhitespace(const std::string& text) {
    std::string result;

    for (char c : text) {
      if (!absl::ascii_isspace(c)) {
        result.push_back(c);
      } else if (!result.empty() && result.back() != ' ') {
        result.push_back(' ');
      }
    }

    while (!result.empty() && result.back() == ' ') {
      result.pop_back();
    }

    return result;
  }

  void AssertArrayForRootExpressionIsImpl(const std::string& hlo_text,
                                          const std::string& root_expression,
                                          bool print_constants) {
    IndexedArrayAnalysis indexed_tensor_analysis;
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                            ParseAndReturnVerifiedModule(hlo_text));

    TF_ASSERT_OK_AND_ASSIGN(IndexedArrayAnalysis::Array* const array_result,
                            indexed_tensor_analysis.GetArrayFor(
                                m->entry_computation()->root_instruction()));
    std::string string_result = CanonicalizeWhitespace(
        indexed_tensor_analysis.ToString(array_result, print_constants));
    LOG(INFO) << string_result;
    ASSERT_EQ(string_result, CanonicalizeWhitespace(root_expression));
  }
};

TEST_F(IndexedArrayAnalysisTest, SimpleOneToOneGather) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[5] parameter(1)
  ROOT gather = s32[5,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,3}
}
)";

  AssertArrayForRootExpressionIs(hlo_text,
                                 "(scalar-indexed %operand %indices 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, SimpleOneToOneConstantGather) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] constant({{1,2,3},{1,2,3},{1,2,3}})
  indices = s32[5] parameter(0)
  ROOT gather = s32[5,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,3}
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text, "(scalar-indexed-const (constant s32[3,3]) %indices 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, GatherIsNotScalarIndexed0) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] constant({{1,2,3},{1,2,3},{1,2,3}})
  indices = s32[5,2] parameter(0)
  ROOT gather = s32[5] gather(operand, indices),
      offset_dims={},
      collapsed_slice_dims={0,1},
      start_index_map={0,1},
      index_vector_dim=1,
      slice_sizes={1,1}
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%gather");
}

TEST_F(IndexedArrayAnalysisTest, GatherIsNotScalarIndexed1) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3,1] parameter(0)
  indices = s32[5] parameter(1)
  ROOT gather = s32[5,3] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0,2},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,3,1}
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%gather");
}

TEST_F(IndexedArrayAnalysisTest, GatherIsNotScalarIndexed2) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3,1] parameter(0)
  indices = s32[5] parameter(1)
  ROOT gather = s32[5,2,3] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={2},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={2,3,1}
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%gather");
}

TEST_F(IndexedArrayAnalysisTest, GatherIsNotScalarIndexed3) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[5] parameter(1)
  ROOT gather = s32[5,2] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,2}
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%gather");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_OneToOne) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,3] constant({{1,2,3},{1,2,3},{1,2,3}})
  indices_a = s32[5] parameter(0)
  indices_b = s32[2] parameter(1)
  gather_a = s32[5,3] gather(operand, indices_a),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,3}
  ROOT gather_b = s32[2,3] gather(gather_a, indices_b),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,3}
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed-const (constant s32[3,3]) (scalar-indexed %indices_a "
      "%indices_b 0->[0]) 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_ManyToOneWithOneToOne) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,2] parameter(0)
  indices_a = s32[5,7] parameter(1)
  indices_b = s32[2] parameter(2)
  gather_a = s32[5,3,7] gather(operand, indices_a),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3,1}
  ROOT gather_b = s32[5,3,2] gather(gather_a, indices_b),
      offset_dims={0,1},
      collapsed_slice_dims={2},
      start_index_map={2},
      index_vector_dim=1,
      slice_sizes={5,3,1}
}
)";

  AssertArrayForRootExpressionIs(hlo_text,
                                 "(scalar-indexed %operand (scalar-indexed "
                                 "%indices_a %indices_b 1->[1]) 1->[0,2])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_OneToOneWithManyToOne) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,6] parameter(0)
  indices_a = s32[2] parameter(1)
  indices_b = s32[5,7] parameter(2)
  gather_a = s32[2,6] gather(operand, indices_a),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,6}
  ROOT gather_b = s32[5,6,7] gather(gather_a, indices_b),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,6}
}
)";

  AssertArrayForRootExpressionIs(hlo_text,
                                 "(scalar-indexed %operand (scalar-indexed "
                                 "%indices_a %indices_b 0->[0,1]) 0->[0,2])");
}

TEST_F(IndexedArrayAnalysisTest, GatherOfGather_ManyToOneWithManyToOne) {
  std::string hlo_text = R"(
HloModule SimpleGather

ENTRY main {
  operand = s32[3,2] parameter(0)
  indices_a = s32[5,7] parameter(1)
  indices_b = s32[4,8] parameter(2)
  gather_a = s32[5,3,7] gather(operand, indices_a),
      offset_dims={1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=2,
      slice_sizes={3,1}
  ROOT gather_b = s32[4,5,3,8] gather(gather_a, indices_b),
      offset_dims={1,2},
      collapsed_slice_dims={2},
      start_index_map={2},
      index_vector_dim=2,
      slice_sizes={5,3,1}
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed %operand (scalar-indexed %indices_a %indices_b "
      "1->[0,2]) 1->[0,1,3])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather0) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4}})
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT reshape = s32[5,2,2] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text, "(scalar-indexed-const (constant s32[3,2,2]) %indices 0->[0])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather1) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4}})
  indices = s32[5,7] parameter(0)
  gather = s32[5,4,7] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,4}
  ROOT reshape = s32[5,2,2,7] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed-const (constant s32[3,2,2]) %indices 0->[0,3])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather2) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,2,6] constant({
      {{1,2,3,4,5,6},{1,2,3,4,5,6}},
      {{1,2,3,4,5,6},{1,2,3,4,5,6}},
      {{1,2,3,4,5,6},{1,2,3,4,5,6}}})
  indices = s32[5,7] parameter(0)
  gather = s32[5,2,6,7] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,2,6}
  ROOT reshape = s32[5,3,4,7] reshape(gather)
}
)";

  AssertArrayForRootExpressionIs(
      hlo_text,
      "(scalar-indexed-const (constant s32[3,3,4]) %indices 0->[0,3])");
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather3) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[2,6] constant({
      {1,2,3,4,5,6},{1,2,3,4,5,6}})
  indices = s32[1] parameter(0)
  gather = s32[1,6] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,6}
  ROOT reshape = s32[1,1,6] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(scalar-indexed-const
  (constant s32[2,1,1,6])
  (reshape %indices to s32[])
  0->[])
)";

  AssertArrayForRootExpressionIs(hlo_text, expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather4) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[2,3]{1,0} constant({ { 1, 2, 3 }, { 1, 2, 3 } })

  i.0 = s64[1,3]{1,0} parameter(0)
  g.0 = s32[1,3,3]{2,1,0} gather(operand, i.0), offset_dims={2},
    collapsed_slice_dims={0}, start_index_map={0},
    index_vector_dim=2, slice_sizes={1,3}

  i.1 = s64[1] parameter(1)
  g.1 = s32[1,1,3]{2,1,0} gather(g.0, i.1), offset_dims={0,2},
    collapsed_slice_dims={1}, start_index_map={1},
    index_vector_dim=1, slice_sizes={1,1,3}

  ROOT reshape = s32[1,3]{1,0} reshape(g.1)
}
)";

  const char* expected_root_expression = R"(
(scalar-indexed-const
  (constant s32[2,1,3])
   (reshape
     (scalar-indexed %i.0 %i.1 1->[1])
     to s64[])
  0->[])
)";

  AssertArrayForRootExpressionIs(hlo_text, expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather5) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[1,6] constant({{1,2,3,4,5,6}})
  indices = s32[1] parameter(0)
  gather = s32[1,6] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,6}
  ROOT reshape = s32[1,1,6] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(scalar-indexed-const
  (constant s32[1,1,1,6])
  (reshape %indices to s32[])
  0->[])
)";

  AssertArrayForRootExpressionIs(hlo_text, expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather6) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[1,2,6] constant({{
      {1,2,3,4,5,6},{1,2,3,4,5,6}}})
  indices = s32[1] parameter(0)
  gather = s32[1,1,6] gather(operand, indices),
      offset_dims={1,2},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={1,1,6}
  ROOT reshape = s32[1,1,1,6] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(scalar-indexed-const
  (constant s32[2,1,1,1,6] s32[2,1,1,1,6] {
    { /*i0=0*/ { /*i1=0*/ { /*i2=0*/ { 1, 2, 3, 4, 5, 6 } } } },
    { /*i0=1*/ { /*i1=0*/ { /*i2=0*/ { 1, 2, 3, 4, 5, 6 } } } } })
  (reshape %indices to s32[])
  0->[])
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text,
                                              expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGather7) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[2,6] constant({
      {1,2,3,4,5,6},{1,2,3,4,5,6}})
  indices = s32[1,5] parameter(0)
  gather = s32[1,5,6] gather(operand, indices),
      offset_dims={2},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,6}
  ROOT reshape = s32[1,1,5,6] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(scalar-indexed-const
  (constant s32[2,1,1,6] s32[2,1,1,6] {
    { /*i0=0*/ { /*i1=0*/ { 1, 2, 3, 4, 5, 6 } } },
    { /*i0=1*/ { /*i1=0*/ { 1, 2, 3, 4, 5, 6 } } } })
  (reshape %indices to s32[5])
  0->[2])
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text,
                                              expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGatherNoFold0) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4] constant({{1,2,3,4},{1,2,3,4},{1,2,3,4}})
  indices = s32[5,6] parameter(0)
  gather = s32[5,4,6] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,4}
  ROOT reshape = s32[5,2,2,2,3] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(reshape
  (scalar-indexed-const
    (constant s32[3,4])
    %indices
    0->[0,2])
  to s32[5,2,2,2,3])
)";

  AssertArrayForRootExpressionIs(hlo_text, expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGatherNoFold1) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,5,2] constant({
      {{1,2},{3,4},{5,6},{7,8},{9,10}},
      {{1,2},{3,4},{5,6},{7,8},{9,10}},
      {{1,2},{3,4},{5,6},{7,8},{9,10}}})
  indices = s32[7] parameter(0)
  gather = s32[3,2,7] gather(operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3,1,2}
  ROOT reshape = s32[6,7] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(reshape
  (scalar-indexed-const
    (constant s32[3,5,2])
    %indices
    1->[2])
  to s32[6,7])
)";

  AssertArrayForRootExpressionIs(hlo_text, expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, ReshapeOfGatherNoFold2) {
  std::string hlo_text = R"(
HloModule ReshapeOfGather

ENTRY main {
  operand = s32[3,4,1] constant({
    {{1},{2},{3},{4}},
    {{1},{2},{3},{4}},
    {{1},{2},{3},{4}}})
  indices = s32[5,6] parameter(0)
  gather = s32[5,4,6,1] gather(operand, indices),
      offset_dims={1,3},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=2,
      slice_sizes={1,4,1}
  ROOT reshape = s32[5,2,2,2,3,1] reshape(gather)
}
)";

  const char* expected_root_expression = R"(
(reshape
  (scalar-indexed-const
    (constant s32[3,4,1])
    %indices
    0->[0,2])
  to s32[5,2,2,2,3,1])
)";

  AssertArrayForRootExpressionIs(hlo_text, expected_root_expression);
}

TEST_F(IndexedArrayAnalysisTest, UnaryOpOfGather) {
  std::string hlo_text = R"(
HloModule UnaryOpOfGather

ENTRY main {
  operand = f32[3,4] constant({{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  indices = s32[5] parameter(0)
  gather = f32[5,4] gather(operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT tanh = f32[5,4] tanh(gather)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const (constant f32[3,4] f32[3,4] {
  { 0.761594176, 0.964027584, 0.995054781, 0.999329329 },
  { 0.761594176, 0.995054781, 0.964027584, 0.999329329 },
  { 0.999329329, 0.995054781, 0.964027584, 0.761594176 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, AddBroadcastedScalarWithGather) {
  std::string hlo_text = R"(
HloModule AddBroadcastedScalarWithGather

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant = s32[] constant(5)
  constant_broadcasted = s32[5,4] broadcast(constant), dimensions={}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT add = s32[5,4] add(gather, constant_broadcasted)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { 6, 7, 8, 9 },
  { 6, 8, 7, 9 },
  { 9, 8, 7, 6 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest,
       SubtractBroadcastedScalarWithGather_GatherIsLhs) {
  std::string hlo_text = R"(
HloModule SubtractBroadcastedScalarWithGather

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant = s32[] constant(5)
  constant_broadcasted = s32[5,4] broadcast(constant), dimensions={}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT sub = s32[5,4] subtract(gather, constant_broadcasted)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { -4, -3, -2, -1 },
  { -4, -2, -3, -1 },
  { -1, -2, -3, -4 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest,
       SubtractBroadcastedScalarWithGather_GatherIsRhs) {
  std::string hlo_text = R"(
HloModule SubtractBroadcastedScalarWithGather

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant = s32[] constant(5)
  constant_broadcasted = s32[5,4] broadcast(constant), dimensions={}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT sub = s32[5,4] subtract(constant_broadcasted, gather)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { 4, 3, 2, 1 },
  { 4, 2, 3, 1 },
  { 1, 2, 3, 4 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, AddBroadcastedVectorWithGather) {
  std::string hlo_text = R"(
HloModule AddBroadcastedVectorWithGather

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant_vect = s32[4] constant({10,11,12,13})
  constant_broadcasted = s32[5,4] broadcast(constant_vect), dimensions={1}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT add = s32[5,4] add(gather, constant_broadcasted)
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const (constant s32[3,4] s32[3,4] {
  { 11, 13, 15, 17 },
  { 11, 14, 14, 17 },
  { 14, 14, 14, 14 }
}) %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, AddBroadcastedVectorWithGather_Negative) {
  std::string hlo_text = R"(
HloModule AddBroadcastedVectorWithGather

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{1,3,2,4},{4,3,2,1}})
  constant_vect = s32[5] constant({10,11,12,13,14})
  constant_broadcasted = s32[5,4] broadcast(constant_vect), dimensions={0}
  indices = s32[5] parameter(0)
  gather = s32[5,4] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT add = s32[5,4] add(gather, constant_broadcasted)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%add");
}

TEST_F(IndexedArrayAnalysisTest, RegularUnaryOp) {
  std::string hlo_text = R"(
HloModule RegularUnaryOp

ENTRY main {
  input = f32[100] parameter(0)
  ROOT tanh = f32[100] tanh(input)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%tanh");
}

TEST_F(IndexedArrayAnalysisTest, RegularBinaryOp) {
  std::string hlo_text = R"(
HloModule RegularUnaryOp

ENTRY main {
  input0 = f32[100] parameter(0)
  input1 = f32[100] parameter(1)
  ROOT add = f32[100] add(input0, input1)
}
)";

  AssertArrayForRootExpressionIs(hlo_text, "%add");
}

TEST_F(IndexedArrayAnalysisTest, DotOpBasic_0) {
  std::string hlo_text = R"(
HloModule DotOp

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
  dot_rhs_constant = s32[4,3] constant({{1,2,3},{4,5,6},{7,8,9},{10,11,12}})
  indices = s32[5] parameter(0)
  dot_lhs = s32[5,4] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,4}
  ROOT dot = s32[5,3] dot(dot_lhs, dot_rhs_constant), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const
  (constant s32[3,3] s32[3,3] {
    { 70, 80, 90 },
    { 158, 184, 210 },
    { 246, 288, 330 } })
  %indices 0->[0]))");
}

TEST_F(IndexedArrayAnalysisTest, DotOpBasic_1) {
  std::string hlo_text = R"(
HloModule DotOp

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
  dot_rhs_constant = s32[3,3] constant({{1,2,3},{4,5,6},{7,8,9}})
  indices = s32[5] parameter(0)
  dot_lhs = s32[3,5] gather(gather_operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3,1}
  ROOT dot = s32[5,3] dot(dot_lhs, dot_rhs_constant), lhs_contracting_dims={0}, rhs_contracting_dims={0}
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const
  (constant s32[4,3] s32[4,3] {
    { 84, 99, 114 },
    { 96, 114, 132 },
    { 108, 129, 150 },
    { 120, 144, 168 } })
   %indices 0->[1]))");
}

TEST_F(IndexedArrayAnalysisTest, DotOpBasic_2) {
  std::string hlo_text = R"(
HloModule DotOp

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
  dot_lhs_constant = s32[4,3] constant({{1,2,3},{4,5,6},{7,8,9},{10,11,12}})
  indices = s32[5] parameter(0)
  dot_rhs = s32[3,5] gather(gather_operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3,1}
  ROOT dot = s32[4,5] dot(dot_lhs_constant, dot_rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const
  (constant s32[4,4] s32[4,4] {
    { 38, 44, 50, 56 },
    { 83, 98, 113, 128 },
    { 128, 152, 176, 200 },
    { 173, 206, 239, 272 } })
  %indices 1->[1])
)");
}

TEST_F(IndexedArrayAnalysisTest, DotOpBasic_3) {
  std::string hlo_text = R"(
HloModule DotOp

ENTRY main {
  gather_operand = s32[4,3] constant({{1,2,3},{4,5,6},{7,8,9},{10,11,12}})
  dot_lhs_constant = s32[4,3] constant({{1,2,3},{4,5,6},{7,8,9},{10,11,12}})
  indices = s32[5] parameter(0)
  dot_rhs = s32[5,3] gather(gather_operand, indices),
      offset_dims={1},
      collapsed_slice_dims={0},
      start_index_map={0},
      index_vector_dim=1,
      slice_sizes={1,3}
  ROOT dot = s32[4,5] dot(dot_lhs_constant, dot_rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const
  (constant s32[4,4] s32[4,4] {
    { 14, 32, 50, 68 },
    { 32, 77, 122, 167 },
    { 50, 122, 194, 266 },
    { 68, 167, 266, 365 } })
  %indices 1->[0])
)");
}

TEST_F(IndexedArrayAnalysisTest, DotOpWithBatch) {
  std::string hlo_text = R"(
HloModule DotOp

ENTRY main {
  gather_operand = s32[2,3,2] constant({{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}})
  dot_lhs_constant = s32[2,2,3] constant({{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}})
  indices = s32[4] parameter(0)
  dot_rhs = s32[2,3,4] gather(gather_operand, indices),
      offset_dims={0,1},
      collapsed_slice_dims={2},
      start_index_map={2},
      index_vector_dim=1,
      slice_sizes={2,3,1}
  ROOT dot = s32[2,2,4] dot(dot_lhs_constant, dot_rhs),
      lhs_contracting_dims={2}, rhs_contracting_dims={1},
      lhs_batch_dims={0}, rhs_batch_dims={0}
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, R"(
(scalar-indexed-const
  (constant s32[2,2,2] s32[2,2,2] {
    { { 22, 28 },
      { 49, 64 } },
    { { 220, 244 },
      { 301, 334 } } })
  %indices 3->[2])
)");
}

TEST_F(IndexedArrayAnalysisTest, DotOpNegative) {
  std::string hlo_text = R"(
HloModule DotOp

ENTRY main {
  gather_operand = s32[3,4] constant({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
  dot_rhs_constant = s32[2,3] constant({{1,2,3},{4,5,6}})
  indices = s32[2] parameter(0)
  dot_lhs = s32[3,2] gather(gather_operand, indices),
      offset_dims={0},
      collapsed_slice_dims={1},
      start_index_map={1},
      index_vector_dim=1,
      slice_sizes={3,1}
  ROOT dot = s32[3,3] dot(dot_lhs, dot_rhs_constant), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  AssertArrayWithConstantsForRootExpressionIs(hlo_text, "%dot");
}

}  // namespace
}  // namespace xla

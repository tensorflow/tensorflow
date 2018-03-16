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

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

// NB!  TODO(b/74360564): These tests do not test out of bounds behavior since
// that hasn't been specced yet.

namespace xla {
namespace {

using tensorflow::gtl::nullopt;

class GatherOperationTest : public HloTestBase {
 protected:
  void RunTest(const string& hlo_text, Literal* operand,
               Literal* gather_indices) {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            tools::Parse(hlo_text, config));
    EXPECT_TRUE(
        RunAndCompare(std::move(module), {operand, gather_indices}, nullopt));
  }
};

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV1) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,3] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 3}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherV2) {
  const string hlo_text = R"(
HloModule TensorFlowGatherV2

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[3,2] gather(operand, indices),
      output_window_dims={0},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=1,
      window_bounds={3, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherMultipleBatchDims) {
  const string hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,3,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=2,
      window_bounds={3, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 2}, {2, 1}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_0) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=2,
      window_bounds={1, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdMultipleBatchDims_1) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNdMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2,2] parameter(1)
  ROOT gather = s32[2,1,1,2] gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=2,
      window_bounds={1, 1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR3<int32>({{{0, 2}, {2, 1}}, {{1, 2}, {2, 0}}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNd) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=1,
      window_bounds={1,1,2}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                {{-7, 7}, {-8, 8}, {-9, 9}}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 0}, {1, 0}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, TensorFlowGatherNdNonDefaultIndexVectorDim) {
  const string hlo_text = R"(
HloModule TensorFlowGatherNd

ENTRY main {
  operand = s32[3,3,2] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,2] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0,1},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1,2}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR3<int32>({{{-1, 1}, {-2, 2}, {-3, 3}},  //
                                {{-4, 4}, {-5, 5}, {-6, 6}},  //
                                {{-7, 7}, {-8, 8}, {-9, 9}}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{0, 0}, {1, 0}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, DynamicSlice) {
  const char* hlo_text = R"(
HloModule DynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[1,1] gather(operand, indices),
      output_window_dims={0,1},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({1, 1});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, BatchDynamicSlice) {
  const string hlo_text = R"(
HloModule BatchDynamicSlice

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2,2] parameter(1)
  ROOT gather = s32[2,1,1] gather(operand, indices),
      output_window_dims={1,2},
      elided_window_dims={},
      gather_dims_to_operand_dims={0,1},
      index_vector_dim=0,
      window_bounds={1,1}
}
)";
  std::unique_ptr<Literal> operand =
      Literal::CreateR2<int32>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  std::unique_ptr<Literal> gather_indices =
      Literal::CreateR2<int32>({{2, 1}, {1, 1}});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

XLA_TEST_F(GatherOperationTest, ZeroDimBounds) {
  const char* hlo_text = R"(
HloModule TensorFlowGatherV1

ENTRY main {
  operand = s32[3,0] parameter(0)
  indices = s32[2] parameter(1)
  ROOT gather = s32[2,0] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={0},
      gather_dims_to_operand_dims={0},
      index_vector_dim=1,
      window_bounds={1, 0}
}
)";
  std::unique_ptr<Literal> operand = Literal::CreateR2<int32>({{}, {}, {}});
  std::unique_ptr<Literal> gather_indices = Literal::CreateR1<int32>({0, 2});
  RunTest(hlo_text, operand.get(), gather_indices.get());
}

}  // namespace
}  // namespace xla

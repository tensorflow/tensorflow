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

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class VecOpsReduceTest : public ClientLibraryTestBase {
 public:
  VecOpsReduceTest() : builder_(client_, TestName()) {}

  ComputationDataHandle BuildSampleConstantCube() {
    // clang-format off
    Array3D<float> x3d({
          {{1.0, 2.0, 3.0},   // | dim 1    // } plane 0 in dim 0
           {4.0, 5.0, 6.0}},  // V          // }
           // ---- dim 2 ---->
          {{1.0, 2.0, 3.0},                 // } plane 1 in dim 0
           {4.0, 5.0, 6.0}},
          {{1.0, 2.0, 3.0},                 // } plane 2 in dim 0
           {4.0, 5.0, 6.0}}});
    // clang-format on
    return builder_.ConstantR3FromArray3D<float>(x3d);
  }

  ComputationBuilder builder_;
  ErrorSpec errspec_{1e-3, 0};
};

TEST_F(VecOpsReduceTest, AddReduceR1F32) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);

  auto x = builder_.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0});

  ComputeAndCompareR0<float>(&builder_, -4.2f, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceBigR1F32) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);

  std::vector<float> input(3000);
  std::iota(input.begin(), input.end(), 100.0f);

  auto x = builder_.ConstantR1<float>(input);
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0});

  float expected = std::accumulate(input.begin(), input.end(), 0.0f);
  ComputeAndCompareR0<float>(&builder_, expected, {}, errspec_);
}

TEST_F(VecOpsReduceTest, MaxReduceR1F32) {
  auto max_reducer = CreateScalarMax();

  auto x = builder_.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto max_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), max_reducer,
                      /*dimensions_to_reduce=*/{0});

  ComputeAndCompareR0<float>(&builder_, 2.6f, {}, errspec_);
}

TEST_F(VecOpsReduceTest, MaxReduceR1F32WithNontrivialInit) {
  auto max_reducer = CreateScalarMax();

  auto x = builder_.ConstantR1<float>(
      {2.1, -2.6, 2.6, -4.0, 2.1, 2.3, -5.0, -0.9, -2.4, 1.6});
  auto max_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(4.0f), max_reducer,
                      /*dimensions_to_reduce=*/{0});

  ComputeAndCompareR0<float>(&builder_, 4.0f, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR2F32Dim1) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);

  // clang-format off
  auto x = builder_.ConstantR2<float>({
    {1.0, 2.0, 3.0},    // | dim 0
    {4.0, 5.0, 6.0}});  // |
  // ------ dim 1 ----------
  // clang-format on

  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{1});

  ComputeAndCompareR1<float>(&builder_, {6.0, 15.0}, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR2F32Dim0) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);

  // clang-format off
  auto x = builder_.ConstantR2<float>({
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0}});
  // clang-format on
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0});

  ComputeAndCompareR1<float>(&builder_, {5.0, 7.0, 9.0}, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR3F32Dim2) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{2});

  Array2D<float> expected_array({{6.0f, 15.0f}, {6.0f, 15.0f}, {6.0f, 15.0f}});

  ComputeAndCompareR2<float>(&builder_, expected_array, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR3F32Dim1) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{1});

  Array2D<float> expected_array(
      {{5.0f, 7.0f, 9.0f}, {5.0f, 7.0f, 9.0f}, {5.0f, 7.0f, 9.0f}});

  ComputeAndCompareR2<float>(&builder_, expected_array, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR3F32Dim0) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0});

  Array2D<float> expected_array({{3.0f, 6.0f, 9.0f}, {12.0f, 15.0f, 18.0f}});

  ComputeAndCompareR2<float>(&builder_, expected_array, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR3F32Dims1and2) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{1, 2});

  ComputeAndCompareR1<float>(&builder_, {21.0, 21.0, 21.0}, {}, errspec_);
}

XLA_TEST_F(VecOpsReduceTest, AddReduceR3F32Dims0and2) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0, 2});

  ComputeAndCompareR1<float>(&builder_, {18.0, 45.0}, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR3F32Dims0and1) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0, 1});

  ComputeAndCompareR1<float>(&builder_, {15.0, 21.0, 27.0}, {}, errspec_);
}

TEST_F(VecOpsReduceTest, AddReduceR3F32AllDims) {
  auto sum_reducer = CreateScalarAddComputation(F32, &builder_);
  auto x = BuildSampleConstantCube();
  auto add_reduce =
      builder_.Reduce(x, builder_.ConstantR0<float>(0.0f), sum_reducer,
                      /*dimensions_to_reduce=*/{0, 1, 2});

  ComputeAndCompareR0<float>(&builder_, 63.0, {}, errspec_);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendDebugOptionsFlags(&flag_list);
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}

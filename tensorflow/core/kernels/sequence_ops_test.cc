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

#include <cmath>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RangeOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType input_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "Range")
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(input_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

class LinSpaceOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType input_type, DataType index_type) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "LinSpace")
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(input_type))
                     .Input(FakeInput(index_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(RangeOpTest, Simple_D32) {
  MakeOp(DT_INT32);

  // Feed and run
  AddInputFromArray<int32_t>(TensorShape({}), {0});
  AddInputFromArray<int32_t>(TensorShape({}), {10});
  AddInputFromArray<int32_t>(TensorShape({}), {2});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_INT32, TensorShape({5}));
  test::FillValues<int32_t>(&expected, {0, 2, 4, 6, 8});
  test::ExpectTensorEqual<int32_t>(expected, *GetOutput(0));
}

TEST_F(RangeOpTest, Simple_Half) {
  MakeOp(DT_HALF);

  // Feed and run
  AddInputFromList<Eigen::half, float>(TensorShape({}), {0.5});
  AddInputFromList<Eigen::half, float>(TensorShape({}), {2});
  AddInputFromList<Eigen::half, float>(TensorShape({}), {0.3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_HALF, TensorShape({5}));
  test::FillValues<Eigen::half, float>(&expected, {0.5, 0.8, 1.1, 1.4, 1.7});
  test::ExpectTensorEqual<Eigen::half>(expected, *GetOutput(0));
}

TEST_F(RangeOpTest, Simple_Float) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {0.5});
  AddInputFromArray<float>(TensorShape({}), {2});
  AddInputFromArray<float>(TensorShape({}), {0.3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({5}));
  test::FillValues<float>(&expected, {0.5, 0.8, 1.1, 1.4, 1.7});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(RangeOpTest, Large_Double) {
  MakeOp(DT_DOUBLE);

  // Feed and run
  AddInputFromArray<double>(TensorShape({}), {0.0});
  AddInputFromArray<double>(TensorShape({}), {10000});
  AddInputFromArray<double>(TensorShape({}), {0.5});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_DOUBLE, TensorShape({20000}));
  std::vector<double> result;
  for (int32_t i = 0; i < 20000; ++i) result.push_back(i * 0.5);
  test::FillValues<double>(&expected, absl::Span<const double>(result));
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(RangeOpTest, Range_Size_Overflow) {
  MakeOp(DT_INT64);

  AddInputFromArray<int64_t>(TensorShape({}), {static_cast<int64_t>(5e18)});
  AddInputFromArray<int64_t>(TensorShape({}), {static_cast<int64_t>(-5e18)});
  AddInputFromArray<int64_t>(TensorShape({}), {-1});

  EXPECT_EQ(absl::StrCat("Requires ((limit - start) / delta) <= ",
                         std::numeric_limits<int64_t>::max()),
            RunOpKernel().message());
}

TEST_F(RangeOpTest, Monotonic_Bfloat16) {
  MakeOp(DT_BFLOAT16);

  // 600 elements go past 256, where bfloat16 stops holding every integer, so
  // casting i to bfloat16 before multiplying rounded it. delta is smaller than
  // the bfloat16 spacing near the top of the range, so neighbouring outputs
  // may be equal; each one must still be start + i * delta rounded once.
  const bfloat16 start(0.0f);
  const bfloat16 delta(0.3f);
  AddInputFromArray<bfloat16>(TensorShape({}), {start});
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(180.0f)});
  AddInputFromArray<bfloat16>(TensorShape({}), {delta});
  TF_ASSERT_OK(RunOpKernel());

  const auto flat = GetOutput(0)->flat<bfloat16>();
  constexpr int64_t num = 600;
  ASSERT_EQ(num, GetOutput(0)->NumElements());
  for (int64_t i = 1; i < num; ++i) {
    ASSERT_GE(static_cast<float>(flat(i)), static_cast<float>(flat(i - 1)))
        << "flat(" << i << ") was less than flat(" << i - 1 << ")";
  }
  for (int64_t i = 0; i < num; ++i) {
    ASSERT_EQ(static_cast<float>(static_cast<bfloat16>(
                  static_cast<double>(start) +
                  static_cast<double>(i) * static_cast<double>(delta))),
              static_cast<float>(flat(i)))
        << "flat(" << i << ") is not start + " << i << " * delta";
  }
}

TEST_F(LinSpaceOpTest, Simple_D32) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {3.0});
  AddInputFromArray<float>(TensorShape({}), {7.0});
  AddInputFromArray<int32_t>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({3}));
  test::FillValues<float>(&expected, {3.0, 5.0, 7.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Exact_Endpoints) {
  MakeOp(DT_FLOAT, DT_INT32);

  // Feed and run. The particular values 0., 1., and 42 are chosen to test that
  // the last value is not calculated via an intermediate delta as (1./41)*41,
  // because for IEEE 32-bit floats that returns 0.99999994 != 1.0.
  AddInputFromArray<float>(TensorShape({}), {0.0});
  AddInputFromArray<float>(TensorShape({}), {1.0});
  AddInputFromArray<int32_t>(TensorShape({}), {42});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor output = *GetOutput(0);
  float expected_start = 0.0;
  float start = output.flat<float>()(0);
  EXPECT_EQ(expected_start, start) << expected_start << " vs. " << start;
  float expected_stop = 1.0;
  float stop = output.flat<float>()(output.NumElements() - 1);
  EXPECT_EQ(expected_stop, stop) << expected_stop << " vs. " << stop;
}

TEST_F(LinSpaceOpTest, Single_D64) {
  MakeOp(DT_FLOAT, DT_INT64);

  // Feed and run
  AddInputFromArray<float>(TensorShape({}), {9.0});
  AddInputFromArray<float>(TensorShape({}), {100.0});
  AddInputFromArray<int64_t>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&expected, {9.0});
  test::ExpectTensorEqual<float>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Simple_Double) {
  MakeOp(DT_DOUBLE, DT_INT32);

  // Feed and run
  AddInputFromArray<double>(TensorShape({}), {5.0});
  AddInputFromArray<double>(TensorShape({}), {6.0});
  AddInputFromArray<int32_t>(TensorShape({}), {6});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_DOUBLE, TensorShape({6}));
  test::FillValues<double>(&expected, {5.0, 5.2, 5.4, 5.6, 5.8, 6.0});
  test::ExpectTensorEqual<double>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Simple_Half) {
  MakeOp(DT_HALF, DT_INT32);

  // Feed and run
  AddInputFromArray<Eigen::half>(TensorShape({}), {Eigen::half(3.0f)});
  AddInputFromArray<Eigen::half>(TensorShape({}), {Eigen::half(7.0f)});
  AddInputFromArray<int32_t>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_HALF, TensorShape({3}));
  test::FillValues<Eigen::half>(
      &expected, {Eigen::half(3.0f), Eigen::half(5.0f), Eigen::half(7.0f)});
  test::ExpectTensorEqual<Eigen::half>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Simple_Bfloat16) {
  MakeOp(DT_BFLOAT16, DT_INT32);

  // Feed and run
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(3.0f)});
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(7.0f)});
  AddInputFromArray<int32_t>(TensorShape({}), {3});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output
  Tensor expected(allocator(), DT_BFLOAT16, TensorShape({3}));
  test::FillValues<bfloat16>(
      &expected, {bfloat16(3.0f), bfloat16(5.0f), bfloat16(7.0f)});
  test::ExpectTensorEqual<bfloat16>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, LargeSequence_Half) {
  MakeOp(DT_HALF, DT_INT32);

  // 70000 - 1 is past half's largest finite value (65504). With the
  // interpolation done in T, num - 1 and every i >= 65520 overflowed to +Inf,
  // step became 0, and the interior filled with 0 * +Inf = NaN.
  constexpr int32_t num = 70000;
  AddInputFromArray<Eigen::half>(TensorShape({}), {Eigen::half(0.0f)});
  AddInputFromArray<Eigen::half>(TensorShape({}), {Eigen::half(1.0f)});
  AddInputFromArray<int32_t>(TensorShape({}), {num});
  TF_ASSERT_OK(RunOpKernel());

  const Tensor* output = GetOutput(0);
  ASSERT_EQ(num, output->NumElements());
  const auto flat = output->flat<Eigen::half>();

  int32_t non_finite = 0;
  int32_t first_non_finite = -1;
  for (int32_t i = 0; i < num; ++i) {
    if (!std::isfinite(static_cast<float>(flat(i)))) {
      if (first_non_finite < 0) first_non_finite = i;
      ++non_finite;
    }
  }
  EXPECT_EQ(0, non_finite) << non_finite << " of " << num
                           << " elements were NaN or Inf, first at index "
                           << first_non_finite;

  const double step = 1.0 / (num - 1);
  for (int32_t i : {1, 300, 65519, 65520, 65521, num - 2}) {
    EXPECT_NEAR(i * step, static_cast<float>(flat(i)), 1e-3) << "index " << i;
  }
  EXPECT_EQ(0.0f, static_cast<float>(flat(0)));
  EXPECT_EQ(1.0f, static_cast<float>(flat(num - 1)));
}

TEST_F(LinSpaceOpTest, Monotonic_Bfloat16_Int64Index) {
  MakeOp(DT_BFLOAT16, DT_INT64);

  // 600 indices exercise the int64 index kernels. The step (1/599) is smaller
  // than the bfloat16 spacing above 0.25, so neighbouring outputs may be equal;
  // each one must still be the bfloat16 nearest to i / (num - 1). Rounding the
  // step and the index to bfloat16 before multiplying drifted by several ulps.
  constexpr int64_t num = 600;
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(0.0f)});
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(1.0f)});
  AddInputFromArray<int64_t>(TensorShape({}), {num});
  TF_ASSERT_OK(RunOpKernel());

  const auto flat = GetOutput(0)->flat<bfloat16>();
  ASSERT_EQ(num, GetOutput(0)->NumElements());
  for (int64_t i = 1; i < num; ++i) {
    ASSERT_GE(static_cast<float>(flat(i)), static_cast<float>(flat(i - 1)))
        << "flat(" << i << ") was less than flat(" << i - 1 << ")";
  }
  const double step = 1.0 / static_cast<double>(num - 1);
  for (int64_t i = 1; i < num - 1; ++i) {
    ASSERT_EQ(static_cast<float>(
                  static_cast<bfloat16>(step * static_cast<double>(i))),
              static_cast<float>(flat(i)))
        << "flat(" << i << ") is not the nearest bfloat16 to " << i << "/"
        << num - 1;
  }
  EXPECT_EQ(0.0f, static_cast<float>(flat(0)));
  EXPECT_EQ(1.0f, static_cast<float>(flat(num - 1)));
}

TEST_F(LinSpaceOpTest, Single_Half) {
  MakeOp(DT_HALF, DT_INT32);

  // num == 1 must return [start] without dividing by num - 1 == 0.
  AddInputFromArray<Eigen::half>(TensorShape({}), {Eigen::half(9.0f)});
  AddInputFromArray<Eigen::half>(TensorShape({}), {Eigen::half(100.0f)});
  AddInputFromArray<int32_t>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_HALF, TensorShape({1}));
  test::FillValues<Eigen::half>(&expected, {Eigen::half(9.0f)});
  test::ExpectTensorEqual<Eigen::half>(expected, *GetOutput(0));
}

TEST_F(LinSpaceOpTest, Single_Bfloat16) {
  MakeOp(DT_BFLOAT16, DT_INT32);

  // num == 1 must return [start] without dividing by num - 1 == 0.
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(9.0f)});
  AddInputFromArray<bfloat16>(TensorShape({}), {bfloat16(100.0f)});
  AddInputFromArray<int32_t>(TensorShape({}), {1});
  TF_ASSERT_OK(RunOpKernel());

  Tensor expected(allocator(), DT_BFLOAT16, TensorShape({1}));
  test::FillValues<bfloat16>(&expected, {bfloat16(9.0f)});
  test::ExpectTensorEqual<bfloat16>(expected, *GetOutput(0));
}

}  // namespace
}  // namespace tensorflow

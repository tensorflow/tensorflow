/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/strided_slice_op.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace {

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension"
template <typename T>
static void SliceHelper(::testing::benchmark::State& state) {
  const int size = state.range(0);
  Graph* g = new Graph(OpRegistry::Global());
  DataType dt = DataTypeToEnum<T>::v();
  int kDim = 100;
  int kMaxSize = 15000;
  CHECK_LT(size, kMaxSize);

  Tensor begin(DT_INT32, TensorShape({2}));
  begin.flat<int32_t>()(0) = 10;
  begin.flat<int32_t>()(1) = 10;

  Tensor end(DT_INT32, TensorShape({2}));
  end.flat<int32_t>()(0) = 10 + kDim;
  end.flat<int32_t>()(1) = 10 + size;

  Tensor strides(DT_INT32, TensorShape({2}));
  strides.flat<int32_t>()(0) = 1;
  strides.flat<int32_t>()(1) = 1;

  Tensor input(dt, TensorShape({2 * kDim, kMaxSize}));
  input.flat<T>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "StridedSlice")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, begin))
                  .Input(test::graph::Constant(g, end))
                  .Input(test::graph::Constant(g, strides))
                  .Attr("T", dt)
                  .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kDim *
                          size * sizeof(T));
}

void BM_SliceFloat(::testing::benchmark::State& state) {
  SliceHelper<float>(state);
}

BENCHMARK(BM_SliceFloat)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_SliceComplex64(::testing::benchmark::State& state) {
  SliceHelper<std::complex<float>>(state);
}

BENCHMARK(BM_SliceComplex64)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_SliceBFloat16(::testing::benchmark::State& state) {
  SliceHelper<bfloat16>(state);
}

BENCHMARK(BM_SliceBFloat16)->UseRealTime()->Arg(100)->Arg(1000)->Arg(10000);

void BM_ValidateStridedSliceOp(::testing::benchmark::State& state) {
  int kDim = 100;
  int kMaxSize = 15000;
  int size = 100;
  Tensor begin = test::AsTensor<int32_t>({10, 10});
  Tensor end = test::AsTensor<int32_t>({10 + kDim, 10 + size});
  Tensor strides = test::AsTensor<int32_t>({1, 1});
  TensorShape input_shape({2 * kDim, kMaxSize});

  for (auto s : state) {
    TensorShape processing_shape, final_shape;
    bool is_identity = true, slice_dim0 = true, is_simple_slice = true;
    absl::InlinedVector<int64_t, 4UL> begin_out, end_out, strides_out;
    const int32_t begin_mask = 0;
    const int32_t end_mask = 0;
    const int32_t ellipsis_mask = 0;
    const int32_t new_axis_mask = 0;
    const int32_t shrink_axis_mask = 0;

    TF_CHECK_OK(ValidateStridedSliceOp(
        &begin, &end, strides, input_shape, begin_mask, end_mask, ellipsis_mask,
        new_axis_mask, shrink_axis_mask, &processing_shape, &final_shape,
        &is_identity, &is_simple_slice, &slice_dim0, &begin_out, &end_out,
        &strides_out));
  }
}

BENCHMARK(BM_ValidateStridedSliceOp);

class StridedSliceZeroCopyTest : public OpsTestBase {
 protected:
  void MakeStridedSliceOp(DataType dtype, int begin_mask = 0, int end_mask = 0,
                          int shrink_axis_mask = 0) {
    TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                     .Input(FakeInput(dtype))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Attr("T", dtype)
                     .Attr("Index", DT_INT32)
                     .Attr("begin_mask", begin_mask)
                     .Attr("end_mask", end_mask)
                     .Attr("shrink_axis_mask", shrink_axis_mask)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(StridedSliceZeroCopyTest,
       ZeroCopiesPrefixSliceWithLeadingUnitDimAndUnalignedInnerDim) {
  MakeStridedSliceOp(DT_FLOAT, /*begin_mask=*/0b101, /*end_mask=*/0b101);

  std::vector<float> values(1 * 8 * 7);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = static_cast<float>(i);
  }
  AddInputFromArray<float>(TensorShape({1, 8, 7}), values);
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 3, 7});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({1, 3, 7}));
  EXPECT_TRUE(output->IsAligned());
  EXPECT_EQ(output->data(), GetInput(0).data());
  for (int i = 0; i < 3 * 7; ++i) {
    EXPECT_FLOAT_EQ(output->flat<float>()(i), static_cast<float>(i))
        << "at index " << i;
  }
}

TEST_F(StridedSliceZeroCopyTest, ZeroCopies2DPrefixSliceWithLeadingUnitDim) {
  MakeStridedSliceOp(DT_FLOAT);

  AddInputFromArray<float>(TensorShape({1, 10}),
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 3});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({1, 3}));
  EXPECT_TRUE(output->IsAligned());
  EXPECT_EQ(output->data(), GetInput(0).data());
  test::ExpectTensorEqual<float>(
      *output, test::AsTensor<float>({0, 1, 2}, TensorShape({1, 3})));
}

TEST_F(StridedSliceZeroCopyTest, ZeroCopiesShrunkLeadingUnitDim) {
  MakeStridedSliceOp(DT_FLOAT, /*begin_mask=*/0b100, /*end_mask=*/0b100,
                     /*shrink_axis_mask=*/0b001);

  std::vector<float> values(1 * 6 * 5);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = static_cast<float>(i);
  }
  AddInputFromArray<float>(TensorShape({1, 6, 5}), values);
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 2, 5});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({2, 5}));
  EXPECT_TRUE(output->IsAligned());
  EXPECT_EQ(output->data(), GetInput(0).data());
}

TEST_F(StridedSliceZeroCopyTest, CopiesWhenStartOffsetIsNonZero) {
  MakeStridedSliceOp(DT_FLOAT, /*begin_mask=*/0b101, /*end_mask=*/0b101);

  // Even when the row stride is 64-byte aligned, non-zero start offsets copy
  // rather than aliasing.
  std::vector<float> values(1 * 6 * 16);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = static_cast<float>(i);
  }
  AddInputFromArray<float>(TensorShape({1, 6, 16}), values);
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 2, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 5, 16});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({1, 3, 16}));
  EXPECT_TRUE(output->IsAligned());
  EXPECT_FALSE(output->SharesBufferWith(GetInput(0)));
  for (int i = 0; i < 3 * 16; ++i) {
    EXPECT_FLOAT_EQ(output->flat<float>()(i), static_cast<float>(i + 2 * 16))
        << "at index " << i;
  }
}

TEST_F(StridedSliceZeroCopyTest,
       DoesNotShareBufferForEmptySliceWithLeadingUnitDim) {
  MakeStridedSliceOp(DT_FLOAT, /*begin_mask=*/0b101, /*end_mask=*/0b101);

  std::vector<float> values(1 * 8 * 7, 1.0f);
  AddInputFromArray<float>(TensorShape({1, 8, 7}), values);
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 0, 7});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({1, 0, 7}));
  EXPECT_FALSE(output->SharesBufferWith(GetInput(0)));
}

TEST_F(StridedSliceZeroCopyTest,
       DoesNotShareBufferForDegenerateEmptySliceWithLeadingUnitDim) {
  MakeStridedSliceOp(DT_FLOAT, /*begin_mask=*/0b101, /*end_mask=*/0b101);

  std::vector<float> values(1 * 8 * 7, 1.0f);
  AddInputFromArray<float>(TensorShape({1, 8, 7}), values);
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 5, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 3, 7});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({1, 0, 7}));
  EXPECT_FALSE(output->SharesBufferWith(GetInput(0)));
}

TEST_F(StridedSliceZeroCopyTest, DoesNotZeroCopyNonMemcpyStringType) {
  MakeStridedSliceOp(DT_STRING, /*begin_mask=*/0b01, /*end_mask=*/0b01);

  AddInputFromArray<tstring>(TensorShape({1, 4}), {"a", "b", "c", "d"});
  AddInputFromArray<int32_t>(TensorShape({2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 2});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({1, 2}));
  EXPECT_NE(output->data(), GetInput(0).data());
  test::ExpectTensorEqual<tstring>(
      *output, test::AsTensor<tstring>({"a", "b"}, TensorShape({1, 2})));
}

TEST_F(StridedSliceZeroCopyTest, CopiesWhenOuterBatchDimExceedsOne) {
  MakeStridedSliceOp(DT_FLOAT, /*begin_mask=*/0b101, /*end_mask=*/0b101);

  std::vector<float> values(2 * 4 * 8);
  for (int i = 0; i < static_cast<int>(values.size()); ++i) {
    values[i] = static_cast<float>(i);
  }
  AddInputFromArray<float>(TensorShape({2, 4, 8}), values);
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 0, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {2, 2, 8});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 1, 1});

  TF_ASSERT_OK(RunOpKernel());

  Tensor* output = GetOutput(0);
  EXPECT_EQ(output->shape(), TensorShape({2, 2, 8}));
  EXPECT_NE(output->data(), GetInput(0).data());
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 2 * 8; ++i) {
      EXPECT_FLOAT_EQ(output->flat<float>()(b * 16 + i),
                      static_cast<float>(b * 32 + i))
          << "at batch " << b << ", index " << i;
    }
  }
}

}  // namespace
}  // namespace tensorflow

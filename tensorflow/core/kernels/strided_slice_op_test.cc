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

#include <functional>
#include <memory>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
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
#include "tensorflow/core/util/strided_slice_op.h"

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
  begin.flat<int32>()(0) = 10;
  begin.flat<int32>()(1) = 10;

  Tensor end(DT_INT32, TensorShape({2}));
  end.flat<int32>()(0) = 10 + kDim;
  end.flat<int32>()(1) = 10 + size;

  Tensor strides(DT_INT32, TensorShape({2}));
  strides.flat<int32>()(0) = 1;
  strides.flat<int32>()(1) = 1;

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
  Tensor begin = test::AsTensor<int32>({10, 10});
  Tensor end = test::AsTensor<int32>({10 + kDim, 10 + size});
  Tensor strides = test::AsTensor<int32>({1, 1});
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

}  // namespace
}  // namespace tensorflow

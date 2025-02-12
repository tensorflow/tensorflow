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
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
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
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

static void BM_UnsortedSegmentReduction(::testing::benchmark::State& state,
                                        const string& reduction, int num_rows,
                                        int num_cols, int segment_size) {
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  // Create inputs
  absl::InlinedVector<TensorValue, 4> reduction_inputs;
  TensorShape shape1({num_rows, num_cols});
  Tensor input(DT_FLOAT, shape1);
  // input.flat<float>().setRandom();
  reduction_inputs.push_back({nullptr, &input});

  TensorShape shape2({num_rows});
  Tensor indices(DT_INT32, shape2);
  test::FillFn<int>(&indices,
                    [&segment_size](int i) -> int { return i % segment_size; });
  reduction_inputs.push_back({nullptr, &indices});

  Tensor num_segments(DT_INT32, TensorShape({}));
  num_segments.scalar<int>()() = segment_size;
  reduction_inputs.push_back({nullptr, &num_segments});

  NodeDef reduction_node_def;
  TF_CHECK_OK(NodeDefBuilder(reduction, reduction)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(DT_INT32))
                  .Finalize(&reduction_node_def));
  absl::Status status;
  std::unique_ptr<OpKernel> reduction_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     reduction_node_def, TF_GRAPH_DEF_VERSION, &status));

  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = reduction_inputs;
  params.op_kernel = reduction_op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> reduction_context(
      new OpKernelContext(&params));

  reduction_op->Compute(reduction_context.get());
  TF_CHECK_OK(reduction_context->status());
  for (auto s : state) {
    delete reduction_context->release_output(0).tensor;
    reduction_op->Compute(reduction_context.get());
  }
  int64_t bytes_per_iter =
      static_cast<int64_t>(num_rows * num_cols * sizeof(float));
  state.SetBytesProcessed(bytes_per_iter * state.iterations());
}

#define BM_UnsortedReduce(O, R, C, S)                                        \
  static void BM_##O##_##R##_##C##_##S(::testing::benchmark::State& state) { \
    BM_UnsortedSegmentReduction(state, #O, R, C, S);                         \
  }                                                                          \
  BENCHMARK(BM_##O##_##R##_##C##_##S);

#define BM_UnsortedReduce_Arg(R, C, S) \
  BM_UnsortedReduce(UnsortedSegmentSum, R, C, S);

BM_UnsortedReduce_Arg(4096, 1024, 1);
BM_UnsortedReduce_Arg(4096, 1024, 128);

template <typename Index>
static void BM_SegmentReduction(::testing::benchmark::State& state,
                                const string& reduction, Index num_rows,
                                Index num_cols, Index segment_size) {
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0"));

  // Create inputs
  absl::InlinedVector<TensorValue, 4> reduction_inputs;
  TensorShape shape1({num_rows, num_cols});
  Tensor input1(DT_FLOAT, shape1);
  reduction_inputs.push_back({nullptr, &input1});

  TensorShape shape2({num_rows});
  Tensor input2(DataTypeToEnum<Index>::v(), shape2);
  test::FillFn<Index>(&input2, [&num_rows, &segment_size](Index i) -> Index {
    return std::min(i / segment_size, num_rows - 1);
  });
  reduction_inputs.push_back({nullptr, &input2});

  NodeDef reduction_node_def;
  TF_CHECK_OK(NodeDefBuilder(reduction, reduction)
                  .Input(FakeInput(DT_FLOAT))
                  .Input(FakeInput(DataTypeToEnum<Index>::v()))
                  .Finalize(&reduction_node_def));
  absl::Status status;
  std::unique_ptr<OpKernel> reduction_op(
      CreateOpKernel(DEVICE_CPU, device.get(), cpu_allocator(),
                     reduction_node_def, TF_GRAPH_DEF_VERSION, &status));
  OpKernelContext::Params params;
  params.device = device.get();
  params.frame_iter = FrameAndIter(0, 0);
  params.inputs = reduction_inputs;
  params.op_kernel = reduction_op.get();
  std::vector<AllocatorAttributes> attrs;
  test::SetOutputAttrs(&params, &attrs);

  std::unique_ptr<OpKernelContext> reduction_context(
      new OpKernelContext(&params));

  reduction_op->Compute(reduction_context.get());
  TF_CHECK_OK(reduction_context->status());
  for (auto s : state) {
    delete reduction_context->release_output(0).tensor;
    reduction_op->Compute(reduction_context.get());
  }
  int64_t bytes_per_iter =
      static_cast<int64_t>(num_rows * num_cols * sizeof(float));
  state.SetBytesProcessed(bytes_per_iter * state.iterations());
}

#define BM_Reduce(O, R, C, S)                          \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int32( \
      ::testing::benchmark::State & state) {           \
    BM_SegmentReduction<int32>(state, #O, R, C, S);    \
  }                                                    \
  static void BM_Reduce_##O##_##R##_##C##_##S##_int64( \
      ::testing::benchmark::State & state) {           \
    BM_SegmentReduction<int64_t>(state, #O, R, C, S);  \
  }                                                    \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int32);  \
  BENCHMARK(BM_Reduce_##O##_##R##_##C##_##S##_int64);

#define BM_Reduce_Arg(R, C, S)    \
  BM_Reduce(SegmentSum, R, C, S); \
  BM_Reduce(SegmentMean, R, C, S);

BM_Reduce_Arg(64, 32, 1);
BM_Reduce_Arg(4096, 128, 1);

BM_Reduce_Arg(16, 8, 2);
BM_Reduce_Arg(64, 32, 2);
BM_Reduce_Arg(4096, 32, 2);
BM_Reduce_Arg(4096, 128, 2);

template <DataType T>
static void SparseSegmentMeanGradHelper(::testing::benchmark::State& state,
                                        float uniqueness, int size) {
  typedef typename EnumToDataType<T>::Type DT;
  Graph* g = new Graph(OpRegistry::Global());
  CHECK_LE(uniqueness, 1.0);
  CHECK_GT(uniqueness, 0.0);

  const int kNumIndices = size;
  Tensor indices(DT_INT32, TensorShape({kNumIndices}));
  auto indices_flat = indices.flat<int32>();
  Tensor segments(DT_INT32, TensorShape({kNumIndices}));
  auto segments_flat = segments.flat<int32>();

  int kUniqueIndices = uniqueness * kNumIndices;
  Tensor output_dim0(DT_INT32, TensorShape({}));
  output_dim0.scalar<int32>()() = kUniqueIndices;

  for (int i = 0; i < kNumIndices; ++i) {
    indices_flat(i) = (i * 31) % kUniqueIndices;
    segments_flat(i) = i * .8;
  }

  const int kDim1 = segments_flat(kNumIndices - 1) + 1;
  const int kDim2 = 128;
  Tensor input(T, TensorShape({kDim1, kDim2}));
  input.flat<DT>().setRandom();

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "SparseSegmentMeanGrad")
                  .Input(test::graph::Constant(g, input))
                  .Input(test::graph::Constant(g, indices))
                  .Input(test::graph::Constant(g, segments))
                  .Input(test::graph::Constant(g, output_dim0))
                  .Attr("T", T)
                  .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (kDim1 * kDim2) * sizeof(float));
}

static void BM_SparseSegmentMeanGrad_Low_FP32(
    ::testing::benchmark::State& state) {
  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_FLOAT>(state, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High_FP32(
    ::testing::benchmark::State& state) {
  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_FLOAT>(state, 0.01, size);
}

static void BM_SparseSegmentMeanGrad_Low_BF16(
    ::testing::benchmark::State& state) {
  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_BFLOAT16>(state, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High_BF16(
    ::testing::benchmark::State& state) {
  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_BFLOAT16>(state, 0.01, size);
}

static void BM_SparseSegmentMeanGrad_Low_FP16(
    ::testing::benchmark::State& state) {
  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_HALF>(state, 1.0, size);
}

static void BM_SparseSegmentMeanGrad_High_FP16(
    ::testing::benchmark::State& state) {
  const int size = state.range(0);

  return SparseSegmentMeanGradHelper<DT_HALF>(state, 0.01, size);
}

BENCHMARK(BM_SparseSegmentMeanGrad_Low_FP32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High_FP32)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_Low_BF16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High_BF16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_Low_FP16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);
BENCHMARK(BM_SparseSegmentMeanGrad_High_FP16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000);

}  // namespace tensorflow

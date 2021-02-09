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
#include <vector>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

template <typename T>
void FillTensorWithRandomValues(Tensor* t, int string_length, int64* bytes) {
  t->flat<T>().setRandom();
  *bytes = t->flat<T>().size() * sizeof(T);
}

template <>
void FillTensorWithRandomValues<tstring>(Tensor* t, int string_length,
                                         int64* bytes) {
  auto ts = t->flat<tstring>();
  *bytes = 0;
  for (int i = 0; i < ts.size(); i++) {
    ts(i) = tstring(string_length, 'x');
    *bytes += sizeof(ts(i)) + ts(i).size();
  }
}

// For the benchmark, we set up two 2-dimensional tensors, each kDim1 x 'dim'
// in size, and concat them together along "concat_dimension".  If T is
// std::string, then the length of individual strings in the tensors will be
// of length "string_length".
template <typename T>
static void ConcatHelper(::testing::benchmark::State& state,
                         int concat_dimension, int dim2,
                         int string_length = 0) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dt = DataTypeToEnum<T>::v();
  const int kDim1 = 100;
  Tensor concat_dim(DT_INT32, TensorShape({}));
  concat_dim.scalar<int32>()() = concat_dimension;
  Tensor in0(dt, TensorShape({kDim1, dim2}));
  Tensor in1(dt, TensorShape({kDim1, dim2}));
  int64 in0_bytes, in1_bytes;
  FillTensorWithRandomValues<T>(&in0, string_length, &in0_bytes);
  FillTensorWithRandomValues<T>(&in1, string_length, &in1_bytes);

  Node* node;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("n"), "Concat")
          .Input(test::graph::Constant(g, concat_dim))
          .Input({test::graph::Constant(g, in0), test::graph::Constant(g, in1)})
          .Attr("N", 2)
          .Attr("T", dt)
          .Finalize(g, &node));

  test::Benchmark("cpu", g, /*old_benchmark_api=*/false).Run(state);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) *
                          (in0_bytes + in1_bytes));
}

void BM_ConcatDim0Float(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<float>(state, 0, dim2);
}

void BM_ConcatDim1Float(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<float>(state, 1, dim2);
}

BENCHMARK(BM_ConcatDim0Float)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);
BENCHMARK(BM_ConcatDim1Float)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);

void BM_ConcatDim0String(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);
  const int string_length = state.range(1);

  ConcatHelper<tstring>(state, 0, dim2, string_length);
}

BENCHMARK(BM_ConcatDim0String)
    ->UseRealTime()
    ->ArgPair(1, 16)
    ->ArgPair(1, 10000)
    ->ArgPair(100, 16);

void BM_ConcatDim1uint8(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<uint8>(state, 1, dim2);
}
void BM_ConcatDim1int16(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<int16>(state, 1, dim2);
}
void BM_ConcatDim1bfloat16(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<bfloat16>(state, 1, dim2);
}

BENCHMARK(BM_ConcatDim1uint8)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);
BENCHMARK(BM_ConcatDim1int16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);
BENCHMARK(BM_ConcatDim1bfloat16)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);

template <typename T>
static void ConcatManyHelper(::testing::benchmark::State& state,
                             int concat_dimension, int dim2) {
  Graph* g = new Graph(OpRegistry::Global());

  DataType dt = DataTypeToEnum<T>::v();
  const int kDim1 = 40000;
  const int kNumInputs = 64;
  Tensor concat_dim(DT_INT32, TensorShape({}));
  concat_dim.scalar<int32>()() = concat_dimension;
  std::vector<NodeBuilder::NodeOut> inputs;
  inputs.reserve(kNumInputs);
  for (int i = 0; i < kNumInputs; ++i) {
    Tensor in(dt, TensorShape({kDim1, dim2}));
    in.flat<T>().setRandom();
    inputs.push_back(test::graph::Constant(g, in));
  }

  Node* node;
  TF_CHECK_OK(NodeBuilder(g->NewName("n"), "Concat")
                  .Input(test::graph::Constant(g, concat_dim))
                  .Input(inputs)
                  .Attr("N", 64)
                  .Attr("T", dt)
                  .Finalize(g, &node));
  test::Benchmark("cpu", g, /*old_benchmark_api*/ false).Run(state);
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * kDim1 *
                          dim2 * kNumInputs * sizeof(T));
}

void BM_ConcatManyDim1bfloat16(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatManyHelper<bfloat16>(state, 1, dim2);
}

BENCHMARK(BM_ConcatManyDim1bfloat16)->UseRealTime()->Arg(18)->Arg(34)->Arg(60);

void MemcpyAlternativeHelper(::testing::benchmark::State& state, int dim2) {
  const int kDim1 = 100;
  std::vector<float> data1(kDim1 * dim2, 1.0f);
  std::vector<float> data2(kDim1 * dim2, 2.0f);

  for (auto s : state) {
    const size_t n0 = data1.size();
    const size_t n1 = data2.size();
    float* result = new float[n0 + n1];
    memcpy(&result[0], &data1[0], n0 * sizeof(float));
    memcpy(&result[n0], &data2[0], n1 * sizeof(float));
    delete[] result;
  }
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) *
                          ((kDim1 * dim2) + (kDim1 * dim2)) * sizeof(float));
}

void BM_MemcpyAlternativeDim0(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  MemcpyAlternativeHelper(state, dim2);
}
void BM_MemcpyAlternativeDim1(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  MemcpyAlternativeHelper(state, dim2);
}

BENCHMARK(BM_MemcpyAlternativeDim0)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);
BENCHMARK(BM_MemcpyAlternativeDim1)
    ->UseRealTime()
    ->Arg(1000)
    ->Arg(100000)
    ->Arg(1000000);

typedef Eigen::TensorMap<Eigen::Tensor<bfloat16, 1, Eigen::RowMajor>,
                         Eigen::Unaligned>
    EigenMap;
void MemcpyManyAlternative1(::testing::benchmark::State& state) {
  int dim2 = state.range(0);
  const int kDim1 = 40000;
  const int kNumCopies = 64;
  const int size = kDim1 * dim2 * kNumCopies;
  bfloat16* data = new bfloat16[size];
  EigenMap map(data, size);
  map.setRandom();

  for (auto s : state) {
    std::vector<bfloat16*> inputs(kNumCopies);
    for (int i = 0; i < kNumCopies; ++i) {
      inputs[i] = &data[i * kDim1 * dim2];
    }
    bfloat16* result = new bfloat16[size];
    for (int j = 0; j < kNumCopies; ++j) {
      bfloat16* output = &result[j * dim2];
      for (int i = 0; i < kDim1; ++i) {
        if (i + 1 < kDim1) {
          port::prefetch<port::PREFETCH_HINT_T0>(inputs[j] + dim2);
        }
        memcpy(output, inputs[j], dim2 * sizeof(bfloat16));
        inputs[j] += dim2;
        output += dim2 * kNumCopies;
      }
    }
    delete[] result;
  }
  delete[] data;
  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * kDim1 *
                          dim2 * kNumCopies * sizeof(bfloat16));
}

void MemcpyManyAlternative2(::testing::benchmark::State& state) {
  int dim2 = state.range(0);
  const int kDim1 = 40000;
  const int kNumCopies = 64;
  const int size = kDim1 * dim2 * kNumCopies;
  bfloat16* data = new bfloat16[size];
  EigenMap map(data, size);
  map.setRandom();

  std::vector<bfloat16*> inputs(kNumCopies);
  for (auto s : state) {
    bfloat16* result = new bfloat16[size];
    for (int i = 0; i < kNumCopies; ++i) {
      inputs[i] = &data[i * kDim1 * dim2];
    }
    bfloat16* output = result;
    for (int i = 0; i < kDim1; ++i) {
      for (int j = 0; j < kNumCopies; ++j) {
        if (j + 1 < kNumCopies) {
          port::prefetch<port::PREFETCH_HINT_T0>(inputs[j + 1]);
        }
        memcpy(output, inputs[j], dim2 * sizeof(bfloat16));
        inputs[j] += dim2;
        output += dim2;
      }
    }
    delete[] result;
  }
  delete[] data;

  state.SetBytesProcessed(static_cast<int64>(state.iterations()) * kDim1 *
                          dim2 * kNumCopies * sizeof(bfloat16));
}

BENCHMARK(MemcpyManyAlternative1)
    ->Arg(16)
    ->Arg(17)
    ->Arg(18)
    ->Arg(32)
    ->Arg(33)
    ->Arg(34)
    ->Arg(60)
    ->Arg(64)
    ->Arg(65);

BENCHMARK(MemcpyManyAlternative2)
    ->Arg(16)
    ->Arg(17)
    ->Arg(18)
    ->Arg(32)
    ->Arg(33)
    ->Arg(34)
    ->Arg(60)
    ->Arg(64)
    ->Arg(65);

}  // namespace
}  // namespace tensorflow

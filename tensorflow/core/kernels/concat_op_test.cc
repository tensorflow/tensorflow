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

#include <gtest/gtest.h>
#include "absl/base/prefetch.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace {

using tensorflow::tstring;

template <typename T>
void FillTensorWithRandomValues(Tensor* t, int string_length, int64_t* bytes) {
  t->flat<T>().setRandom();
  *bytes = t->flat<T>().size() * sizeof(T);
}

template <>
void FillTensorWithRandomValues<tstring>(Tensor* t, int string_length,
                                         int64_t* bytes) {
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
  concat_dim.scalar<int32_t>()() = concat_dimension;
  Tensor in0(dt, TensorShape({kDim1, dim2}));
  Tensor in1(dt, TensorShape({kDim1, dim2}));
  int64_t in0_bytes, in1_bytes;
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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
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

  ConcatHelper<uint8_t>(state, 1, dim2);
}
void BM_ConcatDim1int16(::testing::benchmark::State& state) {
  const int dim2 = state.range(0);

  ConcatHelper<int16_t>(state, 1, dim2);
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
  concat_dim.scalar<int32_t>()() = concat_dimension;
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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kDim1 *
                          dim2 * kNumInputs * sizeof(T));
}

TEST(ConcatOnTStringTest, TestTStringsAreDeepCopied) {
  // 1. Create a new graph scope.
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();

  // 2. Define input placeholders.
  auto input1 =
      tensorflow::ops::Placeholder(root.WithOpName("input1"), DT_STRING,
                                   ops::Placeholder::Shape({
                                       2,
                                   }));
  auto input2 =
      tensorflow::ops::Placeholder(root.WithOpName("input2"), DT_STRING,
                                   ops::Placeholder::Shape({
                                       2,
                                   }));

  // 3. Define the axis for concatenation. Concatenates over the first dimension
  auto axis = ops::Const(root.WithOpName("axis"), {0});

  // 4. Add the ConcatV2 operation.
  std::vector<Output> inputs = {input1, input2};
  auto concat_op =
      tensorflow::ops::Concat(root.WithOpName("my_concat"), inputs, axis);

  // 5. Create a session and run the graph (example)
  ClientSession session(root);
  std::vector<tstring> owned_tstrings = {"abc", "def", "ghi", "jkl"};

  // 6. Create view-typed `tstring` for checking data content are deep-copied.
  tstring first_element;
  first_element.assign_as_view(owned_tstrings[0]);

  tstring second_element;
  second_element.assign_as_view(owned_tstrings[1]);

  tstring third_element;
  third_element.assign_as_view(owned_tstrings[2]);

  tstring fourth_element;
  fourth_element.assign_as_view(owned_tstrings[3]);

  Tensor t1(DT_STRING, TensorShape({
                           2,
                       }));

  t1.flat<tstring>().setValues({first_element, second_element});
  Tensor t2(DT_STRING, TensorShape({
                           2,
                       }));
  t2.flat<tstring>().setValues({third_element, fourth_element});

  std::vector<Tensor> outputs;

  TF_ASSERT_OK(
      session.Run({{input1, t1}, {input2, t2}}, {concat_op.output}, &outputs));

  ASSERT_EQ(outputs.size(), 1);

  Tensor& output = outputs[0];

  EXPECT_EQ(output.flat<tstring>()(0), tstring("abc"));
  EXPECT_EQ(output.flat<tstring>()(1), tstring("def"));
  EXPECT_EQ(output.flat<tstring>()(2), tstring("ghi"));
  EXPECT_EQ(output.flat<tstring>()(3), tstring("jkl"));

  // 7. Mutates the upstream `owned_tstrings` should not change the output
  //    because Concat should always deep copy the data content even when
  //    the input `tstring` are of view type. This is served as a guardrail to
  //    simulate use-after-free scenario when upstream `tstring` is freed but we
  //    still want to manipulate downstream output.
  owned_tstrings[0].mdata()[0] = 'q';
  owned_tstrings[1].mdata()[0] = 'z';
  owned_tstrings[2].mdata()[0] = 'x';
  owned_tstrings[3].mdata()[0] = 'y';

  EXPECT_EQ(output.flat<tstring>()(0), tstring("abc"));
  EXPECT_EQ(output.flat<tstring>()(1), tstring("def"));
  EXPECT_EQ(output.flat<tstring>()(2), tstring("ghi"));
  EXPECT_EQ(output.flat<tstring>()(3), tstring("jkl"));
}

TEST(ConcatOpTest, MultiThreadedDim0UnevenAndZeroSizedInputs) {
  // Test multi-threaded dim0 == 1 path with > 16 unevenly sized inputs,
  // including zero-sized inputs (with data() == nullptr), exceeding the 4 MiB
  // multi-threading threshold.
  const int kNumInputs = 20;
  const std::vector<int> input_sizes = {36000, 0,      102000, 15000,  63000,
                                        24000, 135000, 0,      48000,  69000,
                                        21000, 93000,  57000,  126000, 0,
                                        84000, 33000,  108000, 27000,  156000};
  // Total elements = 1,197,000 floats * sizeof(float) = 4,788,000 bytes
  // (> 4 MiB).

  // 1. Direct ConcatCPU test with explicit multi-threaded DeviceBase.
  // Exercises worker-thread slice loops, boundary conditions, heap fallback
  // for > 16 inputs, and safe handling of zero-sized inputs with null data().
  {
    thread::ThreadPool pool(Env::Default(), "test_pool", 4);
    DeviceBase::CpuWorkerThreads worker_threads;
    worker_threads.num_threads = 4;
    worker_threads.workers = &pool;
    DeviceBase device(Env::Default());
    device.set_tensorflow_cpu_worker_threads(&worker_threads);

    int total_elements = 0;
    for (int sz : input_sizes) {
      total_elements += sz;
    }

    std::vector<Tensor> storage;
    storage.reserve(kNumInputs);
    std::vector<std::unique_ptr<typename TTypes<float, 2>::ConstMatrix>> inputs;
    inputs.reserve(kNumInputs);

    std::vector<float> expected;
    expected.reserve(total_elements);
    float current_val = 1.0f;

    for (int i = 0; i < kNumInputs; ++i) {
      int sz = input_sizes[i];
      if (sz == 0) {
        // Explicitly pass {1, 0} matrix with nullptr data pointer, exactly as
        // callers like TensorListConcat can pass.
        inputs.emplace_back(
            new typename TTypes<float, 2>::ConstMatrix(nullptr, 1, 0));
      } else {
        storage.emplace_back(DT_FLOAT, TensorShape({1, sz}));
        auto matrix = storage.back().matrix<float>();
        for (int j = 0; j < sz; ++j) {
          matrix(0, j) = current_val;
          expected.push_back(current_val);
          current_val += 1.0f;
        }
        // Use flat<float>().data() to correctly construct the read-only
        // ConstMatrix from the tensor's raw pointer; matrix<float>() returns a
        // non-const Eigen::TensorMap which is not implicitly convertible.
        inputs.emplace_back(new typename TTypes<float, 2>::ConstMatrix(
            storage.back().flat<float>().data(), 1, sz));
      }
    }

    Tensor output_tensor(DT_FLOAT, TensorShape({1, total_elements}));
    auto output_matrix = output_tensor.matrix<float>();

    ConcatCPU<float>(&device, inputs, &output_matrix);

    auto flat_out = output_tensor.flat<float>();
    ASSERT_EQ(flat_out.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_EQ(flat_out(i), expected[i])
          << "Direct ConcatCPU value mismatch at index " << i;
    }
  }  // inner scope
}  // TEST(ConcatOpTest, MultiThreadedDim0UnevenAndZeroSizedInputs)

// Test that the multi-row (dim0 > 1) sharded path correctly handles zero-sized
// inputs (sizes[j] == 0 with data() == nullptr) without undefined behaviour.
TEST(ConcatOpTest, MultiRowShardedZeroSizedInputs) {
  const int kDim0 =
      3;  // 1 < kDim0 < num_threads (4) to exercise partial-row fallback
  const int kNumInputs = 20;
  const std::vector<int> input_sizes = {
      24000, 0,     68000, 10000, 42000, 16000, 90000, 0,     32000, 46000,
      14000, 62000, 38000, 84000, 0,     56000, 22000, 72000, 18000, 104000};
  // Total = 798,000 floats * kDim0 rows * sizeof(float) = 9,576,000 bytes
  // (> 4 MiB).

  thread::ThreadPool pool(Env::Default(), "test_pool", 4);
  DeviceBase::CpuWorkerThreads worker_threads;
  worker_threads.num_threads = 4;
  worker_threads.workers = &pool;
  DeviceBase device(Env::Default());
  device.set_tensorflow_cpu_worker_threads(&worker_threads);

  int total_cols = 0;
  for (int sz : input_sizes) total_cols += sz;

  std::vector<Tensor> storage;
  storage.reserve(kNumInputs);
  std::vector<std::unique_ptr<typename TTypes<float, 2>::ConstMatrix>> inputs;
  inputs.reserve(kNumInputs);

  std::vector<float> expected;
  expected.reserve(static_cast<size_t>(kDim0) * total_cols);
  float current_val = 1.0f;

  // Fill each row of each input tensor with sequential values.
  for (int i = 0; i < kNumInputs; ++i) {
    int sz = input_sizes[i];
    if (sz == 0) {
      // Zero-sized input: {kDim0, 0} — data() == nullptr.
      inputs.emplace_back(
          new typename TTypes<float, 2>::ConstMatrix(nullptr, kDim0, 0));
    } else {
      storage.emplace_back(DT_FLOAT, TensorShape({kDim0, sz}));
      auto flat = storage.back().flat<float>();
      for (int k = 0; k < kDim0 * sz; ++k) {
        flat(k) = current_val++;
      }
      inputs.emplace_back(new typename TTypes<float, 2>::ConstMatrix(
          storage.back().flat<float>().data(), kDim0, sz));
    }
  }

  // Build expected output: for each row r, concatenate all inputs' row r.
  for (int r = 0; r < kDim0; ++r) {
    int storage_idx = 0;
    for (int i = 0; i < kNumInputs; ++i) {
      int sz = input_sizes[i];
      if (sz == 0) continue;
      const auto& t = storage[storage_idx++];
      auto flat = t.flat<float>();
      for (int k = 0; k < sz; ++k) {
        expected.push_back(flat(r * sz + k));
      }
    }
  }

  Tensor output_tensor(DT_FLOAT, TensorShape({kDim0, total_cols}));
  auto output_matrix = output_tensor.matrix<float>();
  ConcatCPU<float>(&device, inputs, &output_matrix);

  auto flat_out = output_tensor.flat<float>();
  ASSERT_EQ(static_cast<size_t>(flat_out.size()), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(flat_out(i), expected[i])
        << "MultiRow ConcatCPU value mismatch at index " << i;
  }
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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
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
          absl::PrefetchToLocalCache(inputs[j] + dim2);
        }
        memcpy(output, inputs[j], dim2 * sizeof(bfloat16));
        inputs[j] += dim2;
        output += dim2 * kNumCopies;
      }
    }
    delete[] result;
  }
  delete[] data;
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kDim1 *
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
          absl::PrefetchToLocalCache(inputs[j + 1]);
        }
        memcpy(output, inputs[j], dim2 * sizeof(bfloat16));
        inputs[j] += dim2;
        output += dim2;
      }
    }
    delete[] result;
  }
  delete[] data;

  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * kDim1 *
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

// Verifies that the single-threaded dim0==1 fast-path safely skips zero-sized
// inputs whose data() == nullptr without invoking copier.Copy on null pointers.
TEST(ConcatOpTest, SingleThreadedDim0ZeroSizedInputs) {
  thread::ThreadPool pool(Env::Default(), "test_pool", 4);
  DeviceBase::CpuWorkerThreads worker_threads;
  worker_threads.num_threads = 4;
  worker_threads.workers = &pool;
  DeviceBase device(Env::Default());
  device.set_tensorflow_cpu_worker_threads(&worker_threads);

  Tensor t1(DT_FLOAT, TensorShape({1, 10}));
  t1.flat<float>().setConstant(42.0f);

  // Build inputs: [empty, t1, empty] — the two empty tensors have null data().
  std::vector<std::unique_ptr<typename TTypes<float, 2>::ConstMatrix>> inputs;
  inputs.emplace_back(new
                      typename TTypes<float, 2>::ConstMatrix(nullptr, 1, 0));
  inputs.emplace_back(new typename TTypes<float, 2>::ConstMatrix(
      t1.flat<float>().data(), 1, 10));
  inputs.emplace_back(new
                      typename TTypes<float, 2>::ConstMatrix(nullptr, 1, 0));

  Tensor output(DT_FLOAT, TensorShape({1, 10}));
  auto out_matrix = output.matrix<float>();
  ConcatCPU<float>(&device, inputs, &out_matrix);

  EXPECT_EQ(output.flat<float>()(0), 42.0f);
  EXPECT_EQ(output.flat<float>()(9), 42.0f);
}

class ConcatAndPackCpuTest : public OpsTestBase {
 protected:
  ConcatAndPackCpuTest() {
    worker_threads_.num_threads = 4;
    worker_threads_.workers = &pool_;
    device_->set_tensorflow_cpu_worker_threads(&worker_threads_);
  }

  ~ConcatAndPackCpuTest() override {
    device_->set_tensorflow_cpu_worker_threads(nullptr);
  }

  template <typename T>
  absl::Status VerifyPack(const TensorShape& in_shape, int num, int axis) {
    DataType dt = DataTypeToEnum<T>::v();
    TF_RETURN_IF_ERROR(NodeDefBuilder("pack_op", "Pack")
                           .Input(FakeInput(num, dt))
                           .Attr("N", num)
                           .Attr("T", dt)
                           .Attr("axis", axis)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());
    inputs_.clear();
    managed_.clear();

    for (int i = 0; i < num; ++i) {
      Tensor t(dt, in_shape);
      for (int64_t k = 0; k < t.NumElements(); ++k) {
        if constexpr (std::is_same_v<T, bool>) {
          t.flat<bool>()(k) = ((i + k) % 2) == 0;
        } else {
          t.flat<T>()(k) = static_cast<T>((i * 17 + k) % 97);
        }
      }
      managed_.push_back(std::make_unique<Tensor>(t));
      inputs_.emplace_back(managed_.back().get());
    }
    TF_RETURN_IF_ERROR(RunOpKernel());

    const Tensor& out = *GetOutput(0);
    int eff_axis = axis < 0 ? axis + in_shape.dims() + 1 : axis;
    TensorShape exp_shape(in_shape);
    TF_RETURN_IF_ERROR(exp_shape.InsertDimWithStatus(eff_axis, num));
    if (out.shape() != exp_shape) {
      return absl::InternalError(
          absl::StrCat("shape mismatch: ", out.shape().DebugString(), " vs ",
                       exp_shape.DebugString()));
    }

    int64_t before = 1, after = 1;
    for (int d = 0; d < eff_axis; ++d) before *= exp_shape.dim_size(d);
    for (int d = eff_axis + 1; d < exp_shape.dims(); ++d) {
      after *= exp_shape.dim_size(d);
    }
    const auto out_flat = out.flat<T>();
    for (int64_t b = 0; b < before; ++b) {
      for (int j = 0; j < num; ++j) {
        const auto in_flat = managed_[j]->flat<T>();
        for (int64_t a = 0; a < after; ++a) {
          const T actual = out_flat((b * num + j) * after + a);
          const T expected = in_flat(b * after + a);
          if (actual != expected) {
            return absl::InternalError("element mismatch in Pack output");
          }
        }
      }
    }
    return absl::OkStatus();
  }

  template <typename T, typename AxisT = int32_t>
  absl::Status VerifyConcatV2(int64_t before,
                              const std::vector<int64_t>& cols_per_in,
                              int64_t inner, AxisT axis,
                              bool vector_axis = false) {
    DataType dt = DataTypeToEnum<T>::v();
    DataType axis_dt = DataTypeToEnum<AxisT>::v();
    const int num = static_cast<int>(cols_per_in.size());
    TF_RETURN_IF_ERROR(NodeDefBuilder("concat_v2", "ConcatV2")
                           .Input(FakeInput(num, dt))
                           .Input(FakeInput(axis_dt))
                           .Attr("N", num)
                           .Attr("T", dt)
                           .Attr("Tidx", axis_dt)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());
    inputs_.clear();
    managed_.clear();

    int64_t total_c = 0;
    for (int i = 0; i < num; ++i) {
      total_c += cols_per_in[i];
      TensorShape shape({before, cols_per_in[i], inner});
      Tensor t(dt, shape);
      for (int64_t k = 0; k < t.NumElements(); ++k) {
        if constexpr (std::is_same_v<T, tstring>) {
          t.flat<tstring>()(k) =
              ((i + k) % 3 == 0)
                  ? ""
                  : (((i + k) % 3 == 1)
                         ? absl::StrCat("s_", i, "_", k)
                         : absl::StrCat(
                               "long_heap_backed_string_payload_value_", i, "_",
                               k));
        } else {
          t.flat<T>()(k) = static_cast<T>((i * 13 + k) % 113);
        }
      }
      managed_.push_back(std::make_unique<Tensor>(t));
      inputs_.emplace_back(managed_.back().get());
    }
    Tensor axis_t(axis_dt, vector_axis ? TensorShape({1}) : TensorShape({}));
    axis_t.flat<AxisT>()(0) = axis;
    managed_.push_back(std::make_unique<Tensor>(axis_t));
    inputs_.emplace_back(managed_.back().get());

    TF_RETURN_IF_ERROR(RunOpKernel());
    const Tensor& out = *GetOutput(0);
    const TensorShape exp_shape({before, total_c, inner});
    if (out.shape() != exp_shape) {
      return absl::InternalError(
          absl::StrCat("shape mismatch: ", out.shape().DebugString(), " vs ",
                       exp_shape.DebugString()));
    }

    const auto out_flat = out.flat<T>();
    int64_t out_idx = 0;
    for (int64_t b = 0; b < before; ++b) {
      for (int i = 0; i < num; ++i) {
        const int64_t span = cols_per_in[i] * inner;
        const auto in_flat = managed_[i]->flat<T>();
        for (int64_t a = 0; a < span; ++a) {
          const T actual = out_flat(out_idx++);
          const T expected = in_flat(b * span + a);
          if (actual != expected) {
            return absl::InternalError("element mismatch in ConcatV2 output");
          }
        }
      }
    }
    return absl::OkStatus();
  }

  thread::ThreadPool pool_{Env::Default(), "concat_pack_test_pool", 4};
  DeviceBase::CpuWorkerThreads worker_threads_;
  std::vector<std::unique_ptr<Tensor>> managed_;
};

TEST_F(ConcatAndPackCpuTest, PackAllAxesTypesAndSizes) {
  for (int num : {1, 2, 4, 7, 16, 33}) {
    TF_EXPECT_OK(VerifyPack<float>(TensorShape({}), num, 0));
    TF_EXPECT_OK(VerifyPack<int64_t>(TensorShape({}), num, 0));
    TF_EXPECT_OK(VerifyPack<bool>(TensorShape({19}), num, 0));
    TF_EXPECT_OK(VerifyPack<bool>(TensorShape({19}), num, -1));
    for (int axis : {0, 1, 2, -1}) {
      TF_EXPECT_OK(VerifyPack<float>(TensorShape({13, 17}), num, axis));
      TF_EXPECT_OK(VerifyPack<bfloat16>(TensorShape({13, 17}), num, axis));
      TF_EXPECT_OK(VerifyPack<int32_t>(TensorShape({11, 8}), num, axis));
      TF_EXPECT_OK(VerifyPack<int64_t>(TensorShape({9, 5}), num, axis));
    }
  }
  // Zero-element inputs/outputs.
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({0, 5}), 4, 0));
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({3, 0}), 4, 1));
  // Medium (2-3 MB, L3-inline) and Large (> 16 MB, multi-threaded) Pack.
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({256, 256}), 12, 0));
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({256, 256}), 12, 1));
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({1024, 16}), 35, -1));
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({512, 1024}), 9, 0));
  TF_EXPECT_OK(VerifyPack<float>(TensorShape({512, 1024}), 9, 1));
}

TEST_F(ConcatAndPackCpuTest, ConcatV2ShapesTypesAndLargeShards) {
  TF_EXPECT_OK(VerifyConcatV2<float>(1, {0, 17}, 4, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(1, {0, 19, 0}, 8, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(0, {3, 5}, 4, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(4, {0, 0}, 4, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(17, {3, 0, 5, 1, 8, 4, 7}, 3, 1));
  TF_EXPECT_OK(VerifyConcatV2<bfloat16>(19, {4, 4, 4, 4, 4, 4, 4, 4, 4}, 8, 1));
  TF_EXPECT_OK(VerifyConcatV2<int64_t>(11, {2, 5, 3, 7}, 4, -2));
  // DT_INT64 axis and legacy 1-D length-1 vector axis tensors.
  TF_EXPECT_OK((VerifyConcatV2<float, int64_t>(5, {3, 7}, 4, 1)));
  TF_EXPECT_OK((VerifyConcatV2<float, int32_t>(5, {3, 7}, 4, 1, true)));
  TF_EXPECT_OK((VerifyConcatV2<float, int64_t>(5, {3, 7}, 4, -2, true)));
  TF_EXPECT_OK(VerifyConcatV2<tstring>(1, {0, 5, 0, 7}, 3, 1));
  TF_EXPECT_OK(VerifyConcatV2<tstring>(32, {8, 0, 12, 16}, 4, 1));
  // Medium (2-3 MB, L3-inline) and Large (> 16 MB, multi-threaded) ConcatV2.
  TF_EXPECT_OK(VerifyConcatV2<float>(1, {200, 350, 0, 450}, 768, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(256, {128, 64, 192, 0, 256}, 4, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(1, {1200, 1500, 0, 1800}, 1024, 1));
  TF_EXPECT_OK(VerifyConcatV2<float>(128, {1024, 512, 0, 1536, 1280}, 8, 1));
  TF_EXPECT_OK(VerifyConcatV2<tstring>(16, {64, 0, 96, 128}, 4, 1));
}

TEST_F(ConcatAndPackCpuTest, ValidationErrors) {
  auto run_concat_v2 = [&](const std::vector<TensorShape>& shapes, int axis,
                           const TensorShape& axis_shape = TensorShape({})) {
    const int num = static_cast<int>(shapes.size());
    TF_EXPECT_OK(NodeDefBuilder("concat_v2_err", "ConcatV2")
                     .Input(FakeInput(num, DT_FLOAT))
                     .Input(FakeInput(DT_INT32))
                     .Attr("N", num)
                     .Attr("T", DT_FLOAT)
                     .Attr("Tidx", DT_INT32)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    inputs_.clear();
    managed_.clear();
    for (const TensorShape& s : shapes) {
      managed_.push_back(std::make_unique<Tensor>(DT_FLOAT, s));
      inputs_.emplace_back(managed_.back().get());
    }
    Tensor axis_t(DT_INT32, axis_shape);
    if (axis_t.NumElements() > 0) {
      axis_t.flat<int32_t>()(0) = axis;
    }
    managed_.push_back(std::make_unique<Tensor>(axis_t));
    inputs_.emplace_back(managed_.back().get());
    return RunOpKernel();
  };

  EXPECT_THAT(run_concat_v2({TensorShape({}), TensorShape({})}, 0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(run_concat_v2({TensorShape({2}), TensorShape({})}, 0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(run_concat_v2({TensorShape({2, 3}), TensorShape({2})}, 0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(run_concat_v2({TensorShape({2, 3}), TensorShape({4, 3})}, 1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(run_concat_v2({TensorShape({2, 3}), TensorShape({2, 3})}, 0,
                            TensorShape({2})),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(run_concat_v2({TensorShape({2, 3}), TensorShape({2, 3})}, 0,
                            TensorShape({1, 1})),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  auto run_pack = [&](const std::vector<TensorShape>& shapes, int axis) {
    const int num = static_cast<int>(shapes.size());
    TF_EXPECT_OK(NodeDefBuilder("pack_err", "Pack")
                     .Input(FakeInput(num, DT_FLOAT))
                     .Attr("N", num)
                     .Attr("T", DT_FLOAT)
                     .Attr("axis", axis)
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    inputs_.clear();
    managed_.clear();
    for (const TensorShape& s : shapes) {
      managed_.push_back(std::make_unique<Tensor>(DT_FLOAT, s));
      inputs_.emplace_back(managed_.back().get());
    }
    return RunOpKernel();
  };

  EXPECT_THAT(run_pack({TensorShape({3}), TensorShape({4})}, 0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(run_pack({TensorShape({2, 6}), TensorShape({3, 4})}, 0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

struct DirectConcatPackBenchRunner {
  std::unique_ptr<Device> device;
  std::unique_ptr<OpKernel> kernel;
  std::vector<Tensor> tensors;
  absl::InlinedVector<TensorValue, 4UL> input_values;
  std::vector<AllocatorAttributes> out_attrs;
  OpKernelContext::Params params;

  DirectConcatPackBenchRunner(const NodeDef& ndef,
                              std::vector<Tensor> in_tensors,
                              int num_threads = 4)
      : tensors(std::move(in_tensors)) {
    SessionOptions opts;
    opts.config.set_intra_op_parallelism_threads(num_threads);
    opts.config.set_inter_op_parallelism_threads(1);
    opts.config.set_use_per_session_threads(true);
    device = DeviceFactory::NewDevice("CPU", opts,
                                      "/job:localhost/replica:0/task:0");
    absl::Status s;
    kernel = CreateOpKernel(DEVICE_CPU, device.get(),
                            device->GetAllocator(AllocatorAttributes()), ndef,
                            TF_GRAPH_DEF_VERSION, &s);
    TF_CHECK_OK(s);
    input_values.reserve(tensors.size());
    for (Tensor& t : tensors) {
      input_values.emplace_back(&t);
    }
    out_attrs.resize(kernel->num_outputs());
    params.device = device.get();
    params.op_kernel = kernel.get();
    params.inputs = input_values;
    params.output_attr_array = out_attrs.data();
  }

  void Run(::testing::benchmark::State& state) {
    for (int w = 0; w < 3; ++w) {
      OpKernelContext ctx(&params, kernel->num_outputs());
      device->Compute(kernel.get(), &ctx);
      TF_CHECK_OK(ctx.status());
    }
    for (auto _ : state) {
      OpKernelContext ctx(&params, kernel->num_outputs());
      device->Compute(kernel.get(), &ctx);
    }
  }
};

template <typename T>
Tensor MakePackConcatTensor(const TensorShape& shape, int seed) {
  DataType dt = DataTypeToEnum<T>::v();
  Tensor t(dt, shape);
  for (int64_t i = 0; i < t.NumElements(); ++i) {
    t.flat<T>()(i) = static_cast<T>(((i + seed) % 61) - 30);
  }
  return t;
}

template <typename T>
void RunPackBench(::testing::benchmark::State& state,
                  const TensorShape& in_shape, int num, int axis) {
  DataType dt = DataTypeToEnum<T>::v();
  NodeDef ndef;
  TF_CHECK_OK(NodeDefBuilder("pack_bm", "Pack")
                  .Input(FakeInput(num, dt))
                  .Attr("N", num)
                  .Attr("T", dt)
                  .Attr("axis", axis)
                  .Finalize(&ndef));
  std::vector<Tensor> ins;
  ins.reserve(num);
  for (int i = 0; i < num; ++i) {
    ins.push_back(MakePackConcatTensor<T>(in_shape, i));
  }
  DirectConcatPackBenchRunner runner(ndef, std::move(ins), 4);
  runner.Run(state);
}

template <typename T>
void RunConcatV2Bench(::testing::benchmark::State& state,
                      const std::vector<TensorShape>& shapes, int axis) {
  DataType dt = DataTypeToEnum<T>::v();
  const int num = static_cast<int>(shapes.size());
  NodeDef ndef;
  TF_CHECK_OK(NodeDefBuilder("concat_bm", "ConcatV2")
                  .Input(FakeInput(num, dt))
                  .Input(FakeInput(DT_INT32))
                  .Attr("N", num)
                  .Attr("T", dt)
                  .Attr("Tidx", DT_INT32)
                  .Finalize(&ndef));
  std::vector<Tensor> ins;
  ins.reserve(num + 1);
  for (int i = 0; i < num; ++i) {
    ins.push_back(MakePackConcatTensor<T>(shapes[i], i));
  }
  Tensor axis_t(DT_INT32, TensorShape({}));
  axis_t.scalar<int32_t>()() = axis;
  ins.push_back(axis_t);
  DirectConcatPackBenchRunner runner(ndef, std::move(ins), 4);
  runner.Run(state);
}

void BM_Fleet_Pack_Scalars_Float_N32(::testing::benchmark::State& state) {
  RunPackBench<float>(state, TensorShape({}), 32, 0);
}
BENCHMARK(BM_Fleet_Pack_Scalars_Float_N32)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_Pack_LastAxis_Float_256x1_N32(
    ::testing::benchmark::State& state) {
  RunPackBench<float>(state, TensorShape({256}), 32, 1);
}
BENCHMARK(BM_Fleet_Pack_LastAxis_Float_256x1_N32)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_Pack_Axis0_Float_N16_64x128(::testing::benchmark::State& state) {
  RunPackBench<float>(state, TensorShape({64, 128}), 16, 0);
}
BENCHMARK(BM_Fleet_Pack_Axis0_Float_N16_64x128)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_Pack_InnerAxis_Float_N8_128x32(
    ::testing::benchmark::State& state) {
  RunPackBench<float>(state, TensorShape({128, 32}), 8, 1);
}
BENCHMARK(BM_Fleet_Pack_InnerAxis_Float_N8_128x32)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_ConcatV2_Small_Float_N16_1x32_Axis1(
    ::testing::benchmark::State& state) {
  std::vector<TensorShape> shapes(16, TensorShape({1, 32}));
  RunConcatV2Bench<float>(state, shapes, 1);
}
BENCHMARK(BM_Fleet_ConcatV2_Small_Float_N16_1x32_Axis1)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_ConcatV2_MultiRow_Float_N16_128x16_Axis1(
    ::testing::benchmark::State& state) {
  std::vector<TensorShape> shapes(16, TensorShape({128, 16}));
  RunConcatV2Bench<float>(state, shapes, 1);
}
BENCHMARK(BM_Fleet_ConcatV2_MultiRow_Float_N16_128x16_Axis1)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_ConcatV2_Axis0_Float_N8_64x256(
    ::testing::benchmark::State& state) {
  std::vector<TensorShape> shapes(8, TensorShape({64, 256}));
  RunConcatV2Bench<float>(state, shapes, 0);
}
BENCHMARK(BM_Fleet_ConcatV2_Axis0_Float_N8_64x256)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_ConcatV2_Uneven_BF16_N12_256xVar_Axis1(
    ::testing::benchmark::State& state) {
  const int cols[12] = {8, 16, 64, 12, 32, 48, 16, 24, 64, 8, 32, 60};
  std::vector<TensorShape> shapes;
  shapes.reserve(12);
  for (int c : cols) shapes.push_back(TensorShape({256, c}));
  RunConcatV2Bench<bfloat16>(state, shapes, 1);
}
BENCHMARK(BM_Fleet_ConcatV2_Uneven_BF16_N12_256xVar_Axis1)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Fleet_ConcatV2_Large_Float_N4_1024x512_Axis1(
    ::testing::benchmark::State& state) {
  std::vector<TensorShape> shapes(4, TensorShape({1024, 512}));
  RunConcatV2Bench<float>(state, shapes, 1);
}
BENCHMARK(BM_Fleet_ConcatV2_Large_Float_N4_1024x512_Axis1)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Guard_ConcatOffset_Int32_N8(::testing::benchmark::State& state) {
  NodeDef ndef;
  TF_CHECK_OK(NodeDefBuilder("co", "ConcatOffset")
                  .Input(FakeInput(DT_INT32))
                  .Input(FakeInput(8, DT_INT32))
                  .Attr("N", 8)
                  .Finalize(&ndef));
  std::vector<Tensor> ins;
  Tensor cdim(DT_INT32, TensorShape({}));
  cdim.scalar<int32_t>()() = 1;
  ins.push_back(cdim);
  for (int i = 0; i < 8; ++i) {
    Tensor s(DT_INT32, TensorShape({3}));
    s.vec<int32_t>()(0) = 16;
    s.vec<int32_t>()(1) = 8 + i;
    s.vec<int32_t>()(2) = 32;
    ins.push_back(s);
  }
  DirectConcatPackBenchRunner runner(ndef, std::move(ins), 4);
  runner.Run(state);
}
BENCHMARK(BM_Guard_ConcatOffset_Int32_N8)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

void BM_Guard_ConcatV2_TString_N4_64x8(::testing::benchmark::State& state) {
  NodeDef ndef;
  TF_CHECK_OK(NodeDefBuilder("cs", "ConcatV2")
                  .Input(FakeInput(4, DT_STRING))
                  .Input(FakeInput(DT_INT32))
                  .Attr("N", 4)
                  .Attr("T", DT_STRING)
                  .Attr("Tidx", DT_INT32)
                  .Finalize(&ndef));
  std::vector<Tensor> ins;
  for (int i = 0; i < 4; ++i) {
    Tensor t(DT_STRING, TensorShape({64, 8}));
    for (int k = 0; k < 512; ++k) {
      t.flat<tstring>()(k) = "feature_token_val";
    }
    ins.push_back(t);
  }
  Tensor axis_t(DT_INT32, TensorShape({}));
  axis_t.scalar<int32_t>()() = 1;
  ins.push_back(axis_t);
  DirectConcatPackBenchRunner runner(ndef, std::move(ins), 4);
  runner.Run(state);
}
BENCHMARK(BM_Guard_ConcatV2_TString_N4_64x8)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

}  // namespace
}  // namespace tensorflow

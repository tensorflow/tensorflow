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

#include <unordered_map>

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef std::map<std::pair<int, int>, Tensor> ExampleTensorMap;

// Fillers to fill the underlying repeated array in protobuf.
class BytesFiller {
 public:
  BytesFiller() : dense_default(DT_STRING, TensorShape()) {}
  void operator()(Feature* f) const {
    f->mutable_bytes_list()->add_value("abcd1234abcd1234abcd1234abcd1234!");
  }
  Tensor dense_default;
  DataType dtype = DT_STRING;
};

class Int64Filler {
 public:
  Int64Filler() : dense_default(DT_INT64, TensorShape()) {}
  void operator()(Feature* f) const {
    f->mutable_int64_list()->add_value(1729);
  }
  Tensor dense_default;
  DataType dtype = DT_INT64;
};

class FloatFiller {
 public:
  FloatFiller() : dense_default(DT_FLOAT, TensorShape()) {}
  void operator()(Feature* f) const {
    f->mutable_float_list()->add_value(1.729);
  }
  Tensor dense_default;
  DataType dtype = DT_FLOAT;
};

template <typename T>
struct ExampleStore {
  typedef T Filler;
  static ExampleTensorMap GetSerializedExamples() {
    ExampleTensorMap examples;
    int keys[] = {10, 100, 1000};
    int batch_sizes[] = {128, 512};
    Example example;
    Filler fill;
    for (int num_keys : keys) {
      for (int batch_size : batch_sizes) {
        Tensor record_string(DT_STRING, TensorShape({batch_size}));
        auto string_t = record_string.vec<string>();
        example.Clear();
        for (int b = 0; b < batch_size; ++b) {
          for (int k = 0; k < num_keys; ++k) {
            string k_str = strings::Printf("feature_%d", k);
            Feature f;
            fill(&f);
            Features* features = example.mutable_features();
            (*features->mutable_feature())[k_str] = f;
          }
          CHECK(example.SerializeToString(&string_t(b)));
        }
        examples[std::make_pair(batch_size, num_keys)] = record_string;
      }
    }
    return examples;
  }
  static ExampleTensorMap serialized_example;
};

template <>
ExampleTensorMap ExampleStore<BytesFiller>::serialized_example =
    ExampleStore<BytesFiller>::GetSerializedExamples();

template <>
ExampleTensorMap ExampleStore<Int64Filler>::serialized_example =
    ExampleStore<Int64Filler>::GetSerializedExamples();

template <>
ExampleTensorMap ExampleStore<FloatFiller>::serialized_example =
    ExampleStore<FloatFiller>::GetSerializedExamples();

template <typename S, bool BenchmarkDense>
struct BenchmarkOptions {
  bool benchmark_dense = BenchmarkDense;
  typedef S Store;
  typename S::Filler filler;
};

template <typename Options>
static Graph* ParseExample(int batch_size, int num_keys) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized =
      Options::Store::serialized_example[std::make_pair(batch_size, num_keys)];
  Tensor names(DT_STRING, TensorShape({batch_size}));

  std::vector<NodeBuilder::NodeOut> sparse_keys;
  std::vector<NodeBuilder::NodeOut> dense_keys;
  std::vector<NodeBuilder::NodeOut> dense_defaults;
  std::vector<DataType> sparse_types;
  std::vector<TensorShape> dense_shapes;
  Options opt;
  for (int i = 0; i < num_keys; ++i) {
    Tensor key(DT_STRING, TensorShape());
    key.scalar<string>()() = strings::Printf("feature_%d", i);
    if (opt.benchmark_dense) {
      dense_keys.emplace_back(test::graph::Constant(g, key));
      dense_defaults.emplace_back(
          test::graph::Constant(g, opt.filler.dense_default));
      dense_shapes.push_back(TensorShape());
    } else {
      sparse_keys.emplace_back(test::graph::Constant(g, key));
      sparse_types.push_back(opt.filler.dtype);
    }
  }

  Node* ret;
  TF_EXPECT_OK(NodeBuilder(g->NewName("n"), "ParseExample")
                   .Input(test::graph::Constant(g, serialized))
                   .Input(test::graph::Constant(g, names))
                   .Input(sparse_keys)
                   .Input(dense_keys)
                   .Input(dense_defaults)
                   .Attr("sparse_types", sparse_types)
                   .Attr("dense_shapes", dense_shapes)
                   .Finalize(g, &ret));

  return g;
}

// Benchmark settings (Sparse, Dense) X (Bytes, Int64, Float)
typedef BenchmarkOptions<ExampleStore<BytesFiller>, false> SparseString;
typedef BenchmarkOptions<ExampleStore<BytesFiller>, true> DenseString;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, false> SparseIn64;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, true> DenseInt64;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, false> SparseFloat;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, true> DenseFloat;

// B == batch_size, K == num_keys.  K must be one of 10, 100, 1000
#define BM_ParseExample(TYPE, B, K)                                      \
  static void BM_ParseExample##_##TYPE##_##B##_##K(int iters) {          \
    int64 items_per_iter = static_cast<int64>(B) * K;                    \
    testing::UseRealTime();                                              \
    testing::ItemsProcessed(static_cast<int64>(iters) * items_per_iter); \
    test::Benchmark("cpu", ParseExample<TYPE>(B, K)).Run(iters);         \
  }                                                                      \
  BENCHMARK(BM_ParseExample##_##TYPE##_##B##_##K);

#define BM_AllParseExample(B, K)       \
  BM_ParseExample(SparseString, B, K); \
  BM_ParseExample(DenseString, B, K);  \
  BM_ParseExample(SparseIn64, B, K);   \
  BM_ParseExample(DenseInt64, B, K);   \
  BM_ParseExample(SparseFloat, B, K);  \
  BM_ParseExample(DenseFloat, B, K);

BM_AllParseExample(128, 10);
BM_AllParseExample(128, 100);
BM_AllParseExample(128, 1000);

BM_AllParseExample(512, 10);
BM_AllParseExample(512, 100);
BM_AllParseExample(512, 1000);

}  // end namespace tensorflow

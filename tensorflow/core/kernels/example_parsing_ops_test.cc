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

#include <mutex>
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

typedef std::map<std::tuple<int, int, int>, Tensor> ExampleTensorMap;

// Fillers to fill the underlying repeated array in protobuf.
class BytesFiller {
 public:
  BytesFiller() {}
  void operator()(Feature* f, int feature_size) const {
    for (int i = 0; i < feature_size; ++i) {
      f->mutable_bytes_list()->add_value("abcd1234abcd1234abcd1234abcd1234!");
    }
  }
  Tensor make_dense_default(int feature_size) {
    return Tensor(dtype, TensorShape({feature_size}));
  }
  DataType dtype = DT_STRING;
};

class Int64Filler {
 public:
  Int64Filler() {}
  void operator()(Feature* f, int feature_size) const {
    for (int i = 0; i < feature_size; ++i) {
      f->mutable_int64_list()->add_value(1729);
    }
  }
  Tensor make_dense_default(int feature_size) {
    return Tensor(dtype, TensorShape({feature_size}));
  }
  DataType dtype = DT_INT64;
};

class FloatFiller {
 public:
  FloatFiller() {}
  void operator()(Feature* f, int feature_size) const {
    for (int i = 0; i < feature_size; ++i) {
      f->mutable_float_list()->add_value(1.729);
    }
  }
  Tensor make_dense_default(int feature_size) {
    return Tensor(dtype, TensorShape({feature_size}));
  }
  DataType dtype = DT_FLOAT;
};

template <typename T>
struct ExampleStore {
 private:
  static ExampleTensorMap serialized_example;
  static std::once_flag flags_init;

 public:
  static ExampleTensorMap& GetSerializedExample() {
    std::call_once(flags_init, [] {
      AddExample(&serialized_example, 10, 1, 1);
      AddExample(&serialized_example, 100, 1, 1);
      AddExample(&serialized_example, 1000, 1, 1);
      AddExample(&serialized_example, 10, 128, 1);
      AddExample(&serialized_example, 100, 128, 1);
      AddExample(&serialized_example, 1000, 128, 1);
      AddExample(&serialized_example, 10, 512, 1);
      AddExample(&serialized_example, 100, 512, 1);
      AddExample(&serialized_example, 1000, 512, 1);
      AddExample(&serialized_example, 1, 1, 10);
      AddExample(&serialized_example, 1, 1, 100);
      AddExample(&serialized_example, 1, 1, 1000);
      AddExample(&serialized_example, 1, 1, 10000);
      AddExample(&serialized_example, 1, 1, 100000);
      AddExample(&serialized_example, 1, 1, 1000000);
      AddExample(&serialized_example, 10, 1, 100000);
      AddExample(&serialized_example, 100, 1, 10000);
      AddExample(&serialized_example, 1000, 1, 1000);
    });
    return serialized_example;
  }
  typedef T Filler;
  static void AddExample(ExampleTensorMap* examples, int num_keys,
                         int batch_size, int feature_size) {
    Example example;
    Filler fill;
    Tensor record_string(DT_STRING, TensorShape({batch_size}));
    auto string_t = record_string.vec<string>();
    example.Clear();
    for (int b = 0; b < batch_size; ++b) {
      for (int k = 0; k < num_keys; ++k) {
        string k_str = strings::Printf("feature_%d", k);
        Feature f;
        fill(&f, feature_size);
        Features* features = example.mutable_features();
        (*features->mutable_feature())[k_str] = f;
      }
      CHECK(example.SerializeToString(&string_t(b)));
    }
    (*examples)[std::make_tuple(batch_size, num_keys, feature_size)] =
        record_string;
  }
};
template <typename T>
ExampleTensorMap ExampleStore<T>::serialized_example;
template <typename T>
std::once_flag ExampleStore<T>::flags_init;

template class ExampleStore<BytesFiller>;
template class ExampleStore<Int64Filler>;
template class ExampleStore<FloatFiller>;

enum BenchmarkType { kDense, kSparse, kVarLenDense };

template <typename S, BenchmarkType b_type>
struct BenchmarkOptions {
  int benchmark_type = b_type;
  typedef S Store;
  typename S::Filler filler;
};

template <typename Options>
static Graph* ParseExample(int batch_size, int num_keys, int feature_size) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized = Options::Store::GetSerializedExample()[std::make_tuple(
      batch_size, num_keys, feature_size)];
  Tensor names(DT_STRING, TensorShape({batch_size}));

  std::vector<NodeBuilder::NodeOut> sparse_keys;
  std::vector<NodeBuilder::NodeOut> dense_keys;
  std::vector<NodeBuilder::NodeOut> dense_defaults;
  std::vector<DataType> sparse_types;
  std::vector<PartialTensorShape> dense_shapes;
  Options opt;
  for (int i = 0; i < num_keys; ++i) {
    Tensor key(DT_STRING, TensorShape());
    key.scalar<string>()() = strings::Printf("feature_%d", i);
    switch (opt.benchmark_type) {
      case kDense:
        dense_keys.emplace_back(test::graph::Constant(g, key));
        dense_defaults.emplace_back(test::graph::Constant(
            g, opt.filler.make_dense_default(feature_size)));
        dense_shapes.push_back(PartialTensorShape({feature_size}));
        break;
      case kVarLenDense:
        dense_keys.emplace_back(test::graph::Constant(g, key));
        dense_defaults.emplace_back(
            test::graph::Constant(g, opt.filler.make_dense_default(1)));
        dense_shapes.push_back(PartialTensorShape({-1}));
        break;
      case kSparse:
        sparse_keys.emplace_back(test::graph::Constant(g, key));
        sparse_types.push_back(opt.filler.dtype);
        break;
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

template <typename Options>
static Graph* ParseSingleExample(int num_keys, int feature_size) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized_batch_1 =
      Options::Store::GetSerializedExample()[std::make_tuple(1, num_keys,
                                                             feature_size)];
  Tensor serialized(DT_STRING, TensorShape());
  serialized.scalar<string>()() = serialized_batch_1.vec<string>()(0);

  std::vector<string> sparse_keys;
  std::vector<string> dense_keys;
  std::vector<NodeBuilder::NodeOut> dense_defaults;
  std::vector<DataType> sparse_types;
  std::vector<PartialTensorShape> dense_shapes;
  Options opt;
  for (int i = 0; i < num_keys; ++i) {
    string key = strings::Printf("feature_%d", i);
    switch (opt.benchmark_type) {
      case kDense:
        dense_keys.push_back(key),
            dense_defaults.emplace_back(test::graph::Constant(
                g, opt.filler.make_dense_default(feature_size)));
        dense_shapes.push_back(PartialTensorShape({feature_size}));
        break;
      case kVarLenDense:
        dense_keys.push_back(key),
            dense_defaults.emplace_back(
                test::graph::Constant(g, opt.filler.make_dense_default(1)));
        dense_shapes.push_back(PartialTensorShape({-1}));
        break;
      case kSparse:
        sparse_keys.push_back(key), sparse_types.push_back(opt.filler.dtype);
        break;
    }
  }

  Node* ret;
  TF_EXPECT_OK(NodeBuilder(g->NewName("n"), "ParseSingleExample")
                   .Input(test::graph::Constant(g, serialized))
                   .Input(dense_defaults)
                   .Attr<int64>("num_sparse", sparse_keys.size())
                   .Attr("sparse_keys", sparse_keys)
                   .Attr("sparse_types", sparse_types)
                   .Attr("dense_keys", dense_keys)
                   .Attr("dense_shapes", dense_shapes)
                   .Finalize(g, &ret));

  return g;
}

// Benchmark settings (Sparse, Dense) X (Bytes, Int64, Float)
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kSparse> SparseString;
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kDense> DenseString;
typedef BenchmarkOptions<ExampleStore<BytesFiller>, kVarLenDense>
    VarLenDenseString;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kSparse> SparseInt64;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kDense> DenseInt64;
typedef BenchmarkOptions<ExampleStore<Int64Filler>, kVarLenDense>
    VarLenDenseInt64;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kSparse> SparseFloat;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kDense> DenseFloat;
typedef BenchmarkOptions<ExampleStore<FloatFiller>, kVarLenDense>
    VarLenDenseFloat;

// B == batch_size, K == num_keys. F == feature_size.
// K must be one of 10, 100, 1000
#define BM_ParseExample(TYPE, B, K, F)                                   \
  static void BM_ParseExample##_##TYPE##_##B##_##K##_##F(int iters) {    \
    int64 items_per_iter = static_cast<int64>(B) * K * F;                \
    testing::UseRealTime();                                              \
    testing::ItemsProcessed(static_cast<int64>(iters) * items_per_iter); \
    test::Benchmark("cpu", ParseExample<TYPE>(B, K, F)).Run(iters);      \
  }                                                                      \
  BENCHMARK(BM_ParseExample##_##TYPE##_##B##_##K##_##F);

#define BM_AllParseExample(Type)       \
  BM_ParseExample(Type, 1, 10, 1);     \
  BM_ParseExample(Type, 128, 10, 1);   \
  BM_ParseExample(Type, 512, 10, 1);   \
  BM_ParseExample(Type, 1, 100, 1);    \
  BM_ParseExample(Type, 128, 100, 1);  \
  BM_ParseExample(Type, 512, 100, 1);  \
  BM_ParseExample(Type, 1, 1000, 1);   \
  BM_ParseExample(Type, 128, 1000, 1); \
  BM_ParseExample(Type, 512, 1000, 1); \
  BM_ParseExample(Type, 1, 1, 1000000);

BM_AllParseExample(SparseString);
BM_AllParseExample(DenseString);
BM_AllParseExample(VarLenDenseString);
BM_AllParseExample(SparseInt64);
BM_AllParseExample(DenseInt64);
BM_AllParseExample(VarLenDenseInt64);
BM_AllParseExample(SparseFloat);
BM_AllParseExample(DenseFloat);
BM_AllParseExample(VarLenDenseFloat);

// K == num_keys. F == feature_size.
// K must be one of 10, 100, 1000
#define BM_ParseSingleExample(TYPE, K, F)                                \
  static void BM_ParseSingleExample##_##TYPE##_1_##K##_##F(int iters) {  \
    int64 items_per_iter = K * F;                                        \
    testing::UseRealTime();                                              \
    testing::ItemsProcessed(static_cast<int64>(iters) * items_per_iter); \
    test::Benchmark("cpu", ParseSingleExample<TYPE>(K, F)).Run(iters);   \
  }                                                                      \
  BENCHMARK(BM_ParseSingleExample##_##TYPE##_1_##K##_##F);

#define BM_AllParseSingleExample(Type)     \
  BM_ParseSingleExample(Type, 10, 1);      \
  BM_ParseSingleExample(Type, 100, 1);     \
  BM_ParseSingleExample(Type, 1000, 1);    \
  BM_ParseSingleExample(Type, 1, 10);      \
  BM_ParseSingleExample(Type, 1, 100);     \
  BM_ParseSingleExample(Type, 1, 1000);    \
  BM_ParseSingleExample(Type, 1, 10000);   \
  BM_ParseSingleExample(Type, 1, 100000);  \
  BM_ParseSingleExample(Type, 1, 1000000); \
  BM_ParseSingleExample(Type, 10, 100000); \
  BM_ParseSingleExample(Type, 100, 10000); \
  BM_ParseSingleExample(Type, 1000, 1000);

BM_AllParseSingleExample(SparseString);
BM_AllParseSingleExample(DenseString);
BM_AllParseSingleExample(VarLenDenseString);
BM_AllParseSingleExample(SparseInt64);
BM_AllParseSingleExample(DenseInt64);
BM_AllParseSingleExample(VarLenDenseInt64);
BM_AllParseSingleExample(SparseFloat);
BM_AllParseSingleExample(DenseFloat);
BM_AllParseSingleExample(VarLenDenseFloat);

}  // end namespace tensorflow

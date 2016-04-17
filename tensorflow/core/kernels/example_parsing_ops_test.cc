/* Copyright 2015 Google Inc. All Rights Reserved.

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

struct DenseStringExampleStore {
  static ExampleTensorMap GetSerializedExamples() {
    ExampleTensorMap examples;
    int keys[] = {10, 100, 1000, 10000};
    int batch_sizes[] = {128};
    Example example;
    for (int num_keys : keys) {
      for (int batch_size : batch_sizes) {
        Tensor record_string(DT_STRING, TensorShape({batch_size}));
        auto string_t = record_string.vec<string>();
        example.Clear();
        for (int b = 0; b < batch_size; ++b) {
          for (int k = 0; k < num_keys; ++k) {
            string k_str = strings::Printf("%d", k);
            Feature f;
            f.mutable_bytes_list()->add_value("abc");
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

ExampleTensorMap DenseStringExampleStore::serialized_example =
    DenseStringExampleStore::GetSerializedExamples();

static Graph* ParseDenseStringExample(int batch_size, int num_keys) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor& serialized =
      DenseStringExampleStore::serialized_example[std::make_pair(batch_size,
                                                                 num_keys)];

  Tensor names(DT_STRING, TensorShape({batch_size}));

  std::vector<NodeBuilder::NodeOut> sparse_keys;
  std::vector<NodeBuilder::NodeOut> dense_keys;
  std::vector<NodeBuilder::NodeOut> dense_defaults;
  for (int i = 0; i < num_keys; ++i) {
    Tensor dense_key(DT_STRING, TensorShape());
    dense_key.scalar<string>()() = strings::Printf("%d", i);
    dense_keys.emplace_back(test::graph::Constant(g, dense_key));

    Tensor dense_default(DT_STRING, TensorShape());
    dense_defaults.emplace_back(test::graph::Constant(g, dense_default));
  }

  std::vector<DataType> sparse_types;
  std::vector<TensorShape> dense_shapes(num_keys, TensorShape());

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

// B == batch_size, K == num_keys.  K must be one of 10, 100, 1000, 10000
#define BM_ParseDenseStringExample(B, K)                                 \
  static void BM_ParseDenseStringExample##_##B##_##K(int iters) {        \
    int64 items_per_iter = static_cast<int64>(B) * K;                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * items_per_iter); \
    test::Benchmark("cpu", ParseDenseStringExample(B, K)).Run(iters);    \
  }                                                                      \
  BENCHMARK(BM_ParseDenseStringExample##_##B##_##K);

BM_ParseDenseStringExample(128, 10);
BM_ParseDenseStringExample(128, 100);
BM_ParseDenseStringExample(128, 1000);
BM_ParseDenseStringExample(128, 10000);

}  // end namespace tensorflow

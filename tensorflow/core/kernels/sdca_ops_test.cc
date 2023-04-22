/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

const SessionOptions* GetSingleThreadedOptions() {
  static const SessionOptions* const kSessionOptions = []() {
    SessionOptions* const result = new SessionOptions();
    result->config.set_intra_op_parallelism_threads(1);
    result->config.set_inter_op_parallelism_threads(1);
    result->config.add_session_inter_op_thread_pool()->set_num_threads(1);
    return result;
  }();
  return kSessionOptions;
}

const SessionOptions* GetMultiThreadedOptions() {
  static const SessionOptions* const kSessionOptions = []() {
    SessionOptions* const result = new SessionOptions();
    result->config.set_intra_op_parallelism_threads(0);  // Auto-configured.
    result->config.set_inter_op_parallelism_threads(0);  // Auto-configured.
    result->config.add_session_inter_op_thread_pool()->set_num_threads(
        0);  // Auto-configured.
    return result;
  }();
  return kSessionOptions;
}

Node* Var(Graph* const g, const int n) {
  return test::graph::Var(g, DT_FLOAT, TensorShape({n}));
}

// Returns a vector of size 'nodes' with each node being of size 'node_size'.
std::vector<Node*> VarVector(Graph* const g, const int nodes,
                             const int node_size) {
  std::vector<Node*> result;
  result.reserve(nodes);
  for (int i = 0; i < nodes; ++i) {
    result.push_back(Var(g, node_size));
  }
  return result;
}

Node* Zeros(Graph* const g, const TensorShape& shape) {
  Tensor data(DT_FLOAT, shape);
  data.flat<float>().setZero();
  return test::graph::Constant(g, data);
}

Node* Zeros(Graph* const g, const int n) { return Zeros(g, TensorShape({n})); }

Node* Ones(Graph* const g, const int n) {
  Tensor data(DT_FLOAT, TensorShape({n}));
  test::FillFn<float>(&data, [](const int i) { return 1.0f; });
  return test::graph::Constant(g, data);
}

Node* SparseIndices(Graph* const g, const int sparse_features_per_group) {
  Tensor data(DT_INT64, TensorShape({sparse_features_per_group}));
  test::FillFn<int64>(&data, [&](const int i) { return i; });
  return test::graph::Constant(g, data);
}

Node* SparseExampleIndices(Graph* const g, const int sparse_features_per_group,
                           const int num_examples) {
  const int x_size = num_examples * 4;
  Tensor data(DT_INT64, TensorShape({x_size}));
  test::FillFn<int64>(&data, [&](const int i) { return i / 4; });
  return test::graph::Constant(g, data);
}

Node* SparseFeatureIndices(Graph* const g, const int sparse_features_per_group,
                           const int num_examples) {
  const int x_size = num_examples * 4;
  Tensor data(DT_INT64, TensorShape({x_size}));
  test::FillFn<int64>(
      &data, [&](const int i) { return i % sparse_features_per_group; });
  return test::graph::Constant(g, data);
}

Node* RandomZeroOrOne(Graph* const g, const int n) {
  Tensor data(DT_FLOAT, TensorShape({n}));
  test::FillFn<float>(&data, [](const int i) {
    // Fill with 0.0 or 1.0 at random.
    return (random::New64() % 2) == 0 ? 0.0f : 1.0f;
  });
  return test::graph::Constant(g, data);
}

Node* RandomZeroOrOneMatrix(Graph* const g, const int n, int d) {
  Tensor data(DT_FLOAT, TensorShape({n, d}));
  test::FillFn<float>(&data, [](const int i) {
    // Fill with 0.0 or 1.0 at random.
    return (random::New64() % 2) == 0 ? 0.0f : 1.0f;
  });
  return test::graph::Constant(g, data);
}

void GetGraphs(const int32 num_examples, const int32 num_sparse_feature_groups,
               const int32 sparse_features_per_group,
               const int32 num_dense_feature_groups,
               const int32 dense_features_per_group, Graph** const init_g,
               Graph** train_g) {
  {
    // Build initialization graph
    Graph* g = new Graph(OpRegistry::Global());

    // These nodes have to be created first, and in the same way as the
    // nodes in the graph below.
    std::vector<Node*> sparse_weight_nodes =
        VarVector(g, num_sparse_feature_groups, sparse_features_per_group);
    std::vector<Node*> dense_weight_nodes =
        VarVector(g, num_dense_feature_groups, dense_features_per_group);
    Node* const multi_zero = Zeros(g, sparse_features_per_group);
    for (Node* n : sparse_weight_nodes) {
      test::graph::Assign(g, n, multi_zero);
    }
    Node* const zero = Zeros(g, dense_features_per_group);
    for (Node* n : dense_weight_nodes) {
      test::graph::Assign(g, n, zero);
    }

    *init_g = g;
  }

  {
    // Build execution graph
    Graph* g = new Graph(OpRegistry::Global());

    // These nodes have to be created first, and in the same way as the
    // nodes in the graph above.
    std::vector<Node*> sparse_weight_nodes =
        VarVector(g, num_sparse_feature_groups, sparse_features_per_group);
    std::vector<Node*> dense_weight_nodes =
        VarVector(g, num_dense_feature_groups, dense_features_per_group);

    std::vector<NodeBuilder::NodeOut> sparse_indices;
    std::vector<NodeBuilder::NodeOut> sparse_weights;
    for (Node* n : sparse_weight_nodes) {
      sparse_indices.push_back(
          NodeBuilder::NodeOut(SparseIndices(g, sparse_features_per_group)));
      sparse_weights.push_back(NodeBuilder::NodeOut(n));
    }
    std::vector<NodeBuilder::NodeOut> dense_weights;
    dense_weights.reserve(dense_weight_nodes.size());
    for (Node* n : dense_weight_nodes) {
      dense_weights.push_back(NodeBuilder::NodeOut(n));
    }

    std::vector<NodeBuilder::NodeOut> sparse_example_indices;
    std::vector<NodeBuilder::NodeOut> sparse_feature_indices;
    std::vector<NodeBuilder::NodeOut> sparse_values;
    sparse_example_indices.reserve(num_sparse_feature_groups);
    for (int i = 0; i < num_sparse_feature_groups; ++i) {
      sparse_example_indices.push_back(NodeBuilder::NodeOut(
          SparseExampleIndices(g, sparse_features_per_group, num_examples)));
    }
    sparse_feature_indices.reserve(num_sparse_feature_groups);
    for (int i = 0; i < num_sparse_feature_groups; ++i) {
      sparse_feature_indices.push_back(NodeBuilder::NodeOut(
          SparseFeatureIndices(g, sparse_features_per_group, num_examples)));
    }
    sparse_values.reserve(num_sparse_feature_groups);
    for (int i = 0; i < num_sparse_feature_groups; ++i) {
      sparse_values.push_back(
          NodeBuilder::NodeOut(RandomZeroOrOne(g, num_examples * 4)));
    }

    std::vector<NodeBuilder::NodeOut> dense_features;
    dense_features.reserve(num_dense_feature_groups);
    for (int i = 0; i < num_dense_feature_groups; ++i) {
      dense_features.push_back(NodeBuilder::NodeOut(
          RandomZeroOrOneMatrix(g, num_examples, dense_features_per_group)));
    }

    Node* const weights = Ones(g, num_examples);
    Node* const labels = RandomZeroOrOne(g, num_examples);
    Node* const example_state_data = Zeros(g, TensorShape({num_examples, 4}));

    Node* sdca = nullptr;
    TF_CHECK_OK(
        NodeBuilder(g->NewName("sdca"), "SdcaOptimizer")
            .Attr("loss_type", "logistic_loss")
            .Attr("num_sparse_features", num_sparse_feature_groups)
            .Attr("num_sparse_features_with_values", num_sparse_feature_groups)
            .Attr("num_dense_features", num_dense_feature_groups)
            .Attr("l1", 0.0)
            .Attr("l2", 1.0)
            .Attr("num_loss_partitions", 1)
            .Attr("num_inner_iterations", 2)
            .Input(sparse_example_indices)
            .Input(sparse_feature_indices)
            .Input(sparse_values)
            .Input(dense_features)
            .Input(weights)
            .Input(labels)
            .Input(sparse_indices)
            .Input(sparse_weights)
            .Input(dense_weights)
            .Input(example_state_data)
            .Finalize(g, &sdca));

    *train_g = g;
  }
}

void BM_SDCA(::testing::benchmark::State& state) {
  const int num_examples = state.range(0);
  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 20 /* sparse feature groups */,
            5 /* sparse features per group */, 1 /* dense feature groups*/,
            20 /* dense features per group */, &init, &train);
  test::Benchmark("cpu", train, GetSingleThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
}

void BM_SDCA_LARGE_DENSE(::testing::benchmark::State& state) {
  const int num_examples = state.range(0);

  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 0 /* sparse feature groups */,
            0 /* sparse features per group */, 5 /* dense feature groups*/,
            200000 /* dense features per group */, &init, &train);
  test::Benchmark("cpu", train, GetSingleThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
}

void BM_SDCA_LARGE_SPARSE(::testing::benchmark::State& state) {
  const int num_examples = state.range(0);

  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 65 /* sparse feature groups */,
            1e6 /* sparse features per group */, 0 /* dense feature groups*/,
            0 /* dense features per group */, &init, &train);
  test::Benchmark("cpu", train, GetMultiThreadedOptions(), init, nullptr, "",
                  /*old_benchmark_api*/ false)
      .Run(state);
}
}  // namespace

BENCHMARK(BM_SDCA)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BM_SDCA_LARGE_DENSE)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);
BENCHMARK(BM_SDCA_LARGE_SPARSE)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

}  // namespace tensorflow

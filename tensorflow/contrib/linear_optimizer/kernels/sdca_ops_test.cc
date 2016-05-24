/* Copyright 2016 Google Inc. All Rights Reserved.

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

SessionOptions* GetOptions() {
  static SessionOptions* options = []() {
    // We focus on the single thread performance of training ops.
    SessionOptions* const result = new SessionOptions();
    result->config.set_intra_op_parallelism_threads(1);
    result->config.set_inter_op_parallelism_threads(1);
    return result;
  }();
  return options;
}

Node* Var(Graph* const g, const int n) {
  return test::graph::Var(g, DT_FLOAT, TensorShape({n}));
}

// Returns a vector of size 'nodes' with each node being of size 'node_size'.
std::vector<Node*> VarVector(Graph* const g, const int nodes,
                             const int node_size) {
  std::vector<Node*> result;
  for (int i = 0; i < nodes; ++i) {
    result.push_back(Var(g, node_size));
  }
  return result;
}

Node* Zeros(Graph* const g, const int n) {
  Tensor data(DT_FLOAT, TensorShape({n}));
  data.flat<float>().setZero();
  return test::graph::Constant(g, data);
}

Node* Ones(Graph* const g, const int n) {
  Tensor data(DT_FLOAT, TensorShape({n}));
  test::FillFn<float>(&data, [](const int i) { return 1.0f; });
  return test::graph::Constant(g, data);
}

Node* StringIota(Graph* const g, const int n) {
  Tensor data(DT_STRING, TensorShape({n}));
  test::FillFn<string>(
      &data, [](const int i) { return strings::StrCat(strings::Hex(i)); });
  return test::graph::Constant(g, data);
}

Node* SparseIndices(Graph* const g, const int sparse_features_per_group,
                    const int num_examples) {
  const int x_size = num_examples * 4;
  const int y_size = 2;
  Tensor data(DT_INT64, TensorShape({x_size, y_size}));
  test::FillFn<int64>(&data, [&](const int i) {
    // Convert FillFn index 'i', to (x,y) for this tensor.
    const int x = i % y_size;
    const int y = i / y_size;
    if (y == 0) {
      // Populate example index with 4 features per example.
      return x / 4;
    } else {
      // Assign feature indices sequentially - 0,1,2,3 for example 0,
      // 4,5,6,7 for example 1,....  Wrap back around when we hit
      // num_sparse-features.
      return x % sparse_features_per_group;
    }
  });
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

void GetGraphs(const int32 num_examples, const int32 sparse_feature_groups,
               const int32 sparse_features_per_group,
               const int32 dense_feature_groups, Graph** const init_g,
               Graph** train_g) {
  {
    // Build initialization graph
    Graph* g = new Graph(OpRegistry::Global());

    // These nodes have to be created first, and in the same way as the
    // nodes in the graph below.
    std::vector<Node*> sparse_weight_nodes =
        VarVector(g, sparse_feature_groups, sparse_features_per_group);
    std::vector<Node*> dense_weight_nodes =
        VarVector(g, dense_feature_groups, 1);
    Node* const multi_zero = Zeros(g, sparse_features_per_group);
    for (Node* n : sparse_weight_nodes) {
      test::graph::Assign(g, n, multi_zero);
    }
    Node* const zero = Zeros(g, 1);
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
        VarVector(g, sparse_feature_groups, sparse_features_per_group);
    std::vector<Node*> dense_weight_nodes =
        VarVector(g, dense_feature_groups, 1);

    std::vector<NodeBuilder::NodeOut> sparse_weights;
    for (Node* n : sparse_weight_nodes) {
      sparse_weights.push_back(NodeBuilder::NodeOut(n));
    }
    std::vector<NodeBuilder::NodeOut> dense_weights;
    for (Node* n : dense_weight_nodes) {
      dense_weights.push_back(NodeBuilder::NodeOut(n));
    }

    std::vector<NodeBuilder::NodeOut> sparse_indices;
    std::vector<NodeBuilder::NodeOut> sparse_values;
    for (int i = 0; i < sparse_feature_groups; ++i) {
      sparse_indices.push_back(NodeBuilder::NodeOut(
          SparseIndices(g, sparse_features_per_group, num_examples)));
    }
    for (int i = 0; i < sparse_feature_groups; ++i) {
      sparse_values.push_back(
          NodeBuilder::NodeOut(RandomZeroOrOne(g, num_examples * 4)));
    }

    std::vector<NodeBuilder::NodeOut> dense_features;
    for (int i = 0; i < dense_feature_groups; ++i) {
      dense_features.push_back(
          NodeBuilder::NodeOut(RandomZeroOrOne(g, num_examples)));
    }

    Node* const weights = Ones(g, num_examples);
    Node* const labels = RandomZeroOrOne(g, num_examples);
    Node* const ids = StringIota(g, num_examples);

    Node* sdca = nullptr;
    TF_CHECK_OK(
        NodeBuilder(g->NewName("sdca"), "SdcaSolver")
            .Attr("loss_type", "logistic_loss")
            .Attr("num_sparse_features", sparse_feature_groups)
            .Attr("num_dense_features", dense_feature_groups)
            .Attr("l1", 0.0)
            .Attr("l2", 1.0)
            .Attr("num_inner_iterations", 2)
            .Attr("container", strings::StrCat(strings::Hex(random::New64())))
            .Attr("solver_uuid", strings::StrCat(strings::Hex(random::New64())))
            .Input(sparse_indices)
            .Input(sparse_values)
            .Input(dense_features)
            .Input(weights)
            .Input(labels)
            .Input(ids)
            .Input(sparse_weights)
            .Input(dense_weights)
            .Finalize(g, &sdca));

    *train_g = g;
  }
}

void BM_SDCA(const int iters, const int num_examples) {
  testing::StopTiming();
  Graph* init = nullptr;
  Graph* train = nullptr;
  GetGraphs(num_examples, 20 /* sparse feature groups */,
            5 /* sparse features per group */, 20 /* dense features */, &init,
            &train);
  testing::StartTiming();
  test::Benchmark("cpu", train, GetOptions(), init).Run(iters);
  // TODO(sibyl-toe9oF2e):  Each all to Run() currently creates a container which
  // gets deleted as the context gets deleted.  It would be nicer to
  // explicitly clean up the container ourselves at this point (after calling
  // testing::StopTiming).
}

}  // namespace

BENCHMARK(BM_SDCA)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

}  // namespace tensorflow

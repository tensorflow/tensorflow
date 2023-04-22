/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/platform/stringprintf.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

using grappler::graph_utils::GetScalarConstNodeValue;

constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kExperimentalMapAndBatchOp[] = "ExperimentalMapAndBatchDataset";

constexpr std::array<const char*, 4> kBatchDatasetOps = {
    "BatchDataset",
    "PaddedBatchDataset",
    kMapAndBatchOp,
    kExperimentalMapAndBatchOp,
};

constexpr std::array<const char*, 2> kMultipleInputDatasetOps = {
    "ConcatenateDataset",
    "ZipDataset",
};

constexpr std::array<const char*, 16> kPassThroughOps = {
    "AssertCardinalityDataset",
    "CacheDataset",
    "FilterDataset",
    "FinalizeDataset",
    "Identity",
    "ModelDataset",
    "OptimizeDataset",
    "OptionsDataset",
    "ParseExampleDataset",
    "PrefetchDataset",
    "RepeatDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "SkipDataset",
    "TakeDataset",
};

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
  for (const auto& dataset_op : arr) {
    if (MatchesAnyVersion(dataset_op, node.op())) return true;
  }
  return false;
}

const NodeDef* GetInputNode(const NodeDef& node,
                            const grappler::GraphView& graph,
                            int64 input_index) {
  if (node.input_size() == 0) return nullptr;
  grappler::GraphView::InputPort input_port =
      graph.GetInputPort(node.name(), input_index);
  return graph.GetRegularFanin(input_port).node;
}

// TODO(rachelim): This op traverses the dataset graph using a allowlist-based
// approach. As an alternative, we could instead rewrite all batching datasets'
// drop_remainder parameter to True, then rerun the dataset graph to derive
// new output shapes using C++ shape inference. This is more robust in cases
// where datasets have shape inference implemented in C++. If this allowlist-
// based approach proves hard to maintain, consider doing the alternative.
class ComputeBatchSizeOp : public OpKernel {
 public:
  explicit ComputeBatchSizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));

    std::vector<std::pair<string, Tensor>> input_list;
    GraphDef graph_def;
    string dataset_node_name;
    OP_REQUIRES_OK(ctx, AsGraphDefMinimal(ctx, dataset, &input_list, &graph_def,
                                          &dataset_node_name));

    // Create GraphView for easier traversal of graph.
    grappler::GraphView graph_view(&graph_def);

    const NodeDef* node = graph_view.GetNode(dataset_node_name);
    OP_REQUIRES(ctx, node != nullptr,
                errors::InvalidArgument("Node does not exist in graph"));
    int64 batch_size = GetBatchSize(*node, graph_view);
    Tensor* result;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &result));
    result->scalar<int64>()() = batch_size;
  }

 private:
  int64 GetBatchSizeFromBatchNode(const NodeDef& node,
                                  const grappler::GraphView& graph) {
    int64 arg_index;
    if (node.op() == kMapAndBatchOp ||
        node.op() == kExperimentalMapAndBatchOp) {
      arg_index = node.input_size() - 3;
    } else {
      arg_index = 1;
    }

    auto batch_size_node = GetInputNode(node, graph, arg_index);
    int64 batch_size;
    auto s = GetScalarConstNodeValue(*batch_size_node, &batch_size);
    if (!s.ok()) {
      VLOG(1) << "Could not compute static batch size. Found batching dataset ("
              << node.name() << "), but failed to get its input batch size: "
              << s.error_message();
      return -1;
    }
    return batch_size;
  }

  // Helper function that returns the static 0th dimension of a given dataset
  // node in the graph. It starts from a node in the graph and recursively
  // traverses its inputs until it finds a valid BatchDataset operation,
  // and returns its batch size. If the batch size cannot be determined,
  // returns -1.
  //
  // During recursion, it handles four kinds of cases:
  // 1. BatchDataset type ops: Returns the value from its batch_size input node.
  // 2. Zip / Concatenate dataset ops: Recurses into all inputs to these ops,
  //    which are themselves all datasets, and returns the batch sizes computed
  //    by the inputs if they are all the same.
  // 3. Core dataset ops which cannot change the size of the 0th dimension of
  //    dataset output elements: Recurses into the first input parameter.
  // 4. All other ops: Fail, returning -1 for unknown.
  // TODO(rachelim): For FlatMap type mapping dataset ops, recurse into the
  // function definition.
  int64 GetBatchSize(const NodeDef& node, const grappler::GraphView& graph) {
    if (IsDatasetNodeOfType(node, kBatchDatasetOps)) {
      return GetBatchSizeFromBatchNode(node, graph);
    }
    if (IsDatasetNodeOfType(node, kMultipleInputDatasetOps)) {
      const NodeDef* input_0 = GetInputNode(node, graph, 0);
      int64 batch_size_0 = GetBatchSize(*input_0, graph);
      for (int i = 1; i < node.input_size(); ++i) {
        const NodeDef* input = GetInputNode(node, graph, i);
        auto batch_size_i = GetBatchSize(*input, graph);
        if (batch_size_i != batch_size_0) {
          VLOG(1) << "Could not compute batch size: inputs to " << node.name()
                  << " (" << node.op() << ") had different batch sizes."
                  << " Namely, input 0 had batch size " << batch_size_0
                  << " while input " << i << " had batch size " << batch_size_i
                  << ".";
          return -1;
        }
      }
      return batch_size_0;
    }
    if (IsDatasetNodeOfType(node, kPassThroughOps)) {
      const NodeDef* input = GetInputNode(node, graph, 0);
      return GetBatchSize(*input, graph);
    }
    VLOG(1) << "Encountered dataset node " << node.name() << " (" << node.op()
            << ") that prevented further static batch size analysis.";

    return -1;
  }
};

REGISTER_KERNEL_BUILDER(Name("ComputeBatchSize").Device(DEVICE_CPU),
                        ComputeBatchSizeOp);

}  // anonymous namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

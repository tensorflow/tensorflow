/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"

#include <array>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/kernels/data/shard_dataset_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace grappler {
namespace {

using tensorflow::data::AutoShardPolicy;

constexpr char kAssertCardinalityDatasetOpName[] = "AssertCardinalityDataset";
constexpr char kBatchDatasetOpName[] = "BatchDataset";
constexpr char kBatchDatasetV2OpName[] = "BatchDatasetV2";
constexpr char kMapAndBatchDatasetOpName[] = "MapAndBatchDataset";
constexpr char kMapDatasetOpName[] = "MapDataset";
constexpr char kShardDatasetOpName[] = "ShardDataset";
constexpr char kShuffleDatasetOpName[] = "ShuffleDataset";
constexpr char kShuffleDatasetV2OpName[] = "ShuffleDatasetV2";
constexpr char kShuffleDatasetV3OpName[] = "ShuffleDatasetV3";
constexpr char kParallelBatchDatasetOpName[] = "ParallelBatchDataset";
constexpr char kPrefetchDatasetOpName[] = "PrefetchDataset";
constexpr char kFinalizeDatasetOpName[] = "FinalizeDataset";
constexpr char kOptionsDatasetOpName[] = "OptionsDataset";
constexpr char kRebatchDatasetOpName[] = "RebatchDataset";
constexpr char kRebatchDatasetV2OpName[] = "RebatchDatasetV2";
constexpr char kTensorDatasetOpName[] = "TensorDataset";
constexpr char kTensorSliceDatasetOpName[] = "TensorSliceDataset";
constexpr char kPlaceholderOpName[] = "Placeholder";
constexpr char kConstOpName[] = "Const";

constexpr char kNumWorkersAttrName[] = "num_workers";
constexpr char kNumReplicasAttrName[] = "num_replicas";
constexpr char kIndexAttrName[] = "index";
constexpr char kAutoShardPolicyAttrName[] = "auto_shard_policy";
constexpr char kReshuffleEachIteration[] = "reshuffle_each_iteration";
constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";

// clang-format off
constexpr std::array<const char*, 6> kReaderDatasetOps = {
    "ArrayRecordDataset",
    "FixedLengthRecordDataset",
    "RecordIODataset",
    "SSTableDataset",
    "TextLineDataset",
    "TFRecordDataset"
};

constexpr std::array<const char*, 2> kMultipleInputsDatasetOps = {
    "ConcatenateDataset",
    "ZipDataset"
};

constexpr std::array<const char*, 31> kPassThroughOps = {
    "_Retval",
    "AssertNextDataset",
    "BatchDataset",
    "CacheDataset",
    "ExperimentalMapAndBatchDataset",
    "ExperimentalParseExampleDataset",
    "ExperimentalRebatchDataset",
    "FilterDataset",
    "FinalizeDataset",
    "Identity",
    "MapAndBatchDataset",
    "MapDataset",
    "MaxIntraOpParallelismDataset",
    "ModelDataset",
    "OptimizeDataset",
    "OptionsDataset",
    "PaddedBatchDataset",
    "ParallelBatchDataset",
    "ParallelMapDataset",
    "ParseExampleDataset",
    "PrefetchDataset",
    "PrivateThreadPoolDataset",
    "ReduceDataset",
    "RebatchDataset",
    "RepeatDataset",
    "ShardDataset",
    "ShuffleAndRepeatDataset",
    "ShuffleDataset",
    "SkipDataset",
    "TakeDataset",
    "WindowDataset",
};

// TODO(frankchn): Process functions within kFuncDatasetOps as well.
constexpr std::array<const char*, 5> kFuncDatasetOps = {
    "ExperimentalParallelInterleaveDataset",
    "FlatMapDataset",
    "InterleaveDataset",
    "LegacyParallelInterleaveDataset",
    "ParallelInterleaveDataset",
};

constexpr std::array<const char*, 5> kUnshardableSourceDatasetOps = {
    "GeneratorDataset",
    "RangeDataset",
    "SparseTensorsSliceDataset",
    "TensorDataset",
    "TensorSliceDataset",
};

// The semantics of these ops are not affected by the change of the batch
// size. There are three categories:
//   1. The op doesn't change the elements of the dataset, e.g. CacheDataset and
//   all ops that sets options.
//   2. The op is dataset-element-wise transformation which is orthogonoal to
//   the batch size, e.g. ParseExampleDataset.
//   3. RebatchDataset. This is a special case. RebatchDataset is added by
//   tf.distribute at the end of the input pipeline and will be specially
//   handled.
constexpr std::array<const char*, 20> kBatchSizeOrthogonalDatasetOps = {
    "AssertCardinalityDataset",
    "AssertNextDataset",
    "BytesProducedStatsDataset",
    "CacheDataset",
    "FinalizeDataset",
    "Identity",
    "LatencyStatsDataset",
    "MaxIntraOpParallelismDataset",
    "ModelDataset",
    "NonSerializableDataset",
    "OptimizeDataset",
    "OptionsDataset",
    "ParseExampleDataset",
    "PrefetchDataset",
    "PrivateThreadPoolDataset",
    "RebatchDataset",
    "RepeatDataset",
    "SetStatsAggregatorDataset",
    "SleepDataset",
    "ThreadPoolDataset",
};

constexpr std::array<const char*, 3> kBatchDatasetOps = {
    kBatchDatasetOpName,
    kMapAndBatchDatasetOpName,
    kParallelBatchDatasetOpName,
};

// clang-format on

Status OptimizeGraph(const GrapplerItem& item, int64_t num_workers,
                     int64_t index, AutoShardPolicy policy,
                     int64_t num_replicas, GraphDef* output,
                     AutoShardPolicy* policy_applied);

template <std::size_t SIZE>
bool IsDatasetNodeOfType(const NodeDef& node,
                         const std::array<const char*, SIZE>& arr) {
  for (const auto& dataset_op_name : arr) {
    if (tensorflow::data::MatchesAnyVersion(/*op_prefix=*/dataset_op_name,
                                            /*op_to_match=*/node.op())) {
      return true;
    }
  }
  return false;
}

// Adds a ShardDataset node before `add_before`.
Status AddShardNode(MutableGraphView* graph, const NodeDef& add_before,
                    int64_t num_workers, int64_t index) {
  NodeDef new_node;
  new_node.set_op(kShardDatasetOpName);
  graph_utils::SetUniqueGraphNodeName(kShardDatasetOpName, graph->graph(),
                                      &new_node);

  // Construct argument nodes
  NodeDef* num_shards_node =
      graph_utils::AddScalarConstNode<int64_t>(num_workers, graph);
  NodeDef* index_node = graph_utils::AddScalarConstNode<int64_t>(index, graph);

  // Add inputs to new node
  new_node.add_input(add_before.input(0));
  new_node.add_input(num_shards_node->name());
  new_node.add_input(index_node->name());

  // Ensure that each shard will have at least one element.
  (*(new_node.mutable_attr()))[data::ShardDatasetOp::kRequireNonEmpty].set_b(
      true);

  // Add shapes and other attributes
  NodeDef* add_after = graph->GetNode(add_before.input(0));

  if (absl::StrContains(add_after->op(), "Dataset")) {
    // We still may or may not have the right attributes because Datasets like
    // TFRecordDataset doesn't have a output type or shape, and by default we
    // set them to DT_STRING and an unknown shape.
    if (add_after->attr().count(kOutputShapes) > 0) {
      graph_utils::CopyAttribute(kOutputShapes, *add_after, &new_node);
    } else {
      tensorflow::TensorShapeProto* shape =
          (*(new_node.mutable_attr()))[kOutputShapes]
              .mutable_list()
              ->add_shape();
      shape->set_unknown_rank(true);
    }

    if (add_after->attr().count(kOutputTypes) > 0) {
      graph_utils::CopyAttribute(kOutputTypes, *add_after, &new_node);
    } else if (add_after->attr().count("Toutput_types") > 0) {
      (*(new_node.mutable_attr()))[kOutputTypes] =
          add_after->attr().at("Toutput_types");
    } else {
      (*(new_node.mutable_attr()))[kOutputTypes].mutable_list()->add_type(
          tensorflow::DataType::DT_STRING);
    }
  } else {
    // TODO(frankchn): Make this work for datasets where input(0) is a Const,
    // and we need to shard the Const.
    // This is probably not a dataset, so we bail because we can't infer the
    // output types and shape.
    return errors::NotFound(
        "Unable to shard this input. You may need to wrap the inputs to your "
        "reader dataset in a TensorSliceDataset. Input node is ",
        add_after->DebugString());
  }

  // Add new node into graph and update edges
  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));
  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));

  return OkStatus();
}

Status AddShuffleDataset(MutableGraphView* graph, const NodeDef& add_before,
                         const string& buffer_size_node,
                         const string& seed_node, const string& seed2_node,
                         bool reshuffle_each_iteration) {
  NodeDef* add_after = graph->GetNode(add_before.input(0));
  NodeDef new_node;
  new_node.set_op(kShuffleDatasetOpName);
  graph_utils::SetUniqueGraphNodeName(kShuffleDatasetOpName, graph->graph(),
                                      &new_node);

  new_node.add_input(add_before.input(0));
  new_node.add_input(buffer_size_node);
  new_node.add_input(seed_node);
  new_node.add_input(seed2_node);

  graph_utils::CopyAttribute(kOutputShapes, *add_after, &new_node);
  graph_utils::CopyAttribute(kOutputTypes, *add_after, &new_node);

  AttrValue reshuffle_attr;
  reshuffle_attr.set_b(reshuffle_each_iteration);
  (*new_node.mutable_attr())[kReshuffleEachIteration] = reshuffle_attr;

  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));
  return OkStatus();
}

Status AddShuffleDatasetV2(MutableGraphView* graph, const NodeDef& add_before,
                           const string& buffer_size_node,
                           const string& seed_generator_node) {
  NodeDef* add_after = graph->GetNode(add_before.input(0));
  NodeDef new_node;
  new_node.set_op(kShuffleDatasetV2OpName);
  graph_utils::SetUniqueGraphNodeName(kShuffleDatasetV2OpName, graph->graph(),
                                      &new_node);

  new_node.add_input(add_before.input(0));
  new_node.add_input(buffer_size_node);
  new_node.add_input(seed_generator_node);

  graph_utils::CopyAttribute(kOutputShapes, *add_after, &new_node);
  graph_utils::CopyAttribute(kOutputTypes, *add_after, &new_node);

  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));
  return OkStatus();
}

Status AddShuffleDatasetV3(MutableGraphView* graph, const NodeDef& add_before,
                           const string& buffer_size_node,
                           const string& seed_node, const string& seed2_node,
                           const string& seed_generator_node,
                           bool reshuffle_each_iteration) {
  NodeDef* add_after = graph->GetNode(add_before.input(0));
  NodeDef new_node;
  new_node.set_op(kShuffleDatasetV3OpName);
  graph_utils::SetUniqueGraphNodeName(kShuffleDatasetV3OpName, graph->graph(),
                                      &new_node);

  new_node.add_input(add_before.input(0));
  new_node.add_input(buffer_size_node);
  new_node.add_input(seed_node);
  new_node.add_input(seed2_node);
  new_node.add_input(seed_generator_node);

  graph_utils::CopyAttribute(kOutputShapes, *add_after, &new_node);
  graph_utils::CopyAttribute(kOutputTypes, *add_after, &new_node);

  AttrValue reshuffle_attr;
  reshuffle_attr.set_b(reshuffle_each_iteration);
  (*new_node.mutable_attr())[kReshuffleEachIteration] = reshuffle_attr;

  NodeDef* new_node_graph = graph->AddNode(std::move(new_node));

  TF_RETURN_IF_ERROR(
      graph->UpdateFanouts(add_after->name(), new_node_graph->name()));
  return OkStatus();
}

bool ReaderOpInFunction(const NodeDef& node,
                        const FunctionLibraryDefinition& flib) {
  auto f_attr_it = node.attr().find("f");
  if (f_attr_it == node.attr().end()) return false;
  const FunctionDef* func = flib.Find(f_attr_it->second.func().name());
  for (int i = 0; i < func->node_def_size(); i++) {
    NodeDef node_in_func = func->node_def(i);
    if (IsDatasetNodeOfType(node_in_func, kReaderDatasetOps) &&
        node_in_func.input_size() > 0) {
      return true;
    }
    if (IsDatasetNodeOfType(func->node_def(i), kFuncDatasetOps) &&
        ReaderOpInFunction(func->node_def(i), flib)) {
      return true;
    }
  }
  return false;
}

Status RemoveShuffleDataset(MutableGraphView* graph, const NodeDef& node,
                            absl::flat_hash_set<string>* nodes_to_delete,
                            string* op_name, string* buffer_size_node,
                            string* seed_node, string* seed2_node,
                            bool* reshuffle_each_iteration) {
  if (node.op() == kShuffleDatasetOpName) {
    *op_name = node.op();
    *buffer_size_node = node.input(1);
    *seed_node = node.input(2);
    *seed2_node = node.input(3);
    *reshuffle_each_iteration = node.attr().at(kReshuffleEachIteration).b();
    TF_RETURN_IF_ERROR(graph->UpdateFanouts(node.name(), node.input(0)));
    nodes_to_delete->insert(node.name());
  }

  for (const auto& fanin : graph->GetFanins(node, true)) {
    TF_RETURN_IF_ERROR(RemoveShuffleDataset(
        graph, *fanin.node, nodes_to_delete, op_name, buffer_size_node,
        seed_node, seed2_node, reshuffle_each_iteration));
  }

  // TODO(frankchn): Traverse functions too.
  return OkStatus();
}

Status RemoveShuffleDatasetV2(MutableGraphView* graph, const NodeDef& node,
                              absl::flat_hash_set<string>* nodes_to_delete,
                              string* op_name, string* buffer_size_node,
                              string* seed_generator_node) {
  if (node.op() == kShuffleDatasetV2OpName) {
    *op_name = node.op();
    *buffer_size_node = node.input(1);
    *seed_generator_node = node.input(2);
    TF_RETURN_IF_ERROR(graph->UpdateFanouts(node.name(), node.input(0)));
    nodes_to_delete->insert(node.name());
  }

  for (const auto& fanin : graph->GetFanins(node, true)) {
    TF_RETURN_IF_ERROR(
        RemoveShuffleDatasetV2(graph, *fanin.node, nodes_to_delete, op_name,
                               buffer_size_node, seed_generator_node));
  }

  // TODO(frankchn): Traverse functions too.
  return OkStatus();
}

Status RemoveShuffleDatasetV3(MutableGraphView* graph, const NodeDef& node,
                              absl::flat_hash_set<string>* nodes_to_delete,
                              string* op_name, string* buffer_size_node,
                              string* seed_node, string* seed2_node,
                              string* seed_generator_node,
                              bool* reshuffle_each_iteration) {
  if (node.op() == kShuffleDatasetV3OpName) {
    *op_name = node.op();
    *buffer_size_node = node.input(1);
    *seed_node = node.input(2);
    *seed2_node = node.input(3);
    *seed_generator_node = node.input(4);
    *reshuffle_each_iteration = node.attr().at(kReshuffleEachIteration).b();
    TF_RETURN_IF_ERROR(graph->UpdateFanouts(node.name(), node.input(0)));
    nodes_to_delete->insert(node.name());
  }

  for (const auto& fanin : graph->GetFanins(node, true)) {
    TF_RETURN_IF_ERROR(RemoveShuffleDatasetV3(
        graph, *fanin.node, nodes_to_delete, op_name, buffer_size_node,
        seed_node, seed2_node, seed_generator_node, reshuffle_each_iteration));
  }

  // TODO(frankchn): Traverse functions too.
  return OkStatus();
}

Status ProcessDatasetSourceNode(MutableGraphView* graph, const NodeDef& node,
                                absl::flat_hash_set<string>* nodes_to_delete,
                                int64_t num_workers, int64_t index) {
  string shuffle_op_name = "";
  string buffer_size_node = "";
  string seed_node = "";
  string seed2_node = "";
  string seed_generator_node = "";
  bool reshuffle_each_iteration;

  TF_RETURN_IF_ERROR(AddShardNode(graph, node, num_workers, index));
  TF_RETURN_IF_ERROR(RemoveShuffleDataset(
      graph, node, nodes_to_delete, &shuffle_op_name, &buffer_size_node,
      &seed_node, &seed2_node, &reshuffle_each_iteration));
  if (shuffle_op_name.empty()) {
    TF_RETURN_IF_ERROR(
        RemoveShuffleDatasetV2(graph, node, nodes_to_delete, &shuffle_op_name,
                               &buffer_size_node, &seed_generator_node));
  }
  if (shuffle_op_name.empty()) {
    TF_RETURN_IF_ERROR(RemoveShuffleDatasetV3(
        graph, node, nodes_to_delete, &shuffle_op_name, &buffer_size_node,
        &seed_node, &seed2_node, &seed_generator_node,
        &reshuffle_each_iteration));
  }

  if (shuffle_op_name == kShuffleDatasetOpName) {
    TF_RETURN_IF_ERROR(AddShuffleDataset(graph, node, buffer_size_node,
                                         seed_node, seed2_node,
                                         reshuffle_each_iteration));
  } else if (shuffle_op_name == kShuffleDatasetV2OpName) {
    TF_RETURN_IF_ERROR(AddShuffleDatasetV2(graph, node, buffer_size_node,
                                           seed_generator_node));
  } else if (shuffle_op_name == kShuffleDatasetV3OpName) {
    TF_RETURN_IF_ERROR(AddShuffleDatasetV3(
        graph, node, buffer_size_node, seed_node, seed2_node,
        seed_generator_node, reshuffle_each_iteration));
  }

  return OkStatus();
}

const NodeDef* FindFuncAndTensorSliceDataset(
    const NodeDef* node, int64_t num_workers, int64_t index,
    FunctionLibraryDefinition* flib, MutableGraphView* graph,
    absl::flat_hash_set<string>* nodes_to_delete) {
  if (IsDatasetNodeOfType(*node, kFuncDatasetOps)) {
    const NodeDef* input_node = graph_utils::GetInputNode(*node, *graph, 0);
    if (input_node->op() == kTensorSliceDatasetOpName ||
        input_node->op() == kTensorDatasetOpName) {
      const NodeDef* next_input_node =
          graph_utils::GetInputNode(*input_node, *graph, 0);
      if (next_input_node->op() == kPlaceholderOpName) {
        return node;
      }
    }
  }

  if (!IsDatasetNodeOfType(*node, kPassThroughOps)) {
    return nullptr;
  }

  // Sometimes there are other nodes between the last InterleaveDataset and the
  // second to last FlatMapDataset, so we need to skip over those.
  const NodeDef* input_node = graph_utils::GetInputNode(*node, *graph, 0);
  return FindFuncAndTensorSliceDataset(input_node, num_workers, index, flib,
                                       graph, nodes_to_delete);
}

enum class DropRemainderValue { kUnknown, kTrue, kFalse };

DropRemainderValue GetDropRemainder(const MutableGraphView& graph,
                                    const NodeDef& batch_node) {
  const NodeDef* drop_remainder = nullptr;
  if (batch_node.op() == kBatchDatasetOpName ||
      batch_node.op() == kBatchDatasetV2OpName) {
    drop_remainder = graph.GetNode(batch_node.input(2));
  } else if (batch_node.op() == kParallelBatchDatasetOpName) {
    drop_remainder = graph.GetNode(batch_node.input(3));
  } else if (batch_node.op() == kMapAndBatchDatasetOpName) {
    int drop_remainder_index =
        3 + batch_node.attr().at("Targuments").list().shape_size();
    if (drop_remainder_index >= batch_node.input_size()) {
      LOG(ERROR) << "Fail to find the drop_remainder of op: "
                 << batch_node.DebugString();
      return DropRemainderValue::kUnknown;
    }
    drop_remainder = graph.GetNode(batch_node.input(drop_remainder_index));
  } else {
    LOG(ERROR) << "Expect a batch node but get " << batch_node.DebugString();
    return DropRemainderValue::kUnknown;
  }
  if (!IsConstant(*drop_remainder)) {
    return DropRemainderValue::kUnknown;
  }
  bool drop_remainder_value;
  if (!GetNodeAttr(*drop_remainder, "value", &drop_remainder_value).ok()) {
    return DropRemainderValue::kUnknown;
  }
  return drop_remainder_value ? DropRemainderValue::kTrue
                              : DropRemainderValue::kFalse;
}

Status RecursivelyHandleOp(const NodeDef& node, int64_t num_workers,
                           int64_t index, FunctionLibraryDefinition* flib,
                           MutableGraphView* graph,
                           absl::flat_hash_set<string>* nodes_to_delete) {
  if (node.op() == kAssertCardinalityDatasetOpName) {
    LOG(WARNING) << "The `assert_cardinality` transformation is currently not "
                    "handled by the auto-shard rewrite and will be removed.";
    nodes_to_delete->insert(node.name());
    TF_RETURN_IF_ERROR(graph->UpdateFanouts(node.name(), node.input(0)));
    const NodeDef* input_node = graph_utils::GetInputNode(node, *graph, 0);
    return RecursivelyHandleOp(*input_node, num_workers, index, flib, graph,
                               nodes_to_delete);
  }

  if (IsDatasetNodeOfType(node, kUnshardableSourceDatasetOps)) {
    return errors::NotFound("Found an unshardable source dataset: ",
                            node.DebugString());
  }

  if (IsDatasetNodeOfType(node, kMultipleInputsDatasetOps)) {
    for (int i = 0; i < node.input_size(); ++i) {
      const NodeDef* input_node = graph_utils::GetInputNode(node, *graph, i);
      TF_RETURN_IF_ERROR(RecursivelyHandleOp(*input_node, num_workers, index,
                                             flib, graph, nodes_to_delete));
    }
    return OkStatus();
  }

  // This handles the case for the following subgraph:
  //   Placeholder -> TensorSliceDataset -> FlatMapDataset -x->
  //   (other preprocessing datasets) -> InterleaveDataset
  // and then inserting the shard node immediately after the FlatMapDataset.
  //
  // This is used for some training pipelines where a dataset is created with
  // the following code:
  //
  // def make_dataset_pipeline():
  //   file_globs = [...]
  //   datasets = []
  //   for file_glob in file_globs:
  //     datasets.append(Dataset.list_files(file_glob).map(TFRecordReader))
  //   dataset = Dataset.from_tensor_slices(datasets)
  //   dataset = dataset.flat_map(lambda x: x)
  //   dataset = ...  # additional preprocessing
  //   dataset = dataset.interleave(lambda x: x, cycle_length=...)
  //   return dataset
  if (IsDatasetNodeOfType(node, kFuncDatasetOps)) {
    const NodeDef* input_node = graph_utils::GetInputNode(node, *graph, 0);
    const NodeDef* flat_map_node = FindFuncAndTensorSliceDataset(
        input_node, num_workers, index, flib, graph, nodes_to_delete);

    if (flat_map_node != nullptr) {
      auto fanouts = graph->GetFanouts(*flat_map_node, false);
      // FlatMapDataset should only be the input to one other dataset.
      if (fanouts.size() == 1) {
        return ProcessDatasetSourceNode(graph, *fanouts.begin()->node,
                                        nodes_to_delete, num_workers, index);
      }
    }
  }

  // This handles the case where a reader Dataset is contained within a
  // FuncDataset (e.g. FlatMap, ParallelInterleave, etc...) or within a
  // PassThrough input to a FuncDataset. For example:
  //
  // dataset = Dataset.list_files(...)
  // dataset = dataset.flat_map(core_readers.TFRecordDataset)
  //
  // or
  //
  // dataset = Dataset.list_files(...)
  // dataset = dataset.map(core_readers.TFRecordDataset)
  // dataset = dataset.interleave(lambda x: x, cycle_length=3)
  if ((IsDatasetNodeOfType(node, kFuncDatasetOps) ||
       IsDatasetNodeOfType(node, kPassThroughOps)) &&
      ReaderOpInFunction(node, *flib)) {
    return ProcessDatasetSourceNode(graph, node, nodes_to_delete, num_workers,
                                    index);
  }

  if (IsDatasetNodeOfType(node, kReaderDatasetOps)) {
    // We reached a reader dataset directly and we try to shard input 0.
    return ProcessDatasetSourceNode(graph, node, nodes_to_delete, num_workers,
                                    index);
  }

  if (!IsDatasetNodeOfType(node, kFuncDatasetOps) &&
      !IsDatasetNodeOfType(node, kPassThroughOps)) {
    return errors::NotFound(
        "Did not find a shardable source, walked to ",
        "a node which is not a dataset: ", node.DebugString(),
        ". Consider either turning off auto-sharding or switching the "
        "auto_shard_policy to DATA to shard this dataset. You can do this by "
        "creating a new `tf.data.Options()` object then setting "
        "`options.experimental_distribute.auto_shard_policy = "
        "AutoShardPolicy.DATA` before applying the options object to the "
        "dataset via `dataset.with_options(options)`.");
  }

  const NodeDef* input_node = graph_utils::GetInputNode(node, *graph, 0);
  return RecursivelyHandleOp(*input_node, num_workers, index, flib, graph,
                             nodes_to_delete);
}

// Recursively walk the dataset graph from sink to source, searching for
// the first (i.e. closest to the sink) occurrence of a ReaderDataset, such as
// CSVDataset, TFRecordDataset, etc. We then insert a ShardDataset op before
// that nodes input, so that each worker only reads a subset of files.
// Additionally, we remove sources of randomness (e.g. ShuffleDataset) that
// occur upstream of the ShardDataset transformation to ensure that sharding
// returns a sensible result.
Status ShardByFile(const NodeDef& sink_node, int64_t num_workers, int64_t index,
                   FunctionLibraryDefinition* flib, MutableGraphView* graph) {
  absl::flat_hash_set<string> nodes_to_delete;
  TF_RETURN_IF_ERROR(RecursivelyHandleOp(sink_node, num_workers, index, flib,
                                         graph, &nodes_to_delete));
  return graph->DeleteNodes(nodes_to_delete);
}

Status RewriteRebatchV2ToV1(const NodeDef& sink_node, int64_t num_replicas,
                            MutableGraphView* graph) {
  // The final node before AutoShardDataset is RebatchDataset.
  // This is always the case as RebatchDataset and AutoShardDataset are internal
  // APIs used directly by tf.distribute's input_lib. As such, instead of
  // walking the entire dataset graph, we can walk up directly from the
  // sink_node to get the RebatchDataset.
  NodeDef* input_node = graph_utils::GetInputNode(sink_node, *graph);
  if (input_node->op() != kRebatchDatasetV2OpName) {
    return OkStatus();
  }

  NodeDef* rebatch_node = input_node;
  // Update RebatchDatasetV2 in place. Since Rebatch is an internal API, no
  // other nodes should have it as an input.
  rebatch_node->set_op(kRebatchDatasetOpName);
  // Delete the `batch_sizes` and `drop_remainder` input.
  rebatch_node->mutable_input()->DeleteSubrange(/*start=*/1, /*num=*/2);
  // Add the `num_replicas` input.
  if (num_replicas < 1) {
    return errors::InvalidArgument(
        "Cannot rewrite RebatchDatasetV2 to legacy RebatchDataset with invalid "
        "num_replicas argument. `num_replicas` is ",
        num_replicas, ", but expected to be >= 1.");
  }
  auto num_replicas_node = graph_utils::AddScalarConstNode(num_replicas, graph);
  rebatch_node->add_input(num_replicas_node->name());

  // Set `use_fallback` attr. This attr is not used anywhere, so its value
  // does not matter
  (*rebatch_node->mutable_attr())["use_fallback"].set_b(true);

  // Update the output_shapes attr to set all its batch dimensions to -1
  // (unknown).
  auto* shapes_attr =
      gtl::FindOrNull(*rebatch_node->mutable_attr(), "output_shapes");
  if (shapes_attr == nullptr) {
    return errors::InvalidArgument(
        "Cannot rewrite RebatchDatasetV2 with missing `output_shapes` attr.");
  }
  for (int i = 0; i < shapes_attr->list().shape_size(); ++i) {
    auto* shape = shapes_attr->mutable_list()->mutable_shape(i);
    if (shape->unknown_rank()) continue;
    shape->mutable_dim(0)->set_size(-1);
  }

  return OkStatus();
}

Status ShardByData(const NodeDef& sink_node, int64_t num_workers, int64_t index,
                   int64_t num_replicas, MutableGraphView* graph) {
  const NodeDef* shard_before = &sink_node;
  // We sometimes insert a PrefetchDataset, OptionsDataset, and FinalizeDataset
  // at the end of the input pipeline before autosharding. When sharding by
  // data, we should insert the shard before the these datasets so that the
  // right number of elements is prefetched.
  NodeDef* input_node = graph_utils::GetInputNode(sink_node, *graph);
  while (input_node->op() == kPrefetchDatasetOpName ||
         input_node->op() == kOptionsDatasetOpName ||
         input_node->op() == kFinalizeDatasetOpName) {
    shard_before = input_node;
    input_node = graph_utils::GetInputNode(*input_node, *graph);
  }
  // Sharding by data only works with legacy RebatchDataset. As such, we rewrite
  // all instances of RebatchDatasetV2 to RebatchDataset.
  TF_RETURN_IF_ERROR(RewriteRebatchV2ToV1(*shard_before, num_replicas, graph));
  return AddShardNode(graph, *shard_before, num_workers, index);
}

// Searches the dataset graph replacing any occurrence of `shard(1, 0)` with
// `shard(num_workers, index)`.
Status ShardByHint(const NodeDef& sink_node, int64_t num_workers, int64_t index,
                   int64_t num_replicas, MutableGraphView* graph) {
  auto get_shard_node = [graph](const NodeDef& node) -> const NodeDef* {
    if (node.op() != kShardDatasetOpName) return nullptr;
    auto num_workers_node = graph->GetNode(node.input(1));
    if (num_workers_node->op() != kConstOpName) return nullptr;
    if (num_workers_node->attr().at("value").tensor().int64_val(0) !=
        tensorflow::data::kShardHint)
      return nullptr;
    return &node;
  };

  auto* num_workers_node =
      graph_utils::AddScalarConstNode(static_cast<int64_t>(num_workers), graph);
  auto* worker_index_node =
      graph_utils::AddScalarConstNode(static_cast<int64_t>(index), graph);

  for (const NodeDef& node : graph->graph()->node()) {
    const NodeDef* shard_node = get_shard_node(node);
    if (!shard_node) continue;
    auto mutable_node = graph->GetNode(shard_node->name());
    *mutable_node->mutable_input(1) = num_workers_node->name();
    *mutable_node->mutable_input(2) = worker_index_node->name();
    // Ensure that each shard will have at least one element.
    (*(mutable_node->mutable_attr()))[data::ShardDatasetOp::kRequireNonEmpty]
        .set_b(true);
  }
  return OkStatus();
}

Status ApplyAutoShard(const NodeDef& sink_node, int64_t num_workers,
                      int64_t index, AutoShardPolicy policy,
                      int64_t num_replicas, MutableGraphView* graph,
                      AutoShardPolicy* policy_applied) {
  *policy_applied = policy;
  FunctionLibraryDefinition flib(OpRegistry::Global(),
                                 graph->graph()->library());
  switch (policy) {
    case AutoShardPolicy::OFF:
      return OkStatus();
    case AutoShardPolicy::FILE:
      return ShardByFile(sink_node, num_workers, index, &flib, graph);
    case AutoShardPolicy::DATA:
      return ShardByData(sink_node, num_workers, index, num_replicas, graph);
    case AutoShardPolicy::HINT:
      return ShardByHint(sink_node, num_workers, index, num_replicas, graph);
    case AutoShardPolicy::AUTO:
    default:
      Status s = ShardByFile(sink_node, num_workers, index, &flib, graph);
      if (errors::IsNotFound(s)) {
        LOG(WARNING) << "AUTO sharding policy will apply DATA sharding policy "
                        "as it failed to apply FILE sharding policy because of "
                        "the following reason: "
                     << s.error_message();
        *policy_applied = AutoShardPolicy::DATA;
        return ShardByData(sink_node, num_workers, index, num_replicas, graph);
      }
      *policy_applied = AutoShardPolicy::FILE;
      return s;
  }
}

Status OptimizeGraph(const GrapplerItem& item, int64_t num_workers,
                     int64_t index, AutoShardPolicy policy,
                     int64_t num_replicas, GraphDef* output) {
  *output = item.graph;
  MutableGraphView graph(output);
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  // id for telemetry purpose. item.id is always the same so we use the address
  // of the output as id.
  string id = strings::StrCat(reinterpret_cast<uint64>(output));
  // Only record metrics on the first shard to avoid duplication.
  if (index == 0) {
    std::vector<std::string> ineligible_reason;
    bool is_eligible = internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                            &ineligible_reason);
    metrics::RecordTFDataAutoShardRewriteBatchSize(is_eligible,
                                                   ineligible_reason);
  }

  AutoShardPolicy policy_applied = policy;
  if (policy != AutoShardPolicy::OFF &&
      !(policy == AutoShardPolicy::FILE && num_workers == 1 && index == 0)) {
    TF_RETURN_IF_ERROR(ApplyAutoShard(*sink_node, num_workers, index, policy,
                                      num_replicas, &graph, &policy_applied));
  }
  // Only record metrics on the first shard to avoid duplication.
  if (index == 0) {
    metrics::RecordTFDataAutoShard(id, policy_applied, num_workers,
                                   num_replicas);
  }
  return OkStatus();
}

}  // anonymous namespace

namespace internal {
bool IsEligibleRewriteBatchSize(const NodeDef& sink_node,
                                const MutableGraphView& graph,
                                std::vector<std::string>* ineligible_reason) {
  ineligible_reason->clear();
  NodeDef* input_node = graph_utils::GetInputNode(sink_node, graph);
  // We always traverse the graph until we arrive at a batch node to collect all
  // ineligible reasons;
  while (input_node != nullptr) {
    // 1. Skip RebatchDataset and the MapDataset immediately before it. That map
    // is added by tf.data Python code.
    if (input_node->op() == kRebatchDatasetOpName ||
        input_node->op() == kRebatchDatasetV2OpName) {
      input_node = graph_utils::GetInputNode(*input_node, graph);
      if (input_node == nullptr || input_node->op() != kMapDatasetOpName) {
        ineligible_reason->push_back("BUG_NO_MAP_BEFORE_REBATCH");
        return false;
      }
      input_node = graph_utils::GetInputNode(*input_node, graph);
      continue;
    }
    // 2. If the node is insensitive to the batch size of the input, we continue
    // looking at the input dataset of the node.
    if (IsDatasetNodeOfType(*input_node, kBatchSizeOrthogonalDatasetOps)) {
      input_node = graph_utils::GetInputNode(*input_node, graph);
      continue;
    }
    // 3. We arrive at a batch node. Examine its drop_remainder input and
    // cardinality to determine eligibility.
    if (IsDatasetNodeOfType(*input_node, kBatchDatasetOps)) {
      DropRemainderValue drop_remainder = GetDropRemainder(graph, *input_node);
      int64_t cardinality = data::kUnknownCardinality;
      bool cardinality_available = true;
      AttrSlice attrs(*input_node);
      if (!TryGetNodeAttr(attrs, data::kCardinalityAttrForRewrite,
                          &cardinality)) {
        cardinality_available = false;
      }

      if (drop_remainder == DropRemainderValue::kFalse ||
          (cardinality_available &&
           cardinality == data::kInfiniteCardinality)) {
        return ineligible_reason->empty();
      } else {
        if (drop_remainder == DropRemainderValue::kUnknown) {
          ineligible_reason->push_back("BATCH_DROP_REMAINDER_UNKNOWN");
        }
        if (!cardinality_available) {
          ineligible_reason->push_back("BATCH_CARDINALITY_NOT_AVAILABLE");
        }
        if (drop_remainder == DropRemainderValue::kTrue &&
            cardinality_available &&
            cardinality != data::kInfiniteCardinality) {
          ineligible_reason->push_back("BATCH_DROP_REMAINDER_NOT_INFINITE");
        }
        return false;
      }
    }
    // 4. We encountered other nodes before arriving at a batch node. We don't
    // know whether this node is sensitive to the batch size or not and we err
    // on the safe side.
    ineligible_reason->push_back(
        strings::StrCat("OP_NOT_SUPPORTED_", input_node->op()));
    input_node = graph_utils::GetInputNode(*input_node, graph);
  }
  // If we don't find a batch node, only records BATCH_NOT_FOUND as the reason.
  ineligible_reason->clear();
  ineligible_reason->push_back("BATCH_NOT_FOUND");
  return false;
}
}  // namespace internal

Status AutoShard::Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  if (!config) return errors::InvalidArgument("RewriterConfig not found.");

  if ((config->parameter_map().find(kNumWorkersAttrName) ==
       config->parameter_map().end())) {
    return errors::InvalidArgument(kNumWorkersAttrName, " parameter missing.");
  }

  if ((config->parameter_map().find(kIndexAttrName) ==
       config->parameter_map().end())) {
    return errors::InvalidArgument(kIndexAttrName, " parameter missing.");
  }

  num_workers_ = config->parameter_map().at(kNumWorkersAttrName).i();
  index_ = config->parameter_map().at(kIndexAttrName).i();
  auto_shard_policy_ =
      AutoShardPolicy(config->parameter_map().at(kAutoShardPolicyAttrName).i());
  num_replicas_ = config->parameter_map().at(kNumReplicasAttrName).i();

  if (auto_shard_policy_ != AutoShardPolicy::OFF &&
      auto_shard_policy_ != AutoShardPolicy::AUTO &&
      auto_shard_policy_ != AutoShardPolicy::DATA &&
      auto_shard_policy_ != AutoShardPolicy::FILE &&
      auto_shard_policy_ != AutoShardPolicy::HINT) {
    return errors::InvalidArgument(kAutoShardPolicyAttrName, " is invalid.");
  }

  if (num_workers_ < 1) {
    return errors::InvalidArgument(kNumWorkersAttrName,
                                   " should be >= 1, currently ", num_workers_);
  }

  if (index_ < 0 || index_ >= num_workers_) {
    return errors::InvalidArgument(kIndexAttrName, " should be >= 0 and < ",
                                   num_workers_, ", currently ", index_);
  }

  if (num_replicas_ < 0) {
    return errors::InvalidArgument(kNumReplicasAttrName, " should be >= 0");
  }

  return OkStatus();
}

Status AutoShard::OptimizeAndCollectStats(Cluster* cluster,
                                          const GrapplerItem& item,
                                          GraphDef* output,
                                          OptimizationStats* stats) {
  *output = item.graph;
  TF_RETURN_IF_ERROR(OptimizeGraph(item, num_workers_, index_,
                                   auto_shard_policy_, num_replicas_, output));
  stats->num_changes++;
  return OkStatus();
}

REGISTER_GRAPH_OPTIMIZER_AS(AutoShard, "tf_auto_shard");

}  // namespace grappler
}  // namespace tensorflow

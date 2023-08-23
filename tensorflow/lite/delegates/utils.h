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

#ifndef TENSORFLOW_LITE_DELEGATES_UTILS_H_
#define TENSORFLOW_LITE_DELEGATES_UTILS_H_

// Utility functions and classes for implementing delegates.

#include <functional>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {

// Creates a new Read/Write tensor having the same shape as the original, but
// with a different type. Note that this might void existing references to
// tensors.
TfLiteStatus CreateNewTensorWithDifferentType(TfLiteContext* context,
                                              const int original_tensor_index,
                                              TfLiteType new_type,
                                              TfLiteTensor** new_tensor,
                                              int* new_tensor_index);

// Retrieves the corresponding TfLiteContext of a subgraph given a subgraph
// index and switches to the delegate context for this subgraph. If an invalid
// subgraph index is given, returns kTfLiteError.
// NOTE: This function is expected to be paired with ReleaseSubgraphContext()
// once the delegate preparation is done and/or the delegate context functions
// are no longer needed.
TfLiteStatus AcquireSubgraphContext(const TfLiteContext* context,
                                    int subgraph_index,
                                    TfLiteContext** acquired_context);

// Releases the subgraph context by switching back to the TFLite kernel
// context for this specified subgraph.
// NOTE: This function is expected to be used after AcquireSubgraphContext()
// once the delegate preparation is done and/or the delegate context functions
// are no longer needed.
TfLiteStatus ReleaseSubgraphContext(const TfLiteContext* context,
                                    int subgraph_index);

// Marks the subgraph with the given index as delegation-skippable. Returns
// kTfLiteOk if the given subgraph index is valid and is successfully marked
// as delegation-skippable, and an error status if the subgraph index is
// invalid.
// If a subgraph is delegation-skippable, then the subgraph will be handled by
// a TfLiteDelegate (and that the delegate is supposed to be already aware of
// this state), and therefore, TfLiteInterpreter can skip invoking
// `ModifyGraphWithDelegate` on this subgraph.
// NOTE: This function is expected to be called only when the subgraph that
// `subgraph_index` is pointing to should be skipped by
// interpreter::ModifyGraphWithDelegate (e.g. the subgraph is part of the list
// of callee subgraphs of the same control flow node, and all of those callees
// are supported by the same delegate at once).
//
// For example, this function can be used when the delegate is handling
// control flow ops like while op. E.g. A while op has condition subgraph
// indexed at `i` and body subgraph indexed at `j`. The op can be delegated
// when the following condition satisfied:
//   1. The delegate supports while op
//   2. Both condition subgraph `i` and body subgraph `j` can be fully
//   delegated by the delegate.
// Then if the delegate decides to support the while node along with both body
// and condition subgraphs, it should mark subgraphs `i` and `j` skippable so
// those two subgraphs won't be delegated separately again after being
// absorbed by the parent subgraph.
// WARNING: It is the delegate's responsibility to define when to skip
// subgraph->ModifyGraphWithDelegate, to check any edge cases (i.e. multiple
// references to the subgraph that `subgraph_index` is pointing to), and to mark
// that subgraph as skippable using this function.
// NOTE: Entry point for C node plugin API.
TfLiteStatus MarkSubgraphAsDelegationSkippable(const TfLiteContext* context,
                                               int subgraph_index);

using IsNodeSupportedFn =
    std::function<bool(TfLiteContext*, TfLiteNode*, TfLiteRegistration*,
                       std::string* unsupported_details)>;

// A utility class to help model graph parition.
// Note the class *needs* to be used in TfLiteDelegate::Prepare.
class GraphPartitionHelper {
 public:
  GraphPartitionHelper(TfLiteContext* context,
                       IsNodeSupportedFn is_node_supported_fn)
      : context_(context), is_node_supported_fn_(is_node_supported_fn) {}

  GraphPartitionHelper(TfLiteContext* context,
                       const std::vector<int>& supported_node_indices)
      : context_(context),
        num_total_nodes_(supported_node_indices.size()),
        supported_nodes_(
            ConvertVectorToTfLiteIntArray(supported_node_indices)) {}

  virtual ~GraphPartitionHelper() {
    TfLiteIntArrayFree(supported_nodes_);
    TfLiteIntArrayFree(original_execution_plan_);
  }

  // Partition the graph into node subsets such that each subset could be
  // replaced with one delegate kernel (i.e. a kTfLiteBuiltinDelegate op).
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  virtual TfLiteStatus Partition(
      std::set<std::string>* unsupported_nodes_info) {
    return PartitionImpl(unsupported_nodes_info, 0,
                         std::numeric_limits<int>::max());
  }

#ifdef TFLITE_DEBUG_DELEGATE
  // Partition the graph into node subsets such that each subset could be
  // replaced with one delegate kernel (i.e. a kTfLiteBuiltinDelegate op).
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  // The 'start_node_index' and 'end_node_index' define the range of nodes
  // that could be delegated.
  virtual TfLiteStatus Partition(std::set<std::string>* unsupported_nodes_info,
                                 int start_node_index, int end_node_index) {
    return PartitionImpl(unsupported_nodes_info, start_node_index,
                         end_node_index);
  }
#endif  // TFLITE_DEBUG_DELEGATE

  // Returns the first n largest partitions or all if #partitions is less than
  // 'n' and each parition has at least (>=) 'min_nodes_per_partition' nodes.
  // Note that partitions are ranked according to the number of nodes that
  // a partition has, and the returned TfLiteDelegateParams objects are *owned*
  // by the TfLite runtime.
  // TODO(b/156707497): remove this and use GetNodesOfFirstNLargestPartitions
  std::vector<TfLiteDelegateParams*> GetFirstNLargestPartitions(
      int n = std::numeric_limits<int>::max(),
      int min_nodes_per_partition = 0) const;

  // Returns a list of node indices of all nodes from the first n largest
  // partitions. If there are fewer paritions than n, all nodes will be
  // returned. The partition is ranked according to the number of nodes.
  std::vector<int> GetNodesOfFirstNLargestPartitions(
      int n = std::numeric_limits<int>::max(),
      int min_nodes_per_partition = 0) {
    // Separated implementation that can be overrided, to preserve default value
    return GetNodesOfFirstNLargestPartitionsImpl(n, min_nodes_per_partition);
  }

  int num_total_nodes() const { return num_total_nodes_; }
  int num_supported_nodes() const { return num_supported_nodes_; }
  int num_partitions() const { return partitions_.size(); }

 protected:
  virtual bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                               TfLiteRegistration* registration, int node_id,
                               std::string* unsupported_details) {
    return is_node_supported_fn_(context, node, registration,
                                 unsupported_details);
  }
  virtual std::vector<int> GetNodesOfFirstNLargestPartitionsImpl(
      int n, int min_nodes_per_partition);
  virtual TfLiteStatus PartitionImpl(
      std::set<std::string>* unsupported_nodes_info, int start_node_index,
      int end_node_index);

  TfLiteContext* const context_ = nullptr;

  // Doesn't own the memory of each TfLiteDelegateParams object as it's
  // managed by the TfLite runtime itself. See
  // TfLiteContext::PreviewDelegatePartitioning for details.
  std::vector<TfLiteDelegateParams*> partitions_;

  // Copy of (pre-delegation) execution plan obtained from TfLiteContext in
  // PrepareSupportedNodes
  TfLiteIntArray* original_execution_plan_ = nullptr;

 private:
  // Generate a list of supported nodes (i.e. populating 'supported_nodes_') by
  // iterating over all nodes (i,e. those listed in the execution_plan
  // associated w/ 'context_').
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  // The 'start_node_index' and 'end_node_index' define the range of nodes that
  // could be delegated.
  TfLiteStatus PrepareSupportedNodes(
      std::set<std::string>* unsupported_nodes_info = nullptr,
      int start_node_index = 0,
      int end_node_index = std::numeric_limits<int>::max());

  // The number of total nodes passed in for partitioning (i.e. the
  // execution_plan size associated w/ 'context_')
  int num_total_nodes_ = 0;

  int num_supported_nodes_ = 0;

  // Tells if a node is supported as it could be delegated.
  const IsNodeSupportedFn is_node_supported_fn_ = nullptr;

  // Contains an array of supported node indices.
  TfLiteIntArray* supported_nodes_ = nullptr;  // owns the memory
};

// Specialized partitioner for graphs that possibly contain fp16 tensors.
//
// From nodes that accept fp16 inputs, this delegates the following:
// 1. All nodes (except DEQUANTIZE) that are supported with constant fp16 inputs
// by the delegate (in the TFLite graph, these nodes take in dequantized FP32
// outputs).
// 2. All fp16 DEQUANTIZE nodes that have *all* their consumers in the *first*
// delegated partition. This is because TFLite's partitioning algorithm
// greedily puts all such nodes in the first partition.
class FP16GraphPartitionHelper : public GraphPartitionHelper {
 public:
  FP16GraphPartitionHelper(TfLiteContext* context,
                           IsNodeSupportedFn is_node_supported_fn)
      : GraphPartitionHelper(context, std::move(is_node_supported_fn)) {}

 protected:
  // Specialized function to handle fp16 nodes.
  bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                       TfLiteRegistration* registration, int node_id,
                       std::string* unsupported_details) override;

  // This will remap input tensors by removing FP16 to FP32 dequantized tensors.
  std::vector<int> GetNodesOfFirstNLargestPartitionsImpl(
      int n, int min_nodes_per_partition) override;

 private:
  // This remaps fp32 inputs of the given node to their corresponding fp16
  // version, if applicable. Can be summarized as:
  // fp16 -> DEQUANTIZE -> fp32 -> OP -> output
  // becomes
  // fp16 -> OP -> output
  void RemapFp16InputTensors(TfLiteNode* node,
                             std::vector<int>* orig_inputs) const;

  // Performs the above remapping for all nodes in the given list, without
  // tracking the original inputs.
  void RemapFp16InputTensors(const std::vector<int>& nodes) const;

  // ('dequantize' here refers to fp16 DEQUANTIZE)
  // Mapping of dequantize nodes' output tensor-id to its node id.
  // TODO(b/156707497): Use absl hash_maps here.
  std::unordered_map<int, int> constant_dequant_nodes_;
  // Mapping of DEQUANTIZE node's output (fp32) to its input (fp16).
  std::unordered_map<int, int> constant_dequant_map_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_H_

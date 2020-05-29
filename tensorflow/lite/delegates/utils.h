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
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {

// Creates a new Read/Write tensor having the same shape as the original, but
// with a different type.
TfLiteStatus CreateNewTensorWithDifferentType(TfLiteContext* context,
                                              const int original_tensor_index,
                                              TfLiteType new_type,
                                              TfLiteTensor** new_tensor,
                                              int* new_tensor_index);

std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data);

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

  virtual ~GraphPartitionHelper() { TfLiteIntArrayFree(supported_nodes_); }

  // Partition the graph into node subsets such that each subset could be
  // replaced with one delegate kernel (i.e. a kTfLiteBuiltinDelegate op).
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  virtual TfLiteStatus Partition(std::set<std::string>* unsupported_nodes_info);

  // Returns the first n largest partitions or all if #partitions is less than
  // 'n' and each parition has at least (>=) 'min_nodes_per_partition' nodes.
  // Note that partitions are ranked according to the number of nodes that
  // a partition has, and the returned TfLiteDelegateParams objects are *owned*
  // by the TfLite runtime.
  std::vector<TfLiteDelegateParams*> GetFirstNLargestPartitions(
      int n = std::numeric_limits<int>::max(),
      int min_nodes_per_partition = 0) const;

  int num_total_nodes() const { return num_total_nodes_; }
  int num_partitions() const { return partitions_.size(); }

 protected:
  virtual bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                               TfLiteRegistration* registration, int node_id,
                               std::string* unsupported_details) {
    return is_node_supported_fn_(context, node, registration,
                                 unsupported_details);
  }

  TfLiteContext* const context_ = nullptr;

  // Doesn't own the memory of each TfLiteDelegateParams object as it's
  // managed by the TfLite runtime itself. See
  // TfLiteContext::PreviewDelegatePartitioning for details.
  std::vector<TfLiteDelegateParams*> partitions_;

 private:
  // Generate a list of supported nodes (i.e. populating 'supported_nodes_') by
  // iterating over all nodes (i,e. those listed in the execution_plan
  // associated w/ 'context_').
  // If 'unsupported_nodes_info' is provided, it will be populated with
  // information about all different unsupported nodes.
  TfLiteStatus PrepareSupportedNodes(
      std::set<std::string>* unsupported_nodes_info = nullptr);

  // The number of total nodes passed in for partitioning (i.e. the
  // execution_plan size associated w/ 'context_')
  int num_total_nodes_ = 0;

  // Tells if a node is supported as it could be delegated.
  const IsNodeSupportedFn is_node_supported_fn_ = nullptr;

  // Contains an array of supported node indices.
  TfLiteIntArray* supported_nodes_ = nullptr;  // owns the memory
};

// While partitioning the graph, this claims DEQUANTIZE nodes (FP16->FP32) in
// addition to supported nodes for the delegate, when the DEQUANTIZE node's
// output is an input to the kernel that supports FP16 input.
// Noth that you have to use `GetNodesOfFirstNLargestPartitions` instead of
// superclass' `GetFirstNLargestPartitions` to do actual remapping of FP16
// inputs.
class FP16GraphPartitionHelper : public GraphPartitionHelper {
 public:
  FP16GraphPartitionHelper(TfLiteContext* context,
                           IsNodeSupportedFn is_node_supported_fn)
      : GraphPartitionHelper(context, std::move(is_node_supported_fn)) {}

  TfLiteStatus Partition(
      std::set<std::string>* unsupported_nodes_info) override;

  // Returns a list of node indices of all nodes from the first n largest
  // partitions. If there are fewer paritions than n, all nodes will be
  // returned. The partition is ranked according to the number of nodes.
  // TODO(b/156707497): Add this to superclass besides
  // GetFirstNLargestPartitions (one that returns partitions instead of nodes)
  std::vector<int> GetNodesOfFirstNLargestPartitions(
      int n, int min_nodes_per_partition = 0,
      std::vector<TfLiteDelegateParams*>* partitions = nullptr);

 protected:
  bool IsNodeSupported(TfLiteContext* context, TfLiteNode* node,
                       TfLiteRegistration* registration, int node_id,
                       std::string* unsupported_details) override;

 private:
  // Record 'node' if it is a dequant op (i.e. a fp16 one here) and return true.
  // When it's not a dequant op, remap its inputs to the inputs of the preceding
  // dequant if there's a one and returns false. 'orig_inputs' records original
  // input tensor ids of this node if any input is remapped.
  bool RecordAndRemapInputTensors(int32_t op_code, int node_id,
                                  TfLiteNode* node,
                                  std::vector<int>* orig_inputs);

  // Restore inputs of 'node' to 'orig_inputs' only if two sizes match.
  void RestoreToOrigInputTensors(TfLiteNode* node,
                                 const std::vector<int>& orig_inputs);

  // Remap input tensors of every node in 'nodes' (i.e. node indices) if some of
  // them are from dequant ops.
  void RemapInputTensors(const std::vector<int>& nodes) const;

  void RemoveSingleDequantNodePartitions();

  void RemoveReservedDequantsFromNodes(std::vector<int>* nodes);

  // Remap input tensors of a single 'node' if some of come from a dequant op.
  // If 'orig_inputs' isn't nullptr, it records original input tensor ids of
  // this node if any input is remapped.
  void RemapInputTensors(TfLiteNode* node, std::vector<int>* orig_inputs) const;

  // A map recording dequantize nodes's input/output tensors of this selected
  // graph. The key is the output tensor id, and the value is the input tensor
  // id.
  std::unordered_map<int, int> dequant_nodes_;

  // A set of dequant nodes as in node indices that have to be preserved in the
  // graph.
  std::set<int> dequant_nodes_to_save_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_UTILS_H_

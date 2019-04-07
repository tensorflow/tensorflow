/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_H_

#include <functional>
#include <iterator>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
#include "absl/types/span.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace grappler {

// A utility class to lookup a node and its outputs by node name.
class NodeMap {
 public:
  // Note: The NodeMap will store pointers to nodes in graph, which may become
  // invalid if graph is changed.
  explicit NodeMap(GraphDef* graph);
  NodeDef* GetNode(const string& name) const;
  bool NodeExists(const string& name) const;
  const std::set<NodeDef*>& GetOutputs(const string& node_name) const;
  // This method doesn't record the outputs of the added node; the outputs need
  // to be explicitly added by the AddOutput method.
  void AddNode(const string& name, NodeDef* node);
  void RemoveNode(const string& name);
  void UpdateInput(const string& node_name, const string& old_input_name,
                   const string& new_input_name);
  void AddOutput(const string& node_name, const string& output_name);
  void RemoveInputs(const string& node_name);
  void RemoveOutput(const string& node_name, const string& output_name);
  void RemoveOutputs(const string& node_name);
  void UpdateOutput(const string& node_name, const string& old_output_name,
                    const string& new_output_name);

 private:
  const std::set<NodeDef*> empty_set_;
  gtl::FlatMap<string, NodeDef*> nodes_;
  gtl::FlatMap<string, std::set<NodeDef*>> outputs_;
};

// A vector with a set. The set stores the same elements as the vector, and
// quickly answers whether a value is in the vector. Duplicated elements are not
// allowed for now.
template <class T, class Hash = std::hash<T>>
class SetVector {
 public:
  // Returns false if value already existed in the set, true otherwise.
  bool PushBack(const T& value) {
    if (!set_.insert(value).second) {
      return false;
    }
    vector_.push_back(value);
    return true;
  }

  T PopBack() {
    T back = vector_.back();
    set_.erase(back);
    vector_.pop_back();
    return back;
  }

  bool Exists(const T& value) const { return set_.find(value) != set_.end(); }

  bool Empty() const { return vector_.empty(); }

  void Reserve(int64 size) { vector_.reserve(size); }

 private:
  gtl::FlatSet<T, Hash> set_;
  std::vector<T> vector_;
};

// Returns formatted string from TensorId specific to grappler. Specifically,
// for the 0 port (first output), only the node name is returned.
string TensorIdToString(const TensorId& tensor_id);

// Returns formatted string from SafeTensorId specific to grappler.
// Specifically, for the 0 port (first output), only the node name is returned.
string SafeTensorIdToString(const SafeTensorId& tensor_id);

// True iff 'name' refers to a control inputs, i.e. a node name prefixed with
// the ^ character.
bool IsControlInput(const string& name);

// True iff tensor index refers to a control input.
bool IsControlInput(const TensorId& tensor_id);

// True iff 'name1' and 'name2' refer to the same input.
bool IsSameInput(const string& name1, const string& name2);

// Returns the trailing position number (or zero if no number is present) if
// NodeName(input_name) is equal to node_name. Returns -1 for control inputs.
// Returns -2 if NodeName(input_name) is not equal to node_name.
// Note: This function is used very heavily, and this hand-optimized
// version is 3-4x faster than the version using Scanner, which it replaced.
// This is worth the reduction in readability.
inline int NodePositionIfSameNode(const string& input_name,
                                  const string& node_name) {
  if (input_name.empty()) return -2;
  const bool is_ctrl = input_name[0] == '^';
  auto input_it = is_ctrl ? input_name.begin() + 1 : input_name.begin();
  auto node_it = node_name.begin();
  if (node_name.empty() ||
      std::distance(input_it, input_name.end()) < node_name.size()) {
    return -2;
  }
  while (node_it != node_name.end()) {
    if (*input_it++ != *node_it++) {
      return -2;
    }
  }
  if (input_it == input_name.end()) {
    return is_ctrl ? -1 : 0;
  } else if (*input_it++ == ':') {
    StringPiece remaining(&(*input_it),
                          std::distance(input_it, input_name.end()));
    int position;
    if (!strings::safe_strto32(remaining, &position)) {
      return -2;
    }
    return is_ctrl ? -1 : position;
  } else {
    return -2;
  }
}

// Return the node name corresponding to 'name' if name is valid, or the empty
// string otherwise.
inline StringPiece NodeNameAsStringPiece(const string& name) {
  static const string empty;
  if (name.empty()) return StringPiece(empty);
  const auto begin_it = name[0] == '^' ? name.begin() + 1 : name.begin();
  auto end_it = begin_it;
  while (end_it != name.end() && *end_it != ':') {
    ++end_it;
  }
  if (end_it != name.end() && *end_it != ':') {
    return StringPiece(empty);
  }
  return StringPiece(&(*begin_it), std::distance(begin_it, end_it));
}

// Return the node name corresponding to 'name' if name is valid, or the empty
// string otherwise.
inline string NodeName(const string& name) {
  return string(NodeNameAsStringPiece(name));
}

// Returns the node name and position in a single call.
// DEPRECATED(ezhulenev): Use TensorId and ParseTensorName.
inline StringPiece ParseNodeNameAsStringPiece(const string& name,
                                              int* position) {
  static const string empty;
  if (name.empty()) {
    *position = 0;
    return StringPiece(empty);
  }
  const bool is_ctrl = name[0] == '^';
  const auto begin_it = is_ctrl ? name.begin() + 1 : name.begin();
  *position = is_ctrl ? -1 : 0;
  auto end_it = begin_it;
  while (end_it != name.end() && *end_it != ':') {
    ++end_it;
  }
  const StringPiece node_name(&(*begin_it), std::distance(begin_it, end_it));
  if (end_it != name.end()) {
    if (*end_it != ':') {
      return StringPiece(empty);
    } else if (!is_ctrl) {
      ++end_it;
      StringPiece remaining(&(*end_it), std::distance(end_it, name.end()));
      if (!strings::safe_strto32(remaining, position)) {
        return StringPiece(empty);
      }
    }
  }
  return node_name;
}

// Returns the node name and position in a single call.
// DEPRECATED(ezhulenev): Use SafeTensorId and ParseTensorName.
inline string ParseNodeName(const string& name, int* position) {
  return string(ParseNodeNameAsStringPiece(name, position));
}

inline int NodePosition(const string& name) {
  int position;
  ParseNodeNameAsStringPiece(name, &position);
  return position;
}

// Add a prefix to a node name with a custom delimiter.
string AddPrefixToNodeName(const string& name, const string& prefix,
                           const string& delimiter);

// Add a prefix to a node name.
string AddPrefixToNodeName(const string& name, const string& prefix);

// Executes a 'fn' in the 'thread_pool'. The method waits for the configured
// timeout (in milliseconds) for 'fn' to complete, before returning false.
//
// If returning false, the 'fn' may still continue to execute in the
// thread-pool. It is the responsibility of the caller to reset the thread-pool
// as appropriate.
bool ExecuteWithTimeout(std::function<void()> fn, int64 timeout_in_ms,
                        thread::ThreadPool* thread_pool);

// Returns the node name prefixed with conventional symbol '^'
// for control dependency, given a NodeDef.
string AsControlDependency(const NodeDef& node);

// Returns the node name prefixed with conventional symbol '^'
// for control dependency, given a node name
string AsControlDependency(const string& node);

// Returns true if the node is assigned to run on CPU device.
bool NodeIsOnCpu(const NodeDef* node);

// Returns true if the node is assigned to run on GPU device.
bool NodeIsOnGpu(const NodeDef* node);

// Returns the number of outputs of a node according to its OpDef. Note that
// some of the outputs may be unconnected.
int NumOutputs(const NodeDef& node, GraphDef* graph);

// Returns true iff the node has at least one control input.
bool HasControlInputs(const NodeDef& node);

// Number of connected non-control inputs.
int NumNonControlInputs(const NodeDef& node);

// Number of connected non-control outputs.
int NumNonControlOutputs(const NodeDef& node, const NodeMap& node_map);

// Number of connected non-control data outputs (Ops that consume output tensor
// data, not just it's shape).
int NumNonControlDataOutputs(const NodeDef& node, const NodeMap& node_map);

// Removes redundant control inputs from node.
void DedupControlInputs(NodeDef* node);

// Returns an error if an attribute with the given key does not exist in node.
Status CheckAttrExists(const NodeDef& node, const string& key);

// Returns an error if attributes with the given keys do not exist in node.
Status CheckAttrsExist(const NodeDef& node, absl::Span<const string> keys);

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& type_attr);

// Returns the last node in the simple chain starting at source and traversing
// through the input(0) edge from each node as long as the next node satisfies
// the predicate given in pred_fn. If no nodes satisfy the predicate, &source
// will be returned. Example: For the chain
//    source <- a <- b <- ... <- y <- z
// where
//    pred_fn(a) = pred_fn(b) = ... = pred_fn(y) = true,
//    pred_fn(z) = false,
// the return value will be a pointer to y.
NodeDef* GetTailOfChain(const NodeDef& source, const NodeMap& node_map,
                        bool follow_control_input,
                        const std::function<bool(const NodeDef&)>& pred_fn);

// Permute the nodes of graph in place according to the permutation.
void PermuteNodesInPlace(GraphDef* graph, std::vector<int>* permutation,
                         bool invert_permutation);

// Returns Status::OK() if a kernel is registered for node.op() on the device
// type corresponding to node.device().
Status IsKernelRegisteredForNode(const NodeDef& node);

Status SetTensorValue(DataType dtype, int value, Tensor* tensor);

void EraseNodesFromGraph(const std::set<int>& nodes_to_delete, GraphDef* graph);

void EraseNodesFromGraph(std::vector<int>&& nodes_to_delete, GraphDef* graph);

void EraseNodesFromGraph(const std::set<string>& nodes_to_delete,
                         GraphDef* graph);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_H_

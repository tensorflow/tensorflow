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

#ifndef TENSORFLOW_GRAPPLER_UTILS_H_
#define TENSORFLOW_GRAPPLER_UTILS_H_

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

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
  std::unordered_map<string, NodeDef*> nodes_;
  std::unordered_map<string, std::set<NodeDef*>> outputs_;
};

// A utility class to lookup a node's outputs and the number of times it
// presents in each output.
class OutputMap {
 public:
  explicit OutputMap(GraphDef* graph);
  NodeDef* GetNode(const string& name) const;
  const std::unordered_map<NodeDef*, int>& GetOutputs(
      const string& node_name) const;

 private:
  GraphDef* graph_;
  std::unordered_map<NodeDef*, int> empty_map_;
  std::unordered_map<string, NodeDef*> nodes_;
  std::unordered_map<string, std::unordered_map<NodeDef*, int>> outputs_;
};

// A vector with a set. The set stores the same elements as the vector, and
// quickly answers whether a value is in the vector. Duplicated elements are not
// allowed for now.
template <class T>
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
  std::unordered_set<T> set_;
  std::vector<T> vector_;
};

// True iff 'name' refers to a control inputs, i.e. a node name prefixed with
// the ^ character.
bool IsControlInput(const string& name);

// True iff 'name1' and 'name2' refer to the same input.
bool IsSameInput(const string& name1, const string& name2);

// Return the node name corresponding to 'name' if name is valid, or the empty
// string otherwise.
string NodeName(const string& name);

// Get the trailing position number ":{digits}" (if any) of a node name.
int NodePosition(const string& name);

// Returns the node name and position in a single call.
string ParseNodeName(const string& name, int* position);

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

// Returns the number of outputs of a node according to its OpDef. Note that
// some of the outputs may be unconnected.
int NumOutputs(const NodeDef& node);

// Number of connected non-control inputs.
int NumNonControlInputs(const NodeDef& node);

// Number of connected non-control outputs.
int NumNonControlOutputs(const NodeDef& node, const NodeMap& node_map);

// Returns the data type in attribute `attr_name` of `node`. If that attribute
// doesn't exist, returns DT_INVALID.
DataType GetDataTypeFromAttr(const NodeDef& node, const string& attr_name);

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

class SimpleGraphView {
 public:
  Status Initialize(const GraphDef& graph) {
    return Initialize(graph, true, true);
  }
  Status Initialize(const GraphDef& graph, bool dedup_inputs,
                    bool dedup_outputs);

  inline int num_nodes() const { return index_to_name_.size(); }
  inline const int index(const string& node_name) const {
    const auto& it = name_to_index_.find(node_name);
    DCHECK(it != name_to_index_.end());
    return it == name_to_index_.end() ? -1 : it->second;
  }
  inline const string& node_name(int node_idx) const {
    return index_to_name_[node_idx];
  }
  inline const gtl::InlinedVector<int, 4>& inputs(int node_idx) const {
    return inputs_[node_idx];
  }
  inline const gtl::InlinedVector<int, 2>& outputs(int node_idx) const {
    return outputs_[node_idx];
  }

  string PrintToString() const;

 private:
  std::vector<string> index_to_name_;
  std::unordered_map<string, int> name_to_index_;
  std::vector<gtl::InlinedVector<int, 4>> inputs_;
  std::vector<gtl::InlinedVector<int, 2>> outputs_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_UTILS_H_

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_
#define TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_

#include <unordered_map>

#include "absl/types/optional.h"
#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// ValidateConfig returns OK iff config is valid.
Status ValidateConfig(const tf2xla::Config& config);

// Modifies <graph_def> to include placeholders for each fed tensor, and
// update references to the fed tensors to refer to the placeholders.
// The existing nodes referenced by the feeds are not removed or modified
// (except where their input edges are modified by the replacement of other
// feeds).
Status AddPlaceholdersForFeeds(
    const tf2xla::Config& config, const OpRegistryInterface* op_registry,
    std::unordered_map<string, string>* feed_remapping, GraphDef* graph_def);

// Returns in <out> a copy of <in>, pruned to only include fetches from
// <config>.
Status PruneGraphDefInto(const tf2xla::Config& config, const GraphDef& in,
                         GraphDef* out);

// Returns node:port for the given <id>.
string TensorIdToString(const tf2xla::TensorId& id);

// Updates the sharding of <n> based on the sharding of its neighbors.
// If <out_edges> is true, outgoing edges from <n> are considered; else incoming
// edges are considered.
Status SetNodeShardingFromNeighbors(Node* n, bool out_edges);

// Add an allowed data type to the AttrConstraint with the given name.
void AddDtypeToKernelDefConstraint(absl::string_view name, DataType dtype,
                                   KernelDef* kdef);

// Returns the next random seed to use for seeding xla rng.
uint32 GetXLARandomSeed();

// Indicates how a FunctionDef is associated with a graph node (e.g. the node is
// a function call, or the node has function attrs).
class AssociatedFunctionInfo {
 public:
  enum AssociatedFunctionType {
    kFunctionAttr = 0,
    kFunctionCallNode = 1,
    kSymbolicGradient = 2,
  };

  // The function is an attr of the node.
  static AssociatedFunctionInfo FunctionAttr(const string& func_name,
                                             const AttrValueMap& attrs,
                                             const string& attr_name) {
    return AssociatedFunctionInfo(kFunctionAttr, func_name, attrs, attr_name);
  }

  // The node is a function call.
  static AssociatedFunctionInfo FunctionCall(const string& func_name,
                                             const AttrValueMap& attrs) {
    // attr_name will not be used in this case.
    return AssociatedFunctionInfo(kFunctionCallNode, func_name, attrs,
                                  /*attr_name=*/"");
  }

  // The node is a SymbolicGradient op.
  static AssociatedFunctionInfo SymbolicGradient(const string& func_name,
                                                 const AttrValueMap& attrs) {
    // attr_name will not be used in this case.
    return AssociatedFunctionInfo(kSymbolicGradient, func_name, attrs,
                                  /*attr_name=*/"");
  }

  AssociatedFunctionType type() const { return type_; }

  const string& func_name() const { return func_name_; }

  const string& attr_name() const { return attr_name_; }

  const AttrValueMap& attrs() const { return attrs_; }

 private:
  AssociatedFunctionInfo(AssociatedFunctionType type, const string& func_name,
                         const AttrValueMap& attrs, const string& attr_name)
      : type_(type),
        func_name_(func_name),
        attrs_(attrs),
        attr_name_(attr_name) {}

  // Available for all instances.
  AssociatedFunctionType type_;
  string func_name_;
  AttrValueMap attrs_;

  // Only available if the function is defined in an attr.
  string attr_name_;
};

// Returns if the NodeDef has associated function.
bool HasAssociatedFunction(const NodeDef& node_def,
                           const FunctionLibraryDefinition* fld);

// Gets functions associated with the node. Current cases:
// 1. For function call node, its function name;
// 2. For SymbolicGradient op, returned func_name will be "SymbolicGradient",
//    and returned attrs will be this node's attributes;
// 3. For nodes like XlaWhile/XlaIf, all their function attributes.
std::vector<AssociatedFunctionInfo> GetAssociatedFunctions(
    const Node& node, const FunctionLibraryDefinition* fld);

// Changes associated functions for the node. Current cases:
// 1. For function call node, creates a new node with the new function name and
//    remove the old node;
// 2. For SymbolicGradient op, add or replace GradientDef in
//    FunctionLibraryDefinition;
// 3. For nodes like XlaWhile/XlaIf, modify their function attributes.
Status RewriteAssociatedFunction(
    Graph* graph, Node* node, FunctionLibraryDefinition* fld,
    const AssociatedFunctionInfo& associated_function,
    const string& rewritten_function_name);

// Attribute to mark nodes to be executed on host.
extern const char kXlaOutsideCompilationAttrName[];

// Class to act as cache for FunctionLibraryRuntime::Handle objects.
class CachedFunctionHandles {
 public:
  CachedFunctionHandles(FunctionLibraryRuntime* flr) : flr_(flr) {}

  // Populates `handle` for requested function and attributes. If we have
  // instantiated the function with the same attributes before, `handle` will be
  // cached handle; otherwise instantiate the function and populate `handle`.
  Status GetOrInstantiate(const string& func_name, AttrSlice attrs,
                          FunctionLibraryRuntime::Handle* handle);

  // Releases all handles in the cache. Returns first non-OK status if any;
  // returns OK otherwise.
  Status ReleaseAllHandles();

  ~CachedFunctionHandles() { ReleaseAllHandles().IgnoreError(); }

 private:
  FunctionLibraryRuntime* flr_;
  std::map<string, FunctionLibraryRuntime::Handle> handles_;

  TF_DISALLOW_COPY_AND_ASSIGN(CachedFunctionHandles);
};

// Struct for node's output edge info.
struct OutEdgeInfo {
  Node* dst;
  int src_output, dst_input;
};

// Replaces node `n` with a new node whose NodeDef is `node_def`.
xla::StatusOr<Node*> ReplaceNode(Graph* g, Node* n, const NodeDef& node_def);

// Helper function that builds an Identity node.
xla::StatusOr<Node*> BuildIdentityNode(Graph* graph, const string& node_name,
                                       DataType dtype, const Node* input,
                                       absl::optional<string> requested_device);

// For "If"/"While" nodes, if some of their inputs are Const nodes, rewrite
// body functions to use the Const nodes instead of original _Arg nodes.
//
// For example, say we have the following computation:
//     shape = constant_op.constant([1])
//     return tf.cond(pred, lambda: tf.ones(shape), lambda: tf.zeros(shape))
// If we do not rewrite then/else function, they will use _Arg node as shape
// input for tf.ones/tf.zeros. But XLA requires that shape input to be compile
// time constant, so XLA compilation will fail. This rewriting process will
// change the shape input to Const node.
Status PropagateConstIntoFunctionalNodes(
    Graph* g, const FunctionLibraryDefinition* lookup_fld,
    FunctionLibraryDefinition* fld);

// Prunes unreachable FunctionDefs from FunctionLibraryDefinition.
Status PruneUnreachableFunctionsFromGraph(const Graph& g,
                                          FunctionLibraryDefinition* fld);

// Finds the following pattern in the graph:
// 1) EmptyTensorList -> forward While op -> backward While op,
// 2) in forward While op, a Const node is pushed,
// 3) in backward While op, data is popped from the tensor list.
// And rewrites backward While op to use Const node instead of TensorListPopBack
// result.
// TODO(b/128633174) remove the TensorList and related TensorList ops.
Status RewriteTensorListWithConstElement(Graph* g,
                                         FunctionLibraryDefinition* fld);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_TF2XLA_UTIL_H_

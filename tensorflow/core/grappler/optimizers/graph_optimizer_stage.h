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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_STAGE_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_STAGE_H_

#include <unordered_map>
#include <unordered_set>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

struct NodeScopeAndName {
  string scope;
  string name;
};

// Parse scope and name: "a/b/c/Add_1" -> {"a/b/c", "Add_1"}
const NodeScopeAndName ParseNodeScopeAndName(const string& node_name);

// Context owned by GraphOptimizer, and passed to every stage at construction
// time. Each optimizer stage is responsible for updating it according to the
// changes it made to the graph.
//
// If an optimizer needs access to some helper class that is not present in this
// context, consider creating an extension context, specific to that
// optimizer (see example of ArithmeticOptimizerContext). GraphOptimizerContext
// should only have members that are useful to almost all optimizers.
struct GraphOptimizerContext {
  GraphOptimizerContext(const std::unordered_set<string>* nodes_to_preserve,
                        GraphDef* optimized_graph,
                        GraphProperties* graph_properties, NodeMap* node_map,
                        gtl::FlatSet<string>* feed_nodes,
                        RewriterConfig::Toggle opt_level)
      : nodes_to_preserve(nodes_to_preserve),
        optimized_graph(optimized_graph),
        graph_properties(graph_properties),
        node_map(node_map),
        feed_nodes(feed_nodes),
        opt_level(opt_level) {}

  const std::unordered_set<string>* nodes_to_preserve;
  GraphDef* optimized_graph;
  GraphProperties* graph_properties;
  NodeMap* node_map;
  gtl::FlatSet<string>* feed_nodes;
  RewriterConfig::Toggle opt_level;
};

Status GetInputNode(const GraphOptimizerContext& ctx, const string& input,
                    NodeDef** node);
Status GetTensorProperties(const GraphOptimizerContext& ctx,
                           const string& tensor,
                           const OpInfo::TensorProperties** properties);

NodeDef* AddCopyNode(const GraphOptimizerContext& ctx, const string& name,
                     const NodeDef* node_to_copy);
NodeDef* AddEmptyNode(const GraphOptimizerContext& ctx, const string& name);

// WARNING:
// Optimizer stage must try to re-use original nodes of a graph and
// make all updates in place. This helps to make robust node placement
// decisions. Create new nodes only if there is a reason for that.

// Make a name for a new node obtained by optimizing a single node of the
// original graph. The optimized node is placed under the original node scope.
//
// Node name uniqueness is guaranteed by unique name of an original node in
// a same scope.
//
// Empty sub_scope or prefix ignored. At least one of them must be non-empty.
//
// Example: a/b/c/Add -> a/b/c/${sub_scope}/${prefix}_Add.
const string MakeOptimizedNodeName(const NodeScopeAndName& node,
                                   const string& sub_scope,
                                   const string& prefix);
// Make a name for a new node obtained by optimizing multiple nodes of the
// original graph, starting from "root". The optimized node is placed under
// the original scope of a "root" node.
//
// Example: [a/b/c/Add, x/y/z/Mul] -> a/b/c/${sub_scope}/${prefix}_Add_Mul
const string MakeOptimizedNodeName(const NodeScopeAndName& root,
                                   const std::vector<string> node_names,
                                   const string& sub_scope,
                                   const string& prefix);

// Base class for multi-stage GraphOptimizers (ArithmeticOptimizer, etc...).
//
// If a graph optimizer consists of large number of small independent
// rewrites, each of them should be implemented as a separate stage.
//
// * Result:
// Each graph optimizer choose what result is reported by each stage
// (e.g. each stage can fill in the name of optimized nodes, or have more
// complex result).
template <typename Result>
class GraphOptimizerStage {
 public:
  explicit GraphOptimizerStage(const string& optimizer_name,
                               const string& stage_name,
                               const GraphOptimizerContext& ctx)
      : optimizer_name_(optimizer_name), stage_name_(stage_name), ctx_(ctx) {}
  virtual ~GraphOptimizerStage() = default;

  const string& stage_name() const { return stage_name_; }
  const string& optimizer_name() const { return optimizer_name_; }

  // Check if we should try to simplify node. Returning true doesn't
  // guarantee that node will be simplified.
  //
  // Should implement just a basic sanity check, without any expensive graph
  // traversals.
  virtual bool IsSupported(const NodeDef* node) const = 0;

  // Try to simplify the given node.
  //
  // Return error status only if some precondition is failed, or got an
  // incorrect graph. In every other case return Status:OK(), even if didn't
  // simplify anything.
  //
  // Report result using output argument. Each GraphOptimizer can choose it's
  // own Result type.
  // TODO(ezhulenev): if it will appear that Result output parameter is not
  // sufficiently useful (used with a reason by most optimizers), get rid of it,
  // and remove template parameter.
  virtual Status TrySimplify(NodeDef* node, Result* result) = 0;

  // Return InvalidArgumentError if node is not supported by the optimizer
  // stage.
  // TODO(ezhulenev): make this check part of non-virtual public API
  // (TrySimplify), and make virtual implementation protected.
  Status EnsureNodeIsSupported(const NodeDef* node) const {
    return IsSupported(node)
               ? Status::OK()
               : errors::InvalidArgument(
                     "Node ", node->name(), " is not supported by optimizer ",
                     optimizer_name_, " and stage ", stage_name_);
  }

  // Get a name for a new node, created by this stage, based on one or multiple
  // nodes of an original graph.
  const string OptimizedNodeName(const NodeScopeAndName& node) const {
    return MakeOptimizedNodeName(node, optimizer_name_, stage_name_);
  }
  const string OptimizedNodeName(const NodeScopeAndName& root,
                                 const std::vector<string>& nodes) const {
    return MakeOptimizedNodeName(root, nodes, optimizer_name_, stage_name_);
  }
  const string OptimizedNodeName(const NodeScopeAndName& node,
                                 const string& rewrite_rule) const {
    const string prefix = strings::StrCat(stage_name_, "_", rewrite_rule);
    return MakeOptimizedNodeName(node, optimizer_name_, prefix);
  }

  const string UniqueOptimizedNodeName(const NodeScopeAndName& node) {
    const string node_name = OptimizedNodeName(node);
    return UniqueNodeName(node_name);
  }
  const string UniqueOptimizedNodeName(const NodeScopeAndName& node,
                                       const string& rewrite_rule) {
    const string node_name = OptimizedNodeName(node, rewrite_rule);
    return UniqueNodeName(node_name);
  }

  // Get a node by input name from a node map. Return an error if node was not
  // found.
  Status GetInputNode(const string& input, NodeDef** node) const {
    return ::tensorflow::grappler::GetInputNode(ctx_, input, node);
  }
  // Lookup tensor properties by name. Tensor name might have non-zero port
  // number. Return an error if tensor node doesn't exists in a graph, or it
  // doesn't have properties defined for requested port.
  Status GetTensorProperties(
      const string& tensor, const OpInfo::TensorProperties** properties) const {
    return ::tensorflow::grappler::GetTensorProperties(ctx_, tensor,
                                                       properties);
  }

  NodeDef* AddCopyNode(const string& name, const NodeDef* node_to_copy) {
    return ::tensorflow::grappler::AddCopyNode(ctx_, name, node_to_copy);
  }
  NodeDef* AddEmptyNode(const string& name) {
    return ::tensorflow::grappler::AddEmptyNode(ctx_, name);
  }

 protected:
  const GraphOptimizerContext& ctx() const { return ctx_; }

 private:
  const string UniqueNodeName(absl::string_view name) {
    string node_name = string(name);
    while (ctx_.node_map->NodeExists(node_name)) {
      node_name = absl::StrCat(name, "_unique",
                               optimized_node_name_counter_.fetch_add(1));
    }

    return node_name;
  }

  const string optimizer_name_;
  const string stage_name_;
  const GraphOptimizerContext ctx_;
  std::atomic<int64> optimized_node_name_counter_ = {0};
};

template <typename Result>
class GraphOptimizerStagePipeline {
 public:
  // Break predicate specifies if a pipeline should stop early, and not pass
  // a node to the next registered optimizer stage, typically that should be the
  // case when a stage successfully optimized a node, and it wants to yield
  // control to the optimizer.
  explicit GraphOptimizerStagePipeline(
      const std::function<bool(const Result&)> break_predicate)
      : break_predicate_(break_predicate) {}

  // Add a stage to the pipeline. It should be called with the arguments for the
  // stage constructor:
  //
  //   pipeline.AddStage<FooStage>(constructor_arg1, constructor_arg2);
  //
  // Returns a reference to the added stage.
  template <typename T, typename... Args>
  T& AddStage(Args&&... args) {
    auto stage = new T(std::forward<Args>(args)...);
    stages_.push_back(std::unique_ptr<T>(stage));
    return *stage;
  }

  // Pass a node through all registered optimizer stages, until break predicate
  // is true.
  //
  // Return true, if pipeline exited after a break predicate was evaluated as
  // 'true', which typically means that a node was optimized by one of the
  // registered stages.
  //
  // Return false, if node was not optimized by any of registered stages.
  bool PassThroughAllStages(NodeDef* node, Result* result) {
    for (auto& stage : stages_) {
      if (stage->IsSupported(node)) {
        const Status stage_status = stage->TrySimplify(node, result);
        // Each stage must be "error safe" (just like exception safe). In
        // case of any error it must leave optimized graph unmodified.
        if (!stage_status.ok()) {
          VLOG(2) << "Failed to run optimizer " << stage->optimizer_name()
                  << ", stage " << stage->stage_name() << " node "
                  << node->name()
                  << ". Error: " << stage_status.error_message();
        }
        if (break_predicate_(*result)) return true;
      }
    }
    return false;
  }

  // Pass a node through all registered optimizer stages, until break predicate
  // is true or a stage fails.
  //
  // Returns any stage failure status, or else Status::OK().
  Status PassThroughAllStagesWithStatus(NodeDef* node, Result* result) {
    for (auto& stage : stages_) {
      if (!stage->IsSupported(node)) {
        continue;
      }
      const Status stage_status = stage->TrySimplify(node, result);
      if (!stage_status.ok()) {
        return stage_status;
      } else if (break_predicate_(*result)) {
        break;
      }
    }
    return Status::OK();
  }

  std::size_t NumStages() { return stages_.size(); }

  std::vector<string> StageNames() {
    std::vector<string> names;
    for (const auto& stage : stages_) {
      names.push_back(stage->stage_name());
    }
    return names;
  }

 private:
  std::vector<std::unique_ptr<GraphOptimizerStage<Result>>> stages_;
  std::function<bool(const Result&)> break_predicate_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizerStagePipeline);
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_GRAPH_OPTIMIZER_STAGE_H_

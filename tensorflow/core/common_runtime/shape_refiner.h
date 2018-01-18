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
#ifndef THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_

#include <vector>

#include "tensorflow/core/common_runtime/graph_runner.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace grappler {
class GraphProperties;
}

// This class stores extra inference information in addition to
// InferenceContext, such as inference tree for user-defined functions and node
// input and output types.
class ExtendedInferenceContext {
 public:
  ExtendedInferenceContext(
      std::unique_ptr<shape_inference::InferenceContext> ic, const Node* node)
      : inference_context_(std::move(ic)) {
    input_types_.reserve(node->num_inputs());
    for (int i = 0; i < node->num_inputs(); i++) {
      input_types_.push_back(node->input_type(i));
    }
    output_types_.reserve(node->num_outputs());
    for (int i = 0; i < node->num_outputs(); i++) {
      output_types_.push_back(node->output_type(i));
    }
  }

  const std::unordered_map<string, std::unique_ptr<ExtendedInferenceContext>>&
  nested_inferences() const {
    return nested_inferences_;
  }
  DataType input_type(int64 idx) const { return input_types_[idx]; }
  DataType output_type(int64 idx) const { return output_types_[idx]; }

  shape_inference::InferenceContext* get_context() {
    return inference_context_.get();
  }

  // Sets nested inference info.
  // For composite ops (user-defined functions) only.
  // Inference for trivial ops must not call this setter.
  void set_nested_inferences(
      std::unordered_map<string, std::unique_ptr<ExtendedInferenceContext>>
          inferences) {
    nested_inferences_ = std::move(inferences);
  }

 private:
  std::unique_ptr<shape_inference::InferenceContext> inference_context_;
  std::vector<DataType> input_types_;
  std::vector<DataType> output_types_;

  // Nested inferences for composite ops (user-defined functions).
  // Mapping key is nested node name.
  // For trivial ops this map must be empty.
  std::unordered_map<string, std::unique_ptr<ExtendedInferenceContext>>
      nested_inferences_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExtendedInferenceContext);
};

// ShapeRefiner performs shape inference for TensorFlow Graphs.  It is
// responsible for instantiating InferenceContext objects for each
// Node in the Graph, and providing/storing the 'input_tensor' Tensors
// used by Shape Inference functions, when available at graph
// construction time.
class ShapeRefiner {
 public:
  ShapeRefiner(int graph_def_version, const OpRegistryInterface* ops);

  // Same as ShapeRefiner(versions.producer(), ops)
  ShapeRefiner(const VersionDef& versions, const OpRegistryInterface* ops);

  ~ShapeRefiner();

  // Performs validation of 'node' and runs 'node's shape function,
  // storing its shape outputs.
  //
  // All inputs of 'node' must be added to ShapeRefiner prior to
  // adding 'node'.
  //
  // Returns an error if:
  //  - the shape function for 'node' was not registered.
  //  - 'node' was added before its inputs.
  //  - The shape inference function returns an error.
  Status AddNode(const Node* node);

  // Sets 'node's 'output_port' output to have shape 'shape'.
  //
  // Returns an error if 'node' was not previously added to this
  // object, if 'output_port' is invalid, or if 'shape' is
  // not compatible with the existing shape of the output.
  Status SetShape(const Node* node, int output_port,
                  shape_inference::ShapeHandle shape);

  // Update the input shapes of node in case the shapes of the fan-ins of 'node'
  // have themselves been modified (For example, in case of incremental shape
  // refinement). If 'relax' is true, a new shape with the broadest set of
  // information will be set as the new input (see InferenceContext::RelaxInput
  // for full details and examples). Sets refined to true if any shapes have
  // changed (in their string representations). Note that shapes may have been
  // updated to newer versions (but with identical string representations) even
  // if <*refined> is set to false.
  Status UpdateNode(const Node* node, bool relax, bool* refined);

  // Returns the InferenceContext for 'node', if present.
  shape_inference::InferenceContext* GetContext(const Node* node) const {
    auto it = node_to_context_.find(node);
    if (it == node_to_context_.end()) {
      return nullptr;
    }
    return it->second->get_context();
  }

  // Returns the ExtendedInferenceContext for 'node', if present.
  ExtendedInferenceContext* GetExtendedContext(const Node* node) const {
    auto it = node_to_context_.find(node);
    if (it == node_to_context_.end()) {
      return nullptr;
    }
    return it->second.get();
  }

  // Getters and setters for graph_def_version_.
  int32 graph_def_version() const { return graph_def_version_; }
  void set_graph_def_version(int32 version) { graph_def_version_ = version; }

  void set_require_shape_inference_fns(bool require_shape_inference_fns) {
    require_shape_inference_fns_ = require_shape_inference_fns;
  }
  void set_disable_constant_propagation(bool disable) {
    disable_constant_propagation_ = disable;
  }

  // Set function library to enable function shape inference.
  // Without function library, function inference always yields unknown shapes.
  // With this enabled, shape inference can take more time since it descends
  // into all function calls. It doesn't do inference once for each function
  // definition, but once for each function call.
  // The function library must outlive the shape refiner.
  void set_function_library_for_shape_inference(
      const tensorflow::FunctionLibraryDefinition* lib) {
    function_library_ = lib;
  }

  bool function_shape_inference_supported() const {
    return function_library_ != nullptr;
  }

  // Call this to keep nested shapes information for user-defined functions:
  // nested inferences will be available on the ExtendedInferenceContext for
  // each function node, forming a tree of shape inferences corresponding to the
  // tree of nested function calls. By default this setting is disabled, and
  // only the shapes for the top-level function node will be reported on the
  // InferenceContext for each function node, to reduce memory usage.
  //
  // This flag has no effect when the function inference is not enabled via
  // set_function_library_for_shape_inference.
  void set_keep_nested_shape_inferences() {
    keep_nested_shape_inferences_ = true;
  }

 private:
  friend class ShapeRefinerTest;
  friend class ::tensorflow::grappler::GraphProperties;

  // Returns true if the ranks and all dimensions of <s0> and <s1> are either
  // equal in value or both unknown.
  static bool SameDefinedShape(shape_inference::InferenceContext* c,
                               shape_inference::ShapeHandle s0,
                               shape_inference::ShapeHandle s1);

  // Returns true if the shapes and types stored in <*existing> are identical in
  // value to the shapes and types in <*updated>.
  static bool IsUpdatedShapesOrTypes(
      shape_inference::InferenceContext* c,
      const std::vector<shape_inference::ShapeAndType>& existing,
      const std::vector<shape_inference::ShapeAndType>& updated);

  // Performs shape inference for the given function_def within the
  // given outer_context. Internally it instantiates the function as a graph
  // and runs shape inference recursively on it with the input shapes provided
  // by the outer_context.
  //
  // Returns an error if:
  // - number of inputs/outputs on outer_context doesn't match the function_def
  //
  // On success:
  // - outer_context will contain output shapes inferred from input shapes
  // - outer_context will contain nested inferences collection, iff
  //   keep_nested_shapes is true
  Status InferShapesForFunction(const tensorflow::FunctionDef* function_def,
                                bool keep_nested_shapes,
                                ExtendedInferenceContext* outer_context);

  // Tries to infer tensor output based on the input shapes of the node. In some
  // cases, the shapes of the inputs are sufficient for inferring the contents
  // of the output tensor. For example, a Shape op with fully defined input
  // shapes can have its output tensor inferred.
  Status TryToInferTensorOutputFromInputShapes(const Edge* edge, Tensor* output,
                                               bool* success);

  // Extracts the subgraph ending at 'node' that is statically
  // computable and inserts into 'out_graph'. If statically computable,
  // 'is_constant_graph' will be true.
  Status ExtractConstantSubgraph(
      Node* node, Graph* out_graph, bool* is_constant_graph,
      std::vector<std::pair<string, Tensor>>* const_inputs) TF_MUST_USE_RESULT;

  Status EvaluateConstantTensorForEdge(const Node* node, int dst_idx,
                                       bool* evaluated, Tensor* result);

  // This function tries to materialize as much information about the 'node''s
  // dst_idx input as a statically computable shape, and the result may be
  // partially known, depending on what is statically inferable.
  //
  // This is called when node.input[dst_idx] is a tensor that is used to define
  // the shape of some other tensor (e.g., the second argument to Reshape is a
  // <shape> tensor, where each element of the shape tensor is a dimension of
  // the target tensor).  It returns in <result> a shape for that input.
  //
  // Unlike simply resolving node.input[dst_idx] to a constant and then
  // converting that to a shape, this function can return a partial shape. This
  // is useful for cases where the shape tensor is only partially defined, such
  // as with calls for: reshape(x, shape(y)) where shape(y) is partially
  // defined.
  //
  // The implementation has op implementations for ops commonly called on shape
  // tensors, and the implementations are specialized to shape tensors (namely,
  // the output is a vector).
  //
  // <target_context> is used when creating new DimensionHandle and ShapeHandle
  // objects.
  Status ConstantPartialShape(shape_inference::InferenceContext* target_context,
                              const Node* node, int dst_idx,
                              shape_inference::ShapeHandle* result);

  Status RunShapeFn(const Node* node, const OpRegistrationData* op_reg_data,
                    ExtendedInferenceContext* ec);

  int32 graph_def_version_;
  const OpRegistryInterface* const ops_registry_;

  // The lifetime of the tensors are bound to the runner, so it should be the
  // deleted after the tensors.
  GraphRunner graph_runner_;

  // Stores a map from a node to its ExtendedInferenceContext.
  std::unordered_map<const Node*, std::unique_ptr<ExtendedInferenceContext>>
      node_to_context_;

  // Holds a cache from 'tensor name' to the tensor that is
  // evaluatable as a constant expression.  This reduces repeated
  // execution of the entire constant subgraph as a graph is being
  // built up.  This could be changed to some kind of size-based LRU
  // cache to avoid consuming too much memory, if that eventually
  // becomes a concern.
  //
  // Only tensors less than 1KiB are currently stored in the cache.
  static constexpr int64 kMaxTensorSize = 1024;
  std::unordered_map<string, Tensor> const_tensor_map_;

  bool require_shape_inference_fns_ = true;
  bool disable_constant_propagation_ = false;

  // Function library is optional, but has to be set to enable function
  // shape inference.
  const tensorflow::FunctionLibraryDefinition* function_library_ = nullptr;

  // Determines whether to keep the nested shape inference info for user-
  // defined functions. By default that info is discarded to save memory.
  bool keep_nested_shape_inferences_ = false;

  // Cache the graph corresponding to each functin definition for which shapes
  // are refined.
  std::unordered_map<const FunctionDef*, std::unique_ptr<const Graph>>
      functions_;

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeRefiner);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_

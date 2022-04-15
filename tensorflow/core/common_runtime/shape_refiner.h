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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
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
// InferenceContext, such as node input and output types.
class ExtendedInferenceContext {
 public:
  ExtendedInferenceContext(
      std::unique_ptr<shape_inference::InferenceContext> ic, const Node* node)
      : inference_context_(std::move(ic)), op_(node->name()) {
    input_types_.reserve(node->num_inputs());
    for (int i = 0; i < node->num_inputs(); i++) {
      input_types_.push_back(node->input_type(i));
    }
    output_types_.reserve(node->num_outputs());
    for (int i = 0; i < node->num_outputs(); i++) {
      output_types_.push_back(node->output_type(i));
    }
  }

  DataType input_type(int64_t idx) const { return input_types_[idx]; }
  DataType output_type(int64_t idx) const { return output_types_[idx]; }

  shape_inference::InferenceContext* get_context() {
    return inference_context_.get();
  }

  std::string op() const { return op_; }

 private:
  std::unique_ptr<shape_inference::InferenceContext> inference_context_;
  std::string op_;
  std::vector<DataType> input_types_;
  std::vector<DataType> output_types_;

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
  void set_graph_def_version(int32_t version) { graph_def_version_ = version; }

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
  Status InferShapesForFunction(const FunctionDef* function_def,
                                AttrSlice attributes,
                                ExtendedInferenceContext* outer_context);

  // Performs shape inference for a node inside a function.
  //
  // 'outer_context' is the 'InferenceContext' for the function's call op.
  Status InferShapesForFunctionSubNode(
      const Node* node, shape_inference::InferenceContext* outer_context);

  // Performs validation of 'node' and runs 'node's shape function,
  // storing its shape outputs.
  //
  // All inputs of 'node' must be added to ShapeRefiner prior to
  // adding 'node'.
  //
  // Optionally, if 'node' is in a nested function, the 'InferenceContext' for
  // the call op of the function can be passed as 'outer_context' (pass nullptr
  // otherwise). This gets used to perform constant propagation across Arg nodes
  // by requesting the constant of value of the incoming tensor from the
  // 'outer_context'.
  //
  // Returns an error if:
  //  - the shape function for 'node' was not registered.
  //  - 'node' was added before its inputs.
  //  - The shape inference function returns an error.
  Status AddNodeInternal(const Node* node,
                         shape_inference::InferenceContext* outer_context);

  // Attempts to evaluate the 'dst_idx'-th input to 'node'. If the input edge
  // value can be evaluated, 'evaluated' is set to true and the value returned
  // in 'result'. Otherwise 'evaluated' is set to false.
  //
  // Optionally, if 'node' is in a nested function, the 'InferenceContext' for
  // the call op of the function can be passed as 'outer_context' (pass nullptr
  // otherwise). This gets used to perform constant propagation across Arg nodes
  // by requesting the constant of value of the incoming tensor from the
  // 'outer_context'.
  Status EvaluateConstantTensorForEdge(
      const Node* node, int dst_idx, bool* evaluated, Tensor* result,
      shape_inference::InferenceContext* outer_context);

  // Wrapper around EvaluateConstantTensorForEdge for scalar int32/int64 input
  // tensors. The caller is responsible for checking that the specified edge is
  // scalar and int32 or int64.
  //
  // Optionally, if 'node' is in a nested function, the 'InferenceContext' for
  // the call op of the function can be passed as 'outer_context' (pass nullptr
  // otherwise). This gets used to perform constant propagation across Arg nodes
  // by requesting the constant of value of the incoming tensor from the
  // 'outer_context'.
  Status EvaluateConstantIntScalarEdge(
      const Node* node, int dst_idx, bool* evaluated, int64_t* result,
      shape_inference::InferenceContext* outer_context);

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
  //
  // Optionally, if 'node' is in a nested function, the 'InferenceContext' for
  // the call op of the function can be passed as 'outer_context' (pass nullptr
  // otherwise). This gets used to perform constant propagation across Arg nodes
  // by requesting the constant of value of the incoming tensor from the
  // 'outer_context'.
  Status ConstantPartialShape(shape_inference::InferenceContext* target_context,
                              const Node* node, int dst_idx,
                              shape_inference::ShapeHandle* result,
                              shape_inference::InferenceContext* outer_context);

  // Implementation of ConstantPartialShape for StridedSlice nodes.
  //
  // Optionally, if 'node' is in a nested function, the 'InferenceContext' for
  // the call op of the function can be passed as 'outer_context' (pass nullptr
  // otherwise). This gets used to perform constant propagation across Arg nodes
  // by requesting the constant of value of the incoming tensor from the
  // 'outer_context'.
  Status PartialStridedSliceShape(
      Node* slice_node, shape_inference::InferenceContext* ctx,
      shape_inference::ShapeHandle* result,
      shape_inference::InferenceContext* outer_context);

  // Runs the shape function registered for the node's op type.
  //
  // Optionally, if 'node' is in a nested function, the 'InferenceContext' for
  // the call op of the function can be passed as 'outer_context' (pass nullptr
  // otherwise). This gets used to perform constant propagation across Arg nodes
  // by requesting the constant of value of the incoming tensor from the
  // 'outer_context'.
  Status RunShapeFn(const Node* node, const OpRegistrationData* op_reg_data,
                    ExtendedInferenceContext* ec,
                    shape_inference::InferenceContext* outer_context = nullptr);

  int32 graph_def_version_;
  const OpRegistryInterface* const ops_registry_;

  // The lifetime of the tensors are bound to the runner, so it should be the
  // deleted after the tensors.
  GraphRunner graph_runner_;

  // Stores a map from a node to its ExtendedInferenceContext.
  absl::flat_hash_map<const Node*, std::unique_ptr<ExtendedInferenceContext>,
                      hash<const Node*>>
      node_to_context_;

  // Holds a cache from 'tensor name' to the tensor that is
  // evaluatable as a constant expression.  This reduces repeated
  // execution of the entire constant subgraph as a graph is being
  // built up.  This could be changed to some kind of size-based LRU
  // cache to avoid consuming too much memory, if that eventually
  // becomes a concern.
  //
  // Only tensors less than 1KiB are currently stored in the cache.
  static constexpr int64_t kMaxTensorSize = 1024;
  std::unordered_map<string, Tensor> const_tensor_map_;

  bool require_shape_inference_fns_ = true;
  bool disable_constant_propagation_ = false;

  // Function library is optional, but has to be set to enable function
  // shape inference.
  const tensorflow::FunctionLibraryDefinition* function_library_ = nullptr;

  // Cache the graph corresponding to each function definition for which shapes
  // are refined.
  absl::flat_hash_map<const FunctionDef*, std::unique_ptr<const Graph>,
                      hash<const FunctionDef*>>
      functions_;

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeRefiner);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SHAPE_REFINER_H_

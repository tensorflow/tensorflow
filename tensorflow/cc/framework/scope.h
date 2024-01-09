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

#ifndef TENSORFLOW_CC_FRAMEWORK_SCOPE_H_
#define TENSORFLOW_CC_FRAMEWORK_SCOPE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

class Graph;
class GraphDef;
class NodeBuilder;
struct CompositeOpScopes;

/// @addtogroup core
/// @{

/// A `Scope` object represents a set of related TensorFlow ops that have the
/// same properties such as a common name prefix.
///
/// A Scope object is a container for TensorFlow Op properties. Op constructors
/// get a Scope object as a mandatory first argument and the constructed op
/// acquires the properties in the object.
///
/// A simple example:
///
///     using namespace ops;
///     Scope root = Scope::NewRootScope();
///     auto c1 = Const(root, { {1, 1} });
///     auto m = MatMul(root, c1, { {41}, {1} });
///     GraphDef gdef;
///     Status s = root.ToGraphDef(&gdef);
///     if (!s.ok()) { ... }
///
/// Scope hierarchy:
///
/// The Scope class provides various With<> functions that create a new scope.
/// The new scope typically has one property changed while other properties are
/// inherited from the parent scope.
/// NewSubScope(name) method appends `name` to the prefix of names for ops
/// created within the scope, and WithOpName() changes the suffix which
/// otherwise defaults to the type of the op.
///
/// Name examples:
///
///     Scope root = Scope::NewRootScope();
///     Scope linear = root.NewSubScope("linear");
///     // W will be named "linear/W"
///     auto W = Variable(linear.WithOpName("W"),
///                       {2, 2}, DT_FLOAT);
///     // b will be named "linear/b_3"
///     int idx = 3;
///     auto b = Variable(linear.WithOpName("b_", idx),
///                       {2}, DT_FLOAT);
///     auto x = Const(linear, {...});  // name: "linear/Const"
///     auto m = MatMul(linear, x, W);  // name: "linear/MatMul"
///     auto r = BiasAdd(linear, m, b); // name: "linear/BiasAdd"
///
/// Scope lifetime:
///
/// A new scope is created by calling Scope::NewRootScope. This creates some
/// resources that are shared by all the child scopes that inherit from this
/// scope, directly or transitively. For instance, a new scope creates a new
/// Graph object to which operations are added when the new scope or its
/// children are used by an Op constructor. The new scope also has a Status
/// object which will be used to indicate errors by Op-constructor functions
/// called on any child scope. The Op-constructor functions have to check the
/// scope's status by calling the ok() method before proceeding to construct the
/// op.
///
/// Thread safety:
///
/// A `Scope` object is NOT thread-safe. Threads cannot concurrently call
/// op-constructor functions on the same `Scope` object.
class Scope {
 public:
  Scope(const Scope& other);
  ~Scope();
  Scope& operator=(const Scope& other);

  // The following functions are for users making graphs. They return brand new
  // scopes, or scopes derived from an existing scope object.

  /// Return a new scope.
  /// This creates a new graph and all operations constructed in this graph
  /// should use the returned object as the "root" scope.
  static Scope NewRootScope();

  /// Return a new scope. Ops created with this scope will have
  /// `name/child_scope_name` as the prefix. The actual name will be unique
  /// in the current scope. All other properties are inherited from the current
  /// scope. If `child_scope_name` is empty, the `/` is elided.
  Scope NewSubScope(const string& child_scope_name) const;

  /// Return a new scope. All ops created within the returned scope will have
  /// names of the form `name/StrCat(fragments...)[_suffix]`
  template <typename... Ty>
  Scope WithOpName(Ty... fragments) const {
    return WithOpNameImpl(absl::StrCat(fragments...));
  }

  /// Return a new scope. All ops created within the returned scope will have as
  /// control dependencies the union of operations in the control_deps vector
  /// and the control dependencies of the current scope.
  Scope WithControlDependencies(gtl::ArraySlice<Operation> control_deps) const;
  /// Same as above, but convenient to add control dependency on the operation
  /// producing the control_dep output.
  Scope WithControlDependencies(const Output& control_dep) const;

  /// Return a new scope. All ops created within the returned scope will have no
  /// control dependencies on other operations.
  Scope WithNoControlDependencies() const;

  /// Return a new scope. All ops created within the returned scope will have
  /// the device field set to 'device'.
  Scope WithDevice(const string& device) const;

  /// Returns a new scope.  All ops created within the returned scope will have
  /// their assigned device set to `assigned_device`.
  Scope WithAssignedDevice(const string& assigned_device) const;

  /// Returns a new scope.  All ops created within the returned scope will have
  /// their _XlaCluster attribute set to `xla_cluster`.
  Scope WithXlaCluster(const string& xla_cluster) const;

  /// Return a new scope. All ops created within the returned scope will be
  /// co-located on the device where op is placed.
  /// NOTE: This function is intended to be use internal libraries only for
  /// controlling placement of ops on to devices. Public use is not encouraged
  /// because the implementation of device placement is subject to change.
  Scope ColocateWith(const Operation& op) const;
  /// Convenience function for above.
  Scope ColocateWith(const Output& out) const { return ColocateWith(out.op()); }
  /// Clear all colocation constraints.
  Scope ClearColocation() const;

  /// Return a new scope. The op-constructor functions taking the returned scope
  /// as the scope argument will exit as soon as an error is detected, instead
  /// of setting the status on the scope.
  Scope ExitOnError() const;

  /// Return a new scope. All ops created with the new scope will have
  /// kernel_label as the value for their '_kernel' attribute;
  Scope WithKernelLabel(const string& kernel_label) const;

  // The following functions are for scope object consumers.

  /// Return a unique name, using default_name if an op name has not been
  /// specified.
  string GetUniqueNameForOp(const string& default_name) const;

  /// Update the status on this scope.
  /// Note: The status object is shared between all children of this scope.
  /// If the resulting status is not OkStatus() and exit_on_error_ is set on
  /// this scope, this function exits by calling LOG(FATAL).
  void UpdateStatus(const Status& s) const;

  // START_SKIP_DOXYGEN

  /// Update the builder with properties accumulated in this scope. Does not set
  /// status().
  // TODO(skyewm): NodeBuilder is not part of public API
  void UpdateBuilder(NodeBuilder* builder) const;
  // END_SKIP_DOXYGEN

  CompositeOpScopes GetCompositeOpScopes(const string& composite_op_name) const;

  bool ok() const;

  // TODO(skyewm): Graph is not part of public API
  Graph* graph() const;

  // TODO(skyewm): Graph is not part of public API
  std::shared_ptr<Graph> graph_as_shared_ptr() const;

  Status status() const;

  /// If status() is ok, convert the Graph object stored in this scope
  /// to a GraphDef proto and return an ok Status. Otherwise, return the error
  /// status as is without performing GraphDef conversion. If
  /// `include_debug_info` is true, populate the `debug_info` field of the
  /// GraphDef from stack traces in this Graph.
  Status ToGraphDef(GraphDef* gdef, bool include_debug_info = false) const;

  // START_SKIP_DOXYGEN

  /// If status() is OkStatus(), construct a Graph object using `opts` as the
  /// GraphConstructorOptions, and return Status::OK if graph construction was
  /// successful. Otherwise, return the error status.
  // TODO(josh11b, keveman): Make this faster; right now it converts
  // Graph->GraphDef->Graph.  This cleans up the graph (e.g. adds
  // edges from the source and to the sink node, resolves back edges
  // by name), and makes sure the resulting graph is valid.
  Status ToGraph(
      Graph* g, GraphConstructorOptions opts = GraphConstructorOptions{}) const;

  // Calls AddNode() using this scope's ShapeRefiner. This exists in the public
  // API to prevent custom op wrappers from needing access to shape_refiner.h or
  // scope_internal.h.
  // TODO(skyewm): remove this from public API
  Status DoShapeInference(Node* node) const;

  // Creates a new root scope that causes all DoShapeInference() calls to return
  // OkStatus() (on the returned scope and any subscopes). Used for testing.
  // TODO(skyewm): fix tests that still require this and eventually remove, or
  // at least remove from public API
  static Scope DisabledShapeInferenceScope();
  // END_SKIP_DOXYGEN

  const std::vector<Operation>& control_deps() const;

  // START_SKIP_DOXYGEN
  class Impl;
  Impl* impl() { return impl_.get(); }
  const Impl* impl() const { return impl_.get(); }
  // END_SKIP_DOXYGEN

 private:
  Scope WithOpNameImpl(const string& op_name) const;

  friend class InternalScope;
  std::unique_ptr<Impl> impl_;
  explicit Scope(Impl*);
};

/// A helper struct to hold the scopes that would be used by a function
/// constructing a composite op.
struct CompositeOpScopes {
  /// Scope to be used for creating the local ops (primitive or other composite
  /// ops).
  Scope child;
  /// Scope to be used for creating the last op.
  Scope last;
};

// Creates a node of the given operation, with the given inputs, and assigns the
// result to output. This does not support the ability to add additional
// attributes.
Status CreateOutputWithScope(string op_name,
                             absl::Span<const ::tensorflow::Input> inputs,
                             const Scope& scope, Output* output);
/// @}

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_FRAMEWORK_SCOPE_H_

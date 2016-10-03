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

#ifndef THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_SCOPE_H_
#define THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_SCOPE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {

class GraphDef;
class NodeBuilder;
struct CompositeOpScopes;

// A `Scope` object represents a set of related TensorFlow ops that have the
// same properties such as a common name prefix.
// A Scope object is a container for TensorFlow Op properties. Op constructors
// get a Scope object as a mandatory first argument and the constructed op
// acquires the properties in the object.
//
// A simple example:
//
// using namespace ops;
// Scope root = Scope::NewRootScope();
// auto c1 = Const(root, {{1, 1}});
// auto m = MatMul(root, c1, {{41}, {1}});
// GraphDef gdef;
// Status s = root.ToGraphDef(&gdef);
// if (!s.ok()) { /* Handle error */ }
//
// Scope hierarchy:
// The Scope class provides various With<> functions that create a new scope.
// The new scope typically has one property changed while other properties are
// inherited from the parent scope.
// NewSubScope(name) method appends `name` to the prefix of names for ops
// created within the scope, and WithOpName() changes the suffix which
// otherwise defaults to the type of the op.
//
// Name examples:
// Scope root = Scope::NewRootScope();
// Scope linear = root.NewSubScope("linear");
// /* W will be named "linear/W" */
// auto W = Variable(linear.WithOpName("W"),
//                   {2, 2}, DT_FLOAT);
// /* b will be named "linear/b" */
// auto b = Variable(linear.WithOpName("b"),
//                   {2}, DT_FLOAT);
// auto x = Const(linear, {...});  // name: "linear/Const"
// auto m = MatMul(linear, x, W);  // name: "linear/MatMul"
// auto r = BiasAdd(linear, m, b); // name: "linear/BiasAdd"
//
// Scope lifetime:
// A new scope is created by calling Scope::NewRootScope. This creates some
// resources that are shared by all the child scopes that inherit from this
// scope, directly or transitively. For instance, a new scope creates a new
// Graph object to which operations are added when the new scope or its children
// are used by an Op constructor. The new scope also has a Status object which
// will be used to indicate errors by Op-constructor functions called on any
// child scope. The Op-constructor functions have to check the scope's status by
// calling the ok() method before proceeding to construct the op.
//
// Thread safety:
// A `Scope` object is NOT thread-safe. Threads cannot concurrently call
// op-constructor functions on the same `Scope` object.
class Scope {
 public:
  // The following functions are for users making graphs. They return brand new
  // scopes, or scopes derived from an existing scope object.

  // Return a new scope.
  // This creates a new graph and all operations constructed in this graph
  // should use the returned object as the "root" scope.
  static Scope NewRootScope();

  // Return a new scope. Ops created with this scope will have
  // <name>/<child_scope_name> as the prefix. The actual name will be unique
  // in the current scope. All other properties are inherited from the current
  // scope. If child_scope_name is empty, the '/' is elided.
  Scope NewSubScope(const string& child_scope_name) const;

  // Return a new scope. All ops created within the returned scope will have
  // names of the form <name>/<op_name>[_<suffix].
  Scope WithOpName(const string& op_name) const;

  // Return a new scope. All ops created within the returned scope will have as
  // control dependencies the union of operations in the control_deps vector and
  // the control dependencies of the current scope.
  Scope WithControlDependencies(
      const gtl::ArraySlice<ops::Operation>& control_deps) const;
  // Same as above, but convenient to add control dependency on the operation
  // producing the control_dep output.
  Scope WithControlDependencies(const ops::Output& control_dep) const;

  // Return a new scope. All ops created within the returned scope will have no
  // control dependencies on other operations.
  Scope WithNoControlDependencies() const;

  // Return a new scope. All ops created within the returned scope will have the
  // device field set to 'device'.
  Scope WithDevice(const string& device) const;

  // Return a new scope. All ops created within the returned scope will be
  // co-located on the device where op is placed.
  // NOTE: This function is intended to be use internal libraries only for
  // controlling placement of ops on to devices. Public use is not encouraged
  // because the implementation of device placement is subject to change.
  Scope ColocateWith(const ops::Operation& op) const;
  // Convenience function for above.
  Scope ColocateWith(const ops::Output& out) const {
    return ColocateWith(out.op());
  }
  // Clear all colocation constraints.
  Scope ClearColocation() const;

  // Return a new scope. The op-constructor functions taking the returned scope
  // as the scope argument will exit as soon as an error is detected, instead of
  // setting the status on the scope.
  Scope ExitOnError() const;

  // Return a new scope. All ops created with the new scope will have
  // kernel_label as the value for their '_kernel' attribute;
  Scope WithKernelLabel(const string& kernel_label) const;

  // The following functions are for scope object consumers.

  // Return a unique name, using default_name if an op name has not been
  // specified.
  string GetUniqueNameForOp(const string& default_name) const;

  // Update the status on this scope.
  // Note: The status object is shared between all children of this scope.
  // If the resulting status is not Status::OK() and exit_on_error_ is set on
  // this scope, this function exits by calling LOG(FATAL).
  void UpdateStatus(const Status s) const;

  // Update the builder with properties accumulated in this scope.
  void UpdateBuilder(NodeBuilder* builder) const;

  CompositeOpScopes GetCompositeOpScopes(const string& composite_op_name) const;

  bool ok() const { return status_->ok(); }

  Graph* graph() const { return graph_.get(); }

  ShapeRefiner* refiner() const { return refiner_.get(); }

  std::shared_ptr<Graph> graph_as_shared_ptr() const { return graph_; }

  Status status() const { return *status_; }

  // If status() is Status::OK(), convert the Graph object stored in this scope
  // to a GraphDef proto and return Status::OK(). Otherwise, return the error
  // status as is without performing GraphDef conversion.
  Status ToGraphDef(GraphDef* gdef) const;

  // If status() is Status::OK(), construct a Graph object using the default
  // GraphConstructorOptions, and return Status::OK if graph construction was
  // successful. Otherwise, return the error status.
  // TODO(josh11b, keveman): Make this faster; right now it converts
  // Graph->GraphDef->Graph.  This cleans up the graph (e.g. adds
  // edges from the source and to the sink node, resolves back edges
  // by name), and makes sure the resulting graph is valid.
  Status ToGraph(Graph* g) const;

  const std::vector<ops::Operation>& control_deps() const {
    return control_deps_;
  }

 private:
  // Tag types to choose the constructor to dispatch.
  struct Tags {
    enum class ScopeName;
    enum class OpName;
    enum class ControlDeps;
    enum class Device;
    enum class SingleUseScope;
    enum class ExitOnError;
    enum class KernelLabel;
    enum class Colocate;
  };

  // A NameMap is used to keep track of suffixes for names used in a scope. A
  // name that has not been used so far in a scope will get no suffix. Later
  // uses of the same name will get suffixes _1, _2, _3, etc. Multiple scopes
  // can be sharing the same NameMap. For instance, a new scope created using
  // WithControlDependencies() should would share the same NameMap with the
  // parent.
  typedef std::unordered_map<string, int> NameMap;

  Scope(Graph* graph, Status* status, NameMap* name_map, ShapeRefiner* refiner);
  Scope(const Scope& other, Tags::ScopeName, const string& name,
        bool copy_names);
  Scope(const Scope& other, Tags::OpName, const string& name,
        const string& op_name);
  Scope(const Scope& other, Tags::ControlDeps,
        std::vector<ops::Operation> control_deps, bool clear_control_deps);
  Scope(const Scope& other, Tags::Device, const string& device);
  Scope(const Scope& other, Tags::SingleUseScope, const string& op_name);
  Scope(const Scope& other, Tags::ExitOnError);
  Scope(const Scope& other, Tags::KernelLabel, const string& kernel_label);
  Scope(const Scope& other, Tags::Colocate,
        const ops::Operation& colocate_with_op, bool clear_colocations);

  std::unordered_set<string> GetColocationConstraints(
      const ops::Operation& colocate_with_op) const;

  // Helper functions to get a unique names.
  string GetUniqueName(const string& prefix, bool check_single_use) const;
  string GetNameForOp(const string& default_name) const;

  bool single_use_scope() const { return scope_used_ != nullptr; }

  // The graph, status, and name maps are shared by all child scopes
  // created from a single 'root' scope. A root scope is created by calling the
  // Scope::NewRootScope function, which creates a new graph, a new status and
  // the name maps.
  std::shared_ptr<Graph> graph_ = nullptr;
  std::shared_ptr<Status> status_ = nullptr;
  std::shared_ptr<NameMap> name_map_ = nullptr;
  std::shared_ptr<ShapeRefiner> refiner_ = nullptr;

  // If scope_used_ is not nullptr, op_name_ should be empty and
  // GetUniqueNameForOp can only be called once on this scope. More calls to
  // GetUniqueNameForOp will cause an error status to be set on this scope.
  std::shared_ptr<bool> scope_used_ = nullptr;

  const std::vector<ops::Operation> control_deps_;

  const string name_ = "";
  const string op_name_ = "";
  const bool exit_on_error_ = false;
  const string kernel_label_ = "";
  const string device_ = "";
  const std::unordered_set<string> colocation_constraints_;
};

// A helper struct to hold the scopes that would be used by a function
// constructing a composite op.
struct CompositeOpScopes {
  // Scope to be used for creating the local ops (primitive or other composite
  // ops).
  Scope child;
  // Scope to be used for creating the last op.
  Scope last;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_SCOPE_H_

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

#ifndef THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_SCOPE_INTERNAL_H_
#define THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_SCOPE_INTERNAL_H_

#include "tensorflow/cc/framework/scope.h"

namespace tensorflow {

class ShapeRefiner;

// NewInternalScope returns a new scope which doesn't take ownership of
// graph, status, name_map, and refiner.
// This is intended to enable the C API (which are used by other language
// bindings) to create a Scope and access C++ functionality (i.e. gradients).
Scope NewInternalScope(Graph* graph, Status* status, ShapeRefiner* refiner);

class Scope::Impl {
 public:
  // A NameMap is used to keep track of suffixes for names used in a scope. A
  // name that has not been used so far in a scope will get no suffix. Later
  // uses of the same name will get suffixes _1, _2, _3, etc. Multiple scopes
  // can share the same NameMap. For instance, a new scope created using
  // WithControlDependencies() should would share the same NameMap with the
  // parent.
  typedef std::unordered_map<string, int> NameMap;

  Impl(const std::shared_ptr<Graph>& graph,
       const std::shared_ptr<Status>& status,
       const std::shared_ptr<NameMap>& name_map,
       const std::shared_ptr<ShapeRefiner>& refiner);

  const string& name() const { return name_; }
  const std::vector<Operation>& control_deps() const { return control_deps_; }

 private:
  friend class Scope;

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

  Impl(Graph* graph, Status* status, NameMap* name_map, ShapeRefiner* refiner,
       bool disable_shape_inference);
  Impl(const Scope& other, Tags::ScopeName, const string& name,
       bool copy_names);
  Impl(const Scope& other, Tags::OpName, const string& name,
       const string& op_name);
  Impl(const Scope& other, Tags::ControlDeps,
       std::vector<Operation> control_deps, bool clear_control_deps);
  Impl(const Scope& other, Tags::Device, const string& device);
  Impl(const Scope& other, Tags::SingleUseScope, const string& op_name);
  Impl(const Scope& other, Tags::ExitOnError);
  Impl(const Scope& other, Tags::KernelLabel, const string& kernel_label);
  Impl(const Scope& other, Tags::Colocate, const Operation& colocate_with_op,
       bool clear_colocations);

  std::unordered_set<string> GetColocationConstraints(
      const Operation& colocate_with_op) const;

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

  const std::vector<Operation> control_deps_;

  // The fully-qualified name of this scope (i.e. includes any parent scope
  // names).
  const string name_ = "";
  const string op_name_ = "";
  const bool exit_on_error_ = false;
  const string kernel_label_ = "";
  const string device_ = "";
  const std::unordered_set<string> colocation_constraints_;

  // If true, Scope::DoShapeInference() always returns Status:OK().
  // TODO(skyewm): remove this when possible
  const bool disable_shape_inference_;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CC_FRAMEWORK_SCOPE_INTERNAL_H_

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

#include <algorithm>
#include <vector>

#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

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

  Impl(Graph* graph, Status* status, NameMap* name_map, ShapeRefiner* refiner);
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

  const string name_ = "";
  const string op_name_ = "";
  const bool exit_on_error_ = false;
  const string kernel_label_ = "";
  const string device_ = "";
  const std::unordered_set<string> colocation_constraints_;
};

Scope::Scope(Impl* impl) : impl_(impl) {}

Scope::Scope(const Scope& other) : impl_(new Impl(*other.impl())) {}

Scope::~Scope() {}

Scope& Scope::operator=(const Scope& other) {
  // We can't copy Impls because of the const members, use copy ctor instead
  impl_.reset(new Impl(*other.impl_));
  return *this;
}

Scope::Impl::Impl(Graph* graph, Status* status, NameMap* name_map,
                  ShapeRefiner* refiner)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_() {}

Scope::Impl::Impl(const std::shared_ptr<Graph>& graph,
                  const std::shared_ptr<Status>& status,
                  const std::shared_ptr<NameMap>& name_map,
                  const std::shared_ptr<ShapeRefiner>& refiner)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_() {}

Scope Scope::NewRootScope() {
  Graph* graph = new Graph(OpRegistry::Global());
  ShapeRefiner* refiner =
      new ShapeRefiner(graph->versions().producer(), graph->op_registry());
  return Scope(new Impl(graph, new Status, new Impl::NameMap, refiner));
}

Scope::Impl::Impl(const Scope& other, Tags::ScopeName, const string& name,
                  bool copy_names)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(copy_names ? other.impl()->name_map_
                           : std::shared_ptr<NameMap>(new NameMap)),
      refiner_(other.impl()->refiner_),
      scope_used_(nullptr),
      control_deps_(other.impl()->control_deps_),
      name_(name),
      op_name_(""),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::OpName, const string& name,
                  const string& op_name)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(name),
      op_name_(op_name),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::ControlDeps,
                  std::vector<Operation> control_deps, bool clear_control_deps)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(
          clear_control_deps
              ? std::vector<Operation>()
              : (control_deps.insert(control_deps.begin(),
                                     other.impl()->control_deps_.begin(),
                                     other.impl()->control_deps_.end()),
                 control_deps)),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::Device, const string& device)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(device),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::SingleUseScope,
                  const string& op_name)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(new bool(false)),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(op_name),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::ExitOnError)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(true),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::KernelLabel,
                  const string& kernel_label)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(kernel_label),
      device_(other.impl()->device_),
      colocation_constraints_(other.impl()->colocation_constraints_) {}

Scope::Impl::Impl(const Scope& other, Tags::Colocate,
                  const Operation& colocate_with_op, bool clear_colocations)
    : graph_(other.impl()->graph_),
      status_(other.impl()->status_),
      name_map_(other.impl()->name_map_),
      refiner_(other.impl()->refiner_),
      scope_used_(other.impl()->scope_used_),
      control_deps_(other.impl()->control_deps_),
      name_(other.impl()->name_),
      op_name_(other.impl()->op_name_),
      exit_on_error_(other.impl()->exit_on_error_),
      kernel_label_(other.impl()->kernel_label_),
      device_(other.impl()->device_),
      colocation_constraints_(
          clear_colocations
              ? std::unordered_set<string>()
              : other.impl()->GetColocationConstraints(colocate_with_op)) {}

std::unordered_set<string> Scope::Impl::GetColocationConstraints(
    const Operation& colocate_with_op) const {
  std::unordered_set<string> current_constraints(colocation_constraints_);
  const AttrSlice attrs = colocate_with_op.node()->attrs();
  std::vector<string> node_constraints;
  if (GetNodeAttr(attrs, kColocationAttrName, &node_constraints).ok()) {
    for (const string& entry : node_constraints) {
      StringPiece s(entry);
      if (s.Consume(kColocationGroupPrefix)) {
        current_constraints.insert(s.ToString());
      }
    }
  } else {
    current_constraints.insert(colocate_with_op.node()->name());
  }
  return current_constraints;
}

bool Scope::ok() const { return impl()->status_->ok(); }

Graph* Scope::graph() const { return impl()->graph_.get(); }

std::shared_ptr<Graph> Scope::graph_as_shared_ptr() const {
  return impl()->graph_;
}

Status Scope::status() const { return *impl()->status_; }

const std::vector<Operation>& Scope::control_deps() const {
  return impl()->control_deps_;
}

void Scope::UpdateStatus(const Status s) const {
  impl()->status_->Update(s);
  if (impl()->exit_on_error_ && !ok()) {
    LOG(FATAL) << *impl()->status_;
  }
}

Status Scope::ToGraphDef(GraphDef* gdef) const {
  if (!ok()) {
    return *impl()->status_;
  }
  graph()->ToGraphDef(gdef);
  return Status::OK();
}

Status Scope::ToGraph(Graph* g) const {
  if (ok()) {
    GraphDef graph_def;
    graph()->ToGraphDef(&graph_def);
    GraphConstructorOptions opts;
    UpdateStatus(ConvertGraphDefToGraph(opts, graph_def, g));
  }
  return *impl()->status_;
}

void Scope::UpdateBuilder(NodeBuilder* builder) const {
  std::vector<Node*> control_inputs;
  for (const auto& op : impl()->control_deps_) {
    control_inputs.push_back(op.node());
  }
  builder->ControlInputs(control_inputs);

  if (!impl()->kernel_label_.empty()) {
    builder->Attr("_kernel", impl()->kernel_label_);
  }

  if (!impl()->colocation_constraints_.empty()) {
    std::vector<string> constraints(impl()->colocation_constraints_.begin(),
                                    impl()->colocation_constraints_.end());
    // Sort the set.
    std::sort(constraints.begin(), constraints.end());
    // Add loc:@ prefix
    std::transform(constraints.begin(), constraints.end(), constraints.begin(),
                   [](const string& s) {
                     return strings::StrCat(kColocationGroupPrefix, s);
                   });
    builder->Attr(kColocationAttrName, constraints);
  }
  if (!impl()->device_.empty()) {
    builder->Device(impl()->device_);
  }
}

string Scope::Impl::GetUniqueName(const string& prefix,
                                  bool check_single_use) const {
  if (check_single_use && single_use_scope()) {
    if (*scope_used_) {
      *status_ =
          errors::AlreadyExists(prefix, " already exists in the current scope");
      return "";
    }
    *scope_used_ = true;
    return prefix;
  }
  auto entry = name_map_->find(prefix);
  string unique_name = prefix;
  if (entry == name_map_->end()) {
    name_map_->insert({prefix, 0});
  } else {
    unique_name = strings::StrCat(unique_name, "_", ++entry->second);
  }
  return unique_name;
}

string Scope::Impl::GetNameForOp(const string& default_name) const {
  const string unique_name =
      GetUniqueName(default_name, true /* check_single_use */);
  const string sep = name_.empty() || unique_name.empty() ? "" : "/";
  return strings::StrCat(name_, sep, unique_name);
}

string Scope::GetUniqueNameForOp(const string& default_name) const {
  if (impl()->single_use_scope()) {
    if (impl()->op_name_.empty() || *impl()->scope_used_) {
      *impl()->status_ =
          errors::InvalidArgument("Cannot get a unique name in this scope");
      return "";
    }
    *impl()->scope_used_ = true;
    return impl()->op_name_;
  }
  return impl()->op_name_.empty() ? impl()->GetNameForOp(default_name)
                                  : impl()->GetNameForOp(impl()->op_name_);
}

Scope Scope::NewSubScope(const string& child_scope_name) const {
  if (child_scope_name.empty()) {
    return Scope(new Impl(*this, Impl::Tags::ScopeName(), impl()->name_,
                          true /* copy_names */));
  }
  const string unique_name =
      impl()->GetUniqueName(child_scope_name, false /* check_single_use */);
  const string sep = impl()->name_.empty() || unique_name.empty() ? "" : "/";
  return Scope(new Impl(*this, Impl::Tags::ScopeName(),
                        strings::StrCat(impl()->name_, sep, unique_name),
                        false /* copy_names */));
}

Scope Scope::WithOpName(const string& op_name) const {
  if (impl()->single_use_scope()) {
    UpdateStatus(errors::InvalidArgument("Cannot set op name ", op_name,
                                         " on this scope"));
    return *this;
  }
  return Scope(new Impl(*this, Impl::Tags::OpName(), impl()->name_, op_name));
}

Scope Scope::WithControlDependencies(
    const gtl::ArraySlice<Operation>& control_deps) const {
  return Scope(
      new Impl(*this, Impl::Tags::ControlDeps(),
               std::vector<Operation>(control_deps.begin(), control_deps.end()),
               /* clear_control_deps */ false));
}

Scope Scope::WithControlDependencies(const Output& control_dep) const {
  return Scope(new Impl(*this, Impl::Tags::ControlDeps(),
                        std::vector<Operation>(1, control_dep.op()),
                        /* clear_control_deps */ false));
}

Scope Scope::WithNoControlDependencies() const {
  return Scope(new Impl(*this, Impl::Tags::ControlDeps(),
                        std::vector<Operation>(),
                        /* clear_control_deps */ true));
}

Scope Scope::WithDevice(const string& device) const {
  return Scope(new Impl(*this, Impl::Tags::Device(), device));
}

Scope Scope::ColocateWith(const Operation& op) const {
  return Scope(new Impl(*this, Impl::Tags::Colocate(), op,
                        /* clear_colocations */ false));
}

Scope Scope::ClearColocation() const {
  return Scope(new Impl(*this, Impl::Tags::Colocate(), Operation(),
                        /* clear_colocations */ true));
}

Scope Scope::ExitOnError() const {
  return Scope(new Impl(*this, Impl::Tags::ExitOnError()));
}

Scope Scope::WithKernelLabel(const string& kernel_label) const {
  return Scope(new Impl(*this, Impl::Tags::KernelLabel(), kernel_label));
}

CompositeOpScopes Scope::GetCompositeOpScopes(
    const string& composite_op_name) const {
  if (impl()->op_name_.empty() && composite_op_name.empty()) {
    UpdateStatus(errors::InvalidArgument(
        "Cannot create composite op scopes with empty name"));
    return {*this, *this};
  }
  if (!impl()->single_use_scope()) {
    Scope child = NewSubScope(impl()->op_name_.empty() ? composite_op_name
                                                       : impl()->op_name_);
    const string child_op_sep = impl()->name_.empty() ? "" : "_";
    const string child_name =
        strings::StrCat(impl()->name_, child_op_sep, child.impl()->name_);
    return {child,
            Scope(new Impl(child, Impl::Tags::SingleUseScope(), child_name))};
  } else {
    return {Scope(new Impl(*this, Impl::Tags::ScopeName(), impl()->op_name_,
                           true /* copy_names */)),
            *this};
  }
}

class InternalScope {
 public:
  // NewScope doesn't take ownership of the inputs.
  static Scope NewScope(Graph* graph, Status* status, ShapeRefiner* refiner) {
    Scope::Impl::NameMap* name_map = new Scope::Impl::NameMap;
    for (const Node* node : graph->nodes()) {
      (*name_map)[node->name()] = 0;
    }
    // We provide null destructors for these shared ptrs (except for name_map)
    // since the caller owns them and doesn't want the scope to destroy them.
    return Scope(new Scope::Impl(
        std::shared_ptr<Graph>(graph, [](Graph*) {}),
        std::shared_ptr<Status>(status, [](Status*) {}),
        std::shared_ptr<Scope::Impl::NameMap>(name_map),
        std::shared_ptr<ShapeRefiner>(refiner, [](ShapeRefiner*) {})));
  }
};

Scope NewInternalScope(Graph* graph, Status* status, ShapeRefiner* refiner) {
  return InternalScope::NewScope(graph, status, refiner);
}

}  // namespace tensorflow

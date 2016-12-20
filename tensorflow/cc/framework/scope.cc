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

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"

namespace tensorflow {

Scope::Scope(Graph* graph, Status* status, Scope::NameMap* name_map,
             ShapeRefiner* refiner)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_() {}

Scope Scope::NewRootScope() {
  Graph* graph = new Graph(OpRegistry::Global());
  ShapeRefiner* refiner = new ShapeRefiner(graph->op_registry());
  return Scope(graph, new Status, new Scope::NameMap, refiner);
}

Scope::Scope(const Scope& other, Scope::Tags::ScopeName, const string& name,
             bool copy_names)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(copy_names ? other.name_map_
                           : std::shared_ptr<NameMap>(new NameMap)),
      refiner_(other.refiner_),
      scope_used_(nullptr),
      control_deps_(other.control_deps_),
      name_(name),
      op_name_(""),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(other.kernel_label_),
      device_(other.device_),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::OpName, const string& name,
             const string& op_name)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(other.scope_used_),
      control_deps_(other.control_deps_),
      name_(name),
      op_name_(op_name),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(other.kernel_label_),
      device_(other.device_),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::ControlDeps,
             std::vector<ops::Operation> control_deps, bool clear_control_deps)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(other.scope_used_),
      control_deps_(clear_control_deps
                        ? std::vector<ops::Operation>()
                        : (control_deps.insert(control_deps.begin(),
                                               other.control_deps_.begin(),
                                               other.control_deps_.end()),
                           control_deps)),
      name_(other.name_),
      op_name_(other.op_name_),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(other.kernel_label_),
      device_(other.device_),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::Device, const string& device)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(other.scope_used_),
      control_deps_(other.control_deps_),
      name_(other.name_),
      op_name_(other.op_name_),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(other.kernel_label_),
      device_(device),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::SingleUseScope,
             const string& op_name)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(new bool(false)),
      control_deps_(other.control_deps_),
      name_(other.name_),
      op_name_(op_name),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(other.kernel_label_),
      device_(other.device_),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::ExitOnError)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(other.scope_used_),
      control_deps_(other.control_deps_),
      name_(other.name_),
      op_name_(other.op_name_),
      exit_on_error_(true),
      kernel_label_(other.kernel_label_),
      device_(other.device_),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::KernelLabel,
             const string& kernel_label)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(other.scope_used_),
      control_deps_(other.control_deps_),
      name_(other.name_),
      op_name_(other.op_name_),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(kernel_label),
      device_(other.device_),
      colocation_constraints_(other.colocation_constraints_) {}

Scope::Scope(const Scope& other, Scope::Tags::Colocate,
             const ops::Operation& colocate_with_op, bool clear_colocations)
    : graph_(other.graph_),
      status_(other.status_),
      name_map_(other.name_map_),
      refiner_(other.refiner_),
      scope_used_(other.scope_used_),
      control_deps_(other.control_deps_),
      name_(other.name_),
      op_name_(other.op_name_),
      exit_on_error_(other.exit_on_error_),
      kernel_label_(other.kernel_label_),
      device_(other.device_),
      colocation_constraints_(
          clear_colocations
              ? std::unordered_set<string>()
              : other.GetColocationConstraints(colocate_with_op)) {}

std::unordered_set<string> Scope::GetColocationConstraints(
    const ops::Operation& colocate_with_op) const {
  std::unordered_set<string> current_constraints(colocation_constraints_);
  const NodeDef& node_def = colocate_with_op.node()->def();
  std::vector<string> node_constraints;
  if (GetNodeAttr(node_def, kColocationAttrName, &node_constraints).ok()) {
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

void Scope::UpdateStatus(const Status s) const {
  status_->Update(s);
  if (exit_on_error_ && !status_->ok()) {
    LOG(FATAL) << status_;
  }
}

Status Scope::ToGraphDef(GraphDef* gdef) const {
  if (!status_->ok()) {
    return *status_;
  }
  graph()->ToGraphDef(gdef);
  return Status::OK();
}

Status Scope::ToGraph(Graph* g) const {
  if (status_->ok()) {
    GraphDef graph_def;
    graph()->ToGraphDef(&graph_def);
    GraphConstructorOptions opts;
    UpdateStatus(ConvertGraphDefToGraph(opts, graph_def, g));
  }
  return *status_;
}

void Scope::UpdateBuilder(NodeBuilder* builder) const {
  std::vector<Node*> control_inputs;
  for (const auto& op : control_deps_) {
    control_inputs.push_back(op.node());
  }
  builder->ControlInputs(control_inputs);

  if (!kernel_label_.empty()) {
    builder->Attr("_kernel", kernel_label_);
  }

  if (!colocation_constraints_.empty()) {
    std::vector<string> constraints(colocation_constraints_.begin(),
                                    colocation_constraints_.end());
    // Sort the set.
    std::sort(constraints.begin(), constraints.end());
    // Add loc:@ prefix
    std::transform(constraints.begin(), constraints.end(), constraints.begin(),
                   [](const string& s) {
                     return strings::StrCat(kColocationGroupPrefix, s);
                   });
    builder->Attr(kColocationAttrName, constraints);
  }
  if (!device_.empty()) {
    builder->Device(device_);
  }
}

string Scope::GetUniqueName(const string& prefix, bool check_single_use) const {
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

string Scope::GetNameForOp(const string& default_name) const {
  const string unique_name =
      GetUniqueName(default_name, true /* check_single_use */);
  const string sep = name_.empty() || unique_name.empty() ? "" : "/";
  return strings::StrCat(name_, sep, unique_name);
}

string Scope::GetUniqueNameForOp(const string& default_name) const {
  if (single_use_scope()) {
    if (op_name_.empty() || *scope_used_) {
      *status_ =
          errors::InvalidArgument("Cannot get a unique name in this scope");
      return "";
    }
    *scope_used_ = true;
    return op_name_;
  }
  return op_name_.empty() ? GetNameForOp(default_name) : GetNameForOp(op_name_);
}

Scope Scope::NewSubScope(const string& child_scope_name) const {
  if (child_scope_name.empty()) {
    return Scope(*this, Scope::Tags::ScopeName(), name_, true /* copy_names */);
  }
  const string unique_name =
      GetUniqueName(child_scope_name, false /* check_single_use */);
  const string sep = name_.empty() || unique_name.empty() ? "" : "/";
  return Scope(*this, Scope::Tags::ScopeName(),
               strings::StrCat(name_, sep, unique_name),
               false /* copy_names */);
}

Scope Scope::WithOpName(const string& op_name) const {
  if (single_use_scope()) {
    UpdateStatus(errors::InvalidArgument("Cannot set op name ", op_name,
                                         " on this scope"));
    return *this;
  }
  return Scope(*this, Scope::Tags::OpName(), name_, op_name);
}

Scope Scope::WithControlDependencies(
    const gtl::ArraySlice<ops::Operation>& control_deps) const {
  return Scope(
      *this, Scope::Tags::ControlDeps(),
      std::vector<ops::Operation>(control_deps.begin(), control_deps.end()),
      /* clear_control_deps */ false);
}

Scope Scope::WithControlDependencies(const ops::Output& control_dep) const {
  return Scope(*this, Scope::Tags::ControlDeps(),
               std::vector<ops::Operation>(1, control_dep.op()),
               /* clear_control_deps */ false);
}

Scope Scope::WithNoControlDependencies() const {
  return Scope(*this, Scope::Tags::ControlDeps(), std::vector<ops::Operation>(),
               /* clear_control_deps */ true);
}

Scope Scope::WithDevice(const string& device) const {
  return Scope(*this, Scope::Tags::Device(), device);
}

Scope Scope::ColocateWith(const ops::Operation& op) const {
  return Scope(*this, Scope::Tags::Colocate(), op,
               /* clear_colocations */ false);
}

Scope Scope::ClearColocation() const {
  return Scope(*this, Scope::Tags::Colocate(), ops::Operation(),
               /* clear_colocations */ true);
}

Scope Scope::ExitOnError() const {
  return Scope(*this, Scope::Tags::ExitOnError());
}

Scope Scope::WithKernelLabel(const string& kernel_label) const {
  return Scope(*this, Scope::Tags::KernelLabel(), kernel_label);
}

CompositeOpScopes Scope::GetCompositeOpScopes(
    const string& composite_op_name) const {
  if (op_name_.empty() && composite_op_name.empty()) {
    UpdateStatus(errors::InvalidArgument(
        "Cannot create composite op scopes with empty name"));
    return {*this, *this};
  }
  if (!single_use_scope()) {
    Scope child = NewSubScope(op_name_.empty() ? composite_op_name : op_name_);
    const string child_op_sep = name_.empty() ? "" : "_";
    return {child, Scope(child, Scope::Tags::SingleUseScope(),
                         strings::StrCat(name_, child_op_sep, child.name_))};
  } else {
    return {
        Scope(*this, Scope::Tags::ScopeName(), op_name_, true /* copy_names */),
        *this};
  }
}

}  // namespace tensorflow

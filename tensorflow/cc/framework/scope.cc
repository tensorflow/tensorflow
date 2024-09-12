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
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

Scope::Scope(Impl* impl) : impl_(impl) {}

Scope::Scope(const Scope& other) : impl_(new Impl(*other.impl())) {}

Scope::~Scope() {}

Scope& Scope::operator=(const Scope& other) {
  // We can't copy Impls because of the const members, use copy ctor instead
  impl_.reset(new Impl(*other.impl_));
  return *this;
}

namespace {
const char kScopeSeparator[] = "/";
const char kSuffixSeparator[] = "_";
}  // namespace

Scope::Impl::Impl(Graph* graph, Status* status, NameMap* name_map,
                  ShapeRefiner* refiner, bool disable_shape_inference)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_(),
      disable_shape_inference_(disable_shape_inference) {}

Scope::Impl::Impl(const std::shared_ptr<Graph>& graph,
                  const std::shared_ptr<Status>& status,
                  const std::shared_ptr<NameMap>& name_map,
                  const std::shared_ptr<ShapeRefiner>& refiner)
    : graph_(graph),
      status_(status),
      name_map_(name_map),
      refiner_(refiner),
      scope_used_(nullptr),
      colocation_constraints_(),
      disable_shape_inference_(refiner_ == nullptr) {}

Scope Scope::NewRootScope() {
  Graph* graph = new Graph(OpRegistry::Global());
  ShapeRefiner* refiner =
      new ShapeRefiner(graph->versions(), graph->op_registry());
  return Scope(new Impl(graph, new Status, new Impl::NameMap, refiner,
                        /* disable_shape_inference */ false));
}

Scope Scope::DisabledShapeInferenceScope() {
  Graph* graph = new Graph(OpRegistry::Global());
  ShapeRefiner* refiner =
      new ShapeRefiner(graph->versions(), graph->op_registry());
  return Scope(new Impl(graph, new Status, new Impl::NameMap, refiner,
                        /* disable_shape_inference */ true));
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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(
          clear_colocations
              ? std::unordered_set<string>()
              : other.impl()->GetColocationConstraints(colocate_with_op)),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

Scope::Impl::Impl(const Scope& other, Tags::AssignedDevice,
                  const string& assigned_device)
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
      assigned_device_(assigned_device),
      xla_cluster_(other.impl()->xla_cluster_),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

Scope::Impl::Impl(const Scope& other, Tags::XlaCluster,
                  const string& xla_cluster)
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
      assigned_device_(other.impl()->assigned_device_),
      xla_cluster_(xla_cluster),
      colocation_constraints_(other.impl()->colocation_constraints_),
      disable_shape_inference_(other.impl()->disable_shape_inference_) {}

std::unordered_set<string> Scope::Impl::GetColocationConstraints(
    const Operation& colocate_with_op) const {
  std::unordered_set<string> current_constraints(colocation_constraints_);
  const AttrSlice attrs = colocate_with_op.node()->attrs();
  std::vector<string> node_constraints;
  if (TryGetNodeAttr(attrs, kColocationAttrName, &node_constraints)) {
    for (const string& entry : node_constraints) {
      StringPiece s(entry);
      if (absl::ConsumePrefix(&s, kColocationGroupPrefix)) {
        current_constraints.emplace(s);
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

void Scope::UpdateStatus(const Status& s) const {
  impl()->status_->Update(s);
  if (impl()->exit_on_error_ && !ok()) {
    LOG(FATAL) << *impl()->status_;
  }
}

Status Scope::ToGraphDef(GraphDef* gdef, bool include_debug_info) const {
  if (!ok()) {
    return *impl()->status_;
  }
  graph()->ToGraphDef(gdef, /*include_flib_def=*/true, include_debug_info);
  return absl::OkStatus();
}

Status Scope::ToGraph(Graph* g, GraphConstructorOptions opts) const {
  if (ok()) {
    GraphDef graph_def;
    graph()->ToGraphDef(&graph_def);
    UpdateStatus(ConvertGraphDefToGraph(opts, std::move(graph_def), g));
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
  if (!impl()->assigned_device_.empty()) {
    builder->AssignedDevice(impl()->assigned_device_);
  }
  if (!impl()->xla_cluster_.empty()) {
    builder->XlaCluster(impl()->xla_cluster_);
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
  if (entry == name_map_->end()) {
    name_map_->insert({prefix, 0});
    return prefix;
  }
  string unique_name;
  do {
    unique_name = strings::StrCat(prefix, kSuffixSeparator, ++entry->second);
  } while (name_map_->find(unique_name) != name_map_->end());
  name_map_->insert({unique_name, 0});
  return unique_name;
}

string Scope::Impl::GetNameForOp(const string& default_name) const {
  const string unique_name =
      GetUniqueName(default_name, true /* check_single_use */);
  const string sep =
      name_.empty() || unique_name.empty() ? "" : kScopeSeparator;
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
  const string sep =
      impl()->name_.empty() || unique_name.empty() ? "" : kScopeSeparator;
  return Scope(new Impl(*this, Impl::Tags::ScopeName(),
                        strings::StrCat(impl()->name_, sep, unique_name),
                        false /* copy_names */));
}

Scope Scope::WithOpNameImpl(const string& op_name) const {
  if (impl()->single_use_scope()) {
    UpdateStatus(errors::InvalidArgument("Cannot set op name ", op_name,
                                         " on this scope"));
    return *this;
  }
  return Scope(new Impl(*this, Impl::Tags::OpName(), impl()->name_, op_name));
}

Scope Scope::WithControlDependencies(
    const absl::Span<const Operation> control_deps) const {
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

Scope Scope::WithAssignedDevice(const string& assigned_device) const {
  return Scope(new Impl(*this, Impl::Tags::AssignedDevice(), assigned_device));
}

Scope Scope::WithXlaCluster(const string& xla_cluster) const {
  return Scope(new Impl(*this, Impl::Tags::XlaCluster(), xla_cluster));
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
    const string child_op_sep = impl()->name_.empty() ? "" : kSuffixSeparator;
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

Status Scope::DoShapeInference(Node* node) const {
  if (impl_->disable_shape_inference_) return absl::OkStatus();
  return impl_->refiner_->AddNode(node);
}

class InternalScope {
 public:
  // NewScope doesn't take ownership of the inputs.
  static Scope NewScope(Graph* graph, Status* status, ShapeRefiner* refiner) {
    Scope::Impl::NameMap* name_map = new Scope::Impl::NameMap;
    for (const Node* node : graph->nodes()) {
      const string& name = node->name();
      (*name_map)[name] = 0;
      // Add all name prefixes ('/' separated).
      size_t idx = -1;
      while ((idx = name.find(kScopeSeparator, idx + 1)) != string::npos) {
        (*name_map)[name.substr(0, idx)] = 0;
      }
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

Status CreateOutputWithScope(string op_name,
                             absl::Span<const ::tensorflow::Input> inputs,
                             const Scope& scope, Output* output) {
  TF_RETURN_IF_ERROR(scope.status());
  const auto unique_name = scope.GetUniqueNameForOp(op_name);
  auto builder = ::tensorflow::NodeBuilder(unique_name, op_name);
  for (const auto& input : inputs) {
    TF_RETURN_IF_ERROR(scope.status());
    builder = builder.Input(input.node());
  }
  ::tensorflow::Node* ret;
  scope.UpdateBuilder(&builder);
  TF_RETURN_IF_ERROR(scope.status());
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  TF_RETURN_IF_ERROR(scope.status());
  *output = Output(ret, 0);
  return absl::OkStatus();
}

}  // namespace tensorflow

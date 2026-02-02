/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"

#include <algorithm>
#include <memory>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// OpRegistry -----------------------------------------------------------------

OpRegistryInterface::~OpRegistryInterface() {}

Status OpRegistryInterface::LookUpOpDef(const string& op_type_name,
                                        const OpDef** op_def) const {
  *op_def = nullptr;
  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(LookUp(op_type_name, &op_reg_data));
  *op_def = &op_reg_data->op_def;
  return Status::OK();
}

OpRegistry::OpRegistry() : initialized_(false) {}

OpRegistry::~OpRegistry() {
  for (const auto& e : registry_) delete e.second;
}

void OpRegistry::Register(OpRegistrationDataFactory op_data_factory) {
  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
}

Status OpRegistry::LookUp(const string& op_type_name,
                          const OpRegistrationData** op_reg_data) const {
  *op_reg_data = nullptr;
  const OpRegistrationData* res = nullptr;

  bool first_call = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = MustCallDeferred();
    res = gtl::FindWithDefault(registry_, op_type_name, nullptr);
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(ValidateKernelRegistrations(*this));
  }
  if (res == nullptr) {
    static bool first_unregistered = true;
    if (first_unregistered) {
      OpList op_list;
      Export(true, &op_list);
      VLOG(1) << "All registered Ops:";
      for (const auto& op : op_list.op()) {
        VLOG(1) << SummarizeOpDef(op);
      }
      first_unregistered = false;
    }
    Status status =
        errors::NotFound("Op type not registered '", op_type_name, "'");
    VLOG(1) << status.ToString();
    return status;
  }
  *op_reg_data = res;
  return Status::OK();
}

void OpRegistry::GetRegisteredOps(std::vector<OpDef>* op_defs) {
  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_defs->push_back(p.second->op_def);
  }
}

Status OpRegistry::SetWatcher(const Watcher& watcher) {
  mutex_lock lock(mu_);
  if (watcher_ && watcher) {
    return errors::AlreadyExists(
        "Cannot over-write a valid watcher with another.");
  }
  watcher_ = watcher;
  return Status::OK();
}

void OpRegistry::Export(bool include_internal, OpList* ops) const {
  mutex_lock lock(mu_);
  MustCallDeferred();

  std::vector<std::pair<string, const OpRegistrationData*>> sorted(
      registry_.begin(), registry_.end());
  std::sort(sorted.begin(), sorted.end());

  auto out = ops->mutable_op();
  out->Clear();
  out->Reserve(sorted.size());

  for (const auto& item : sorted) {
    if (include_internal || !StringPiece(item.first).starts_with("_")) {
      *out->Add() = item.second->op_def;
    }
  }
}

void OpRegistry::DeferRegistrations() {
  mutex_lock lock(mu_);
  initialized_ = false;
}

void OpRegistry::ClearDeferredRegistrations() {
  mutex_lock lock(mu_);
  deferred_.clear();
}

Status OpRegistry::ProcessRegistrations() const {
  mutex_lock lock(mu_);
  return CallDeferred();
}

string OpRegistry::DebugString(bool include_internal) const {
  OpList op_list;
  Export(include_internal, &op_list);
  string ret;
  for (const auto& op : op_list.op()) {
    strings::StrAppend(&ret, SummarizeOpDef(op), "\n");
  }
  return ret;
}

bool OpRegistry::MustCallDeferred() const {
  if (initialized_) return false;
  initialized_ = true;
  for (int i = 0; i < deferred_.size(); ++i) {
    TF_QCHECK_OK(RegisterAlreadyLocked(deferred_[i]));
  }
  deferred_.clear();
  return true;
}

Status OpRegistry::CallDeferred() const {
  if (initialized_) return Status::OK();
  initialized_ = true;
  for (int i = 0; i < deferred_.size(); ++i) {
    Status s = RegisterAlreadyLocked(deferred_[i]);
    if (!s.ok()) {
      return s;
    }
  }
  deferred_.clear();
  return Status::OK();
}

Status OpRegistry::RegisterAlreadyLocked(
    OpRegistrationDataFactory op_data_factory) const {
  std::unique_ptr<OpRegistrationData> op_reg_data(new OpRegistrationData);
  Status s = op_data_factory(op_reg_data.get());
  if (s.ok()) {
    s = ValidateOpDef(op_reg_data->op_def);
    if (s.ok() &&
        !gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
                                 op_reg_data.get())) {
      s = errors::AlreadyExists("Op with name ", op_reg_data->op_def.name());
    }
  }
  Status watcher_status = s;
  if (watcher_) {
    watcher_status = watcher_(s, op_reg_data->op_def);
  }
  if (s.ok()) {
    op_reg_data.release();
  } else {
    op_reg_data.reset();
  }
  return watcher_status;
}

// static
OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}

// OpListOpRegistry -----------------------------------------------------------

OpListOpRegistry::OpListOpRegistry(const OpList* op_list) {
  for (const OpDef& op_def : op_list->op()) {
    auto* op_reg_data = new OpRegistrationData();
    op_reg_data->op_def = op_def;
    index_[op_def.name()] = op_reg_data;
  }
}

OpListOpRegistry::~OpListOpRegistry() {
  for (const auto& e : index_) delete e.second;
}

Status OpListOpRegistry::LookUp(const string& op_type_name,
                                const OpRegistrationData** op_reg_data) const {
  auto iter = index_.find(op_type_name);
  if (iter == index_.end()) {
    *op_reg_data = nullptr;
    return errors::NotFound("Op type not registered '", op_type_name, "'");
  }
  *op_reg_data = iter->second;
  return Status::OK();
}

// Other registration ---------------------------------------------------------

namespace register_op {
OpDefBuilderReceiver::OpDefBuilderReceiver(
    const OpDefBuilderWrapper<true>& wrapper) {
  OpRegistry::Global()->Register(
      [wrapper](OpRegistrationData* op_reg_data) -> Status {
        return wrapper.builder().Finalize(op_reg_data);
      });
}
}  // namespace register_op

}  // namespace tensorflow

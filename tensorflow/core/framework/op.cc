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
#include <utility>
#include <vector>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

absl::Status DefaultValidator(const OpRegistryInterface& op_registry) {
  LOG(WARNING) << "No kernel validator registered with OpRegistry.";
  return absl::OkStatus();
}

// OpRegistry -----------------------------------------------------------------

absl::Status OpRegistryInterface::LookUpOpDef(const string& op_type_name,
                                              const OpDef** op_def) const {
  *op_def = nullptr;
  const OpRegistrationData* op_reg_data = nullptr;
  TF_RETURN_IF_ERROR(LookUp(op_type_name, &op_reg_data));
  *op_def = &op_reg_data->op_def;
  return absl::OkStatus();
}

OpRegistry::OpRegistry()
    : initialized_(false), op_registry_validator_(DefaultValidator) {}

void OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory) {
  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
}

namespace {
// Helper function that returns Status message for failed LookUp.
absl::Status OpNotFound(const string& op_type_name) {
  absl::Status status = errors::NotFound(
      "Op type not registered '", op_type_name, "' in binary running on ",
      port::Hostname(), ". ",
      "Make sure the Op and Kernel are registered in the binary running in "
      "this process. Note that if you are loading a saved graph which used ops "
      "from tf.contrib (e.g. `tf.contrib.resampler`), accessing should be done "
      "before importing the graph, as contrib ops are lazily registered when "
      "the module is first accessed.");
  VLOG(1) << status.ToString();
  return status;
}
}  // namespace

absl::Status OpRegistry::LookUp(const string& op_type_name,
                                const OpRegistrationData** op_reg_data) const {
  if ((*op_reg_data = LookUp(op_type_name))) return absl::OkStatus();
  return OpNotFound(op_type_name);
}

const OpRegistrationData* OpRegistry::LookUp(const string& op_type_name) const {
  {
    tf_shared_lock l(mu_);
    if (initialized_) {
      if (const OpRegistrationData* res =
              gtl::FindWithDefault(registry_, op_type_name, nullptr).get()) {
        return res;
      }
    }
  }
  return LookUpSlow(op_type_name);
}

const OpRegistrationData* OpRegistry::LookUpSlow(
    const string& op_type_name) const {
  const OpRegistrationData* res = nullptr;

  bool first_call = false;
  bool first_unregistered = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = MustCallDeferred();
    res = gtl::FindWithDefault(registry_, op_type_name, nullptr).get();

    static bool unregistered_before = false;
    first_unregistered = !unregistered_before && (res == nullptr);
    if (first_unregistered) {
      unregistered_before = true;
    }
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(op_registry_validator_(*this));
  }
  if (res == nullptr) {
    if (first_unregistered) {
      OpList op_list;
      Export(true, &op_list);
      if (VLOG_IS_ON(3)) {
        LOG(INFO) << "All registered Ops:";
        for (const auto& op : op_list.op()) {
          LOG(INFO) << SummarizeOpDef(op);
        }
      }
    }
  }
  return res;
}

void OpRegistry::GetRegisteredOps(std::vector<OpDef>* op_defs) {
  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_defs->push_back(p.second->op_def);
  }
}

void OpRegistry::GetOpRegistrationData(
    std::vector<OpRegistrationData>* op_data) {
  mutex_lock lock(mu_);
  MustCallDeferred();
  for (const auto& p : registry_) {
    op_data->push_back(*p.second);
  }
}

absl::Status OpRegistry::SetWatcher(const Watcher& watcher) {
  mutex_lock lock(mu_);
  if (watcher_ && watcher) {
    return errors::AlreadyExists(
        "Cannot over-write a valid watcher with another.");
  }
  watcher_ = watcher;
  return absl::OkStatus();
}

void OpRegistry::Export(bool include_internal, OpList* ops) const {
  mutex_lock lock(mu_);
  MustCallDeferred();

  std::vector<std::pair<absl::string_view, const OpRegistrationData*>> sorted;
  sorted.reserve(registry_.size());
  for (const auto& item : registry_) {
    sorted.emplace_back(item.first, item.second.get());
  }
  std::sort(sorted.begin(), sorted.end());

  auto out = ops->mutable_op();
  out->Clear();
  out->Reserve(sorted.size());

  for (const auto& item : sorted) {
    if (include_internal || !absl::StartsWith(item.first, "_")) {
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

absl::Status OpRegistry::ProcessRegistrations() const {
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
  registry_.reserve(registry_.size() + deferred_.size());
  for (const auto& op_data_factory : deferred_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  }
  deferred_.clear();
  return true;
}

absl::Status OpRegistry::CallDeferred() const {
  if (initialized_) return absl::OkStatus();
  initialized_ = true;
  registry_.reserve(registry_.size() + deferred_.size());
  for (const auto& op_data_factory : deferred_) {
    absl::Status s = RegisterAlreadyLocked(op_data_factory);
    if (!s.ok()) {
      return s;
    }
  }
  deferred_.clear();
  return absl::OkStatus();
}

absl::Status OpRegistry::RegisterAlreadyLocked(
    const OpRegistrationDataFactory& op_data_factory) const {
  auto op_reg_data = std::make_unique<OpRegistrationData>();
  const auto* op_reg_data_raw = op_reg_data.get();
  absl::Status s = op_data_factory(op_reg_data.get());
  if (s.ok()) {
    s = ValidateOpDef(op_reg_data->op_def);
  }
  if (s.ok() &&
      !registry_.try_emplace(op_reg_data->op_def.name(), std::move(op_reg_data))
           .second) {
    s = errors::AlreadyExists("Op with name ", op_reg_data->op_def.name());
  }
  absl::Status watcher_status = s;
  if (watcher_) {
    watcher_status = watcher_(s, op_reg_data_raw->op_def);
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
  index_.reserve(op_list->op_size());
  for (const OpDef& op_def : op_list->op()) {
    auto op_reg_data = std::make_unique<OpRegistrationData>();
    op_reg_data->op_def = op_def;
    index_[op_def.name()] = std::move(op_reg_data);
  }
}

const OpRegistrationData* OpListOpRegistry::LookUp(
    const string& op_type_name) const {
  auto iter = index_.find(op_type_name);
  if (iter == index_.end()) {
    return nullptr;
  }
  return iter->second.get();
}

absl::Status OpListOpRegistry::LookUp(
    const string& op_type_name, const OpRegistrationData** op_reg_data) const {
  if ((*op_reg_data = LookUp(op_type_name))) return absl::OkStatus();
  return OpNotFound(op_type_name);
}

namespace register_op {

InitOnStartupMarker OpDefBuilderWrapper::operator()() {
  OpRegistry::Global()->Register(
      [builder = std::move(builder_)](OpRegistrationData* op_reg_data)
          -> absl::Status { return builder.Finalize(op_reg_data); });
  return {};
}

}  //  namespace register_op

}  // namespace tensorflow

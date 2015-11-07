#include "tensorflow/core/framework/op.h"

#include <algorithm>
#include <memory>
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// OpRegistry -----------------------------------------------------------------

OpRegistryInterface::~OpRegistryInterface() {}

OpRegistry::OpRegistry() : initialized_(false) {}

void OpRegistry::Register(std::function<OpDef(void)> func) {
  mutex_lock lock(mu_);
  if (initialized_) {
    OpDef def = func();
    TF_QCHECK_OK(RegisterAlreadyLocked(def)) << "Attempting to register: "
                                             << SummarizeOpDef(def);
  } else {
    deferred_.push_back(func);
  }
}

const OpDef* OpRegistry::LookUp(const string& op_type_name,
                                Status* status) const {
  const OpDef* op_def = nullptr;
  bool first_call = false;
  {  // Scope for lock.
    mutex_lock lock(mu_);
    first_call = CallDeferred();
    op_def = gtl::FindWithDefault(registry_, op_type_name, nullptr);
    // Note: Can't hold mu_ while calling Export() below.
  }
  if (first_call) {
    TF_QCHECK_OK(ValidateKernelRegistrations(this));
  }
  if (op_def == nullptr) {
    status->Update(
        errors::NotFound("Op type not registered '", op_type_name, "'"));
    static bool first = true;
    if (first) {
      OpList op_list;
      Export(true, &op_list);
      LOG(INFO) << "All registered Ops:";
      for (const auto& op : op_list.op()) {
        LOG(INFO) << SummarizeOpDef(op);
      }
      first = false;
    }
  }
  return op_def;
}

void OpRegistry::Export(bool include_internal, OpList* ops) const {
  mutex_lock lock(mu_);
  CallDeferred();

  std::vector<std::pair<string, const OpDef*>> sorted(registry_.begin(),
                                                      registry_.end());
  std::sort(sorted.begin(), sorted.end());

  auto out = ops->mutable_op();
  out->Clear();
  out->Reserve(sorted.size());

  for (const auto& item : sorted) {
    if (include_internal || !StringPiece(item.first).starts_with("_")) {
      *out->Add() = *item.second;
    }
  }
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

bool OpRegistry::CallDeferred() const {
  if (initialized_) return false;
  initialized_ = true;
  for (const auto& fn : deferred_) {
    OpDef def = fn();
    TF_QCHECK_OK(RegisterAlreadyLocked(def)) << "Attempting to register: "
                                             << SummarizeOpDef(def);
  }
  deferred_.clear();
  return true;
}

Status OpRegistry::RegisterAlreadyLocked(const OpDef& def) const {
  TF_RETURN_IF_ERROR(ValidateOpDef(def));

  std::unique_ptr<OpDef> copy(new OpDef(def));
  if (gtl::InsertIfNotPresent(&registry_, def.name(), copy.get())) {
    copy.release();  // Ownership transferred to op_registry
    return Status::OK();
  } else {
    return errors::AlreadyExists("Op with name ", def.name());
  }
}

// static
OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}

namespace register_op {
OpDefBuilder& RegisterOp(StringPiece name) {
  VLOG(1) << "RegisterOp: " << name;
  OpDefBuilder* b = new OpDefBuilder(name);
  OpRegistry::Global()->Register([b]() -> ::tensorflow::OpDef {
    OpDef op_def;
    TF_QCHECK_OK(b->Finalize(&op_def));
    delete b;
    return op_def;
  });
  return *b;
}
}  // namespace register_op

}  // namespace tensorflow

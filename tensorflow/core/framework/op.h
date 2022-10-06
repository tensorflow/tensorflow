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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_H_

#include <functional>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_inference_util.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Users that want to look up an OpDef by type name should take an
// OpRegistryInterface.  Functions accepting a
// (const) OpRegistryInterface* may call LookUp() from multiple threads.
class OpRegistryInterface {
 public:
  virtual ~OpRegistryInterface();

  // Returns an error status and sets *op_reg_data to nullptr if no OpDef is
  // registered under that name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  virtual Status LookUp(const std::string& op_type_name,
                        const OpRegistrationData** op_reg_data) const = 0;

  // Shorthand for calling LookUp to get the OpDef.
  Status LookUpOpDef(const std::string& op_type_name,
                     const OpDef** op_def) const;
};

// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering ops via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register(
//     [](OpRegistrationData* op_reg_data)->Status {
//       // Populate *op_reg_data here.
//       return OkStatus();
//   });
class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;

  OpRegistry();
  ~OpRegistry() override;

  void Register(const OpRegistrationDataFactory& op_data_factory);

  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Returns OpRegistrationData* of registered op type, else returns nullptr.
  const OpRegistrationData* LookUp(const std::string& op_type_name) const;

  // Fills *ops with all registered OpDefs (except those with names
  // starting with '_' if include_internal == false) sorted in
  // ascending alphabetical order.
  void Export(bool include_internal, OpList* ops) const;

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  std::string DebugString(bool include_internal) const;

  // A singleton available at startup.
  static OpRegistry* Global();

  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);

  // Get all `OpRegistrationData`s.
  void GetOpRegistrationData(std::vector<OpRegistrationData>* op_data);

  // Registers a function that validates op registry.
  void RegisterValidator(
      std::function<Status(const OpRegistryInterface&)> validator) {
    op_registry_validator_ = std::move(validator);
  }

  // Watcher, a function object.
  // The watcher, if set by SetWatcher(), is called every time an op is
  // registered via the Register function. The watcher is passed the Status
  // obtained from building and adding the OpDef to the registry, and the OpDef
  // itself if it was successfully built. A watcher returns a Status which is in
  // turn returned as the final registration status.
  typedef std::function<Status(const Status&, const OpDef&)> Watcher;

  // An OpRegistry object has only one watcher. This interface is not thread
  // safe, as different clients are free to set the watcher any time.
  // Clients are expected to atomically perform the following sequence of
  // operations :
  // SetWatcher(a_watcher);
  // Register some ops;
  // op_registry->ProcessRegistrations();
  // SetWatcher(nullptr);
  // Returns a non-OK status if a non-null watcher is over-written by another
  // non-null watcher.
  Status SetWatcher(const Watcher& watcher);

  // Process the current list of deferred registrations. Note that calls to
  // Export, LookUp and DebugString would also implicitly process the deferred
  // registrations. Returns the status of the first failed op registration or
  // OkStatus() otherwise.
  Status ProcessRegistrations() const;

  // Defer the registrations until a later call to a function that processes
  // deferred registrations are made. Normally, registrations that happen after
  // calls to Export, LookUp, ProcessRegistrations and DebugString are processed
  // immediately. Call this to defer future registrations.
  void DeferRegistrations();

  // Clear the registrations that have been deferred.
  void ClearDeferredRegistrations();

 private:
  // Ensures that all the functions in deferred_ get called, their OpDef's
  // registered, and returns with deferred_ empty.  Returns true the first
  // time it is called. Prints a fatal log if any op registration fails.
  bool MustCallDeferred() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Calls the functions in deferred_ and registers their OpDef's
  // It returns the Status of the first failed op registration or OkStatus()
  // otherwise.
  Status CallDeferred() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add 'def' to the registry with additional data 'data'. On failure, or if
  // there is already an OpDef with that name registered, returns a non-okay
  // status.
  Status RegisterAlreadyLocked(const OpRegistrationDataFactory& op_data_factory)
      const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const OpRegistrationData* LookUpSlow(const std::string& op_type_name) const;

  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ TF_GUARDED_BY(mu_);
  // Values are owned.
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      TF_GUARDED_BY(mu_);
  mutable bool initialized_ TF_GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ TF_GUARDED_BY(mu_);

  std::function<Status(const OpRegistryInterface&)> op_registry_validator_;
};

// An adapter to allow an OpList to be used as an OpRegistryInterface.
//
// Note that shape inference functions are not passed in to OpListOpRegistry, so
// it will return an unusable shape inference function for every op it supports;
// therefore, it should only be used in contexts where this is okay.
class OpListOpRegistry : public OpRegistryInterface {
 public:
  // Does not take ownership of op_list, *op_list must outlive *this.
  explicit OpListOpRegistry(const OpList* op_list);
  ~OpListOpRegistry() override;
  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Returns OpRegistrationData* of op type in list, else returns nullptr.
  const OpRegistrationData* LookUp(const std::string& op_type_name) const;

 private:
  // Values are owned.
  std::unordered_map<string, const OpRegistrationData*> index_;
};

// Support for defining the OpDef (specifying the semantics of the Op and how
// it should be created) and registering it in the OpRegistry::Global()
// registry.  Usage:
//
// REGISTER_OP("my_op_name")
//     .Attr("<name>:<type>")
//     .Attr("<name>:<type>=<default>")
//     .Input("<name>:<type-expr>")
//     .Input("<name>:Ref(<type-expr>)")
//     .Output("<name>:<type-expr>")
//     .Doc(R"(
// <1-line summary>
// <rest of the description (potentially many lines)>
// <name-of-attr-input-or-output>: <description of name>
// <name-of-attr-input-or-output>: <description of name;
//   if long, indent the description on subsequent lines>
// )");
//
// Note: .Doc() should be last.
// For details, see the OpDefBuilder class in op_def_builder.h.

namespace register_op {

class OpDefBuilderWrapper {
 public:
  explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}
  OpDefBuilderWrapper& Attr(std::string spec) {
    builder_.Attr(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Attr(const char* spec) TF_ATTRIBUTE_NOINLINE {
    return Attr(std::string(spec));
  }
  OpDefBuilderWrapper& Input(std::string spec) {
    builder_.Input(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Input(const char* spec) TF_ATTRIBUTE_NOINLINE {
    return Input(std::string(spec));
  }
  OpDefBuilderWrapper& Output(std::string spec) {
    builder_.Output(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Output(const char* spec) TF_ATTRIBUTE_NOINLINE {
    return Output(std::string(spec));
  }
  OpDefBuilderWrapper& SetIsCommutative() {
    builder_.SetIsCommutative();
    return *this;
  }
  OpDefBuilderWrapper& SetIsAggregate() {
    builder_.SetIsAggregate();
    return *this;
  }
  OpDefBuilderWrapper& SetIsStateful() {
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper& SetDoNotOptimize() {
    // We don't have a separate flag to disable optimizations such as constant
    // folding and CSE so we reuse the stateful flag.
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper& SetAllowsUninitializedInput() {
    builder_.SetAllowsUninitializedInput();
    return *this;
  }
  OpDefBuilderWrapper& Deprecated(int version, std::string explanation) {
    builder_.Deprecated(version, std::move(explanation));
    return *this;
  }
  OpDefBuilderWrapper& Doc(std::string text) {
    builder_.Doc(std::move(text));
    return *this;
  }
  OpDefBuilderWrapper& SetShapeFn(OpShapeInferenceFn fn) {
    builder_.SetShapeFn(std::move(fn));
    return *this;
  }
  OpDefBuilderWrapper& SetIsDistributedCommunication() {
    builder_.SetIsDistributedCommunication();
    return *this;
  }

  OpDefBuilderWrapper& SetTypeConstructor(OpTypeConstructor fn) {
    builder_.SetTypeConstructor(std::move(fn));
    return *this;
  }

  OpDefBuilderWrapper& SetForwardTypeFn(TypeInferenceFn fn) {
    builder_.SetForwardTypeFn(std::move(fn));
    return *this;
  }

  OpDefBuilderWrapper& SetReverseTypeFn(int input_number, TypeInferenceFn fn) {
    builder_.SetReverseTypeFn(input_number, std::move(fn));
    return *this;
  }

  const ::tensorflow::OpDefBuilder& builder() const { return builder_; }

  InitOnStartupMarker operator()();

 private:
  mutable ::tensorflow::OpDefBuilder builder_;
};

}  // namespace register_op

#define REGISTER_OP_IMPL(ctr, name, is_system_op)                         \
  static ::tensorflow::InitOnStartupMarker const register_op##ctr         \
      TF_ATTRIBUTE_UNUSED =                                               \
          TF_INIT_ON_STARTUP_IF(is_system_op || SHOULD_REGISTER_OP(name)) \
          << ::tensorflow::register_op::OpDefBuilderWrapper(name)

#define REGISTER_OP(name)        \
  TF_ATTRIBUTE_ANNOTATE("tf:op") \
  TF_NEW_ID_FOR_INIT(REGISTER_OP_IMPL, name, false)

// The `REGISTER_SYSTEM_OP()` macro acts as `REGISTER_OP()` except
// that the op is registered unconditionally even when selective
// registration is used.
#define REGISTER_SYSTEM_OP(name)        \
  TF_ATTRIBUTE_ANNOTATE("tf:op")        \
  TF_ATTRIBUTE_ANNOTATE("tf:op:system") \
  TF_NEW_ID_FOR_INIT(REGISTER_OP_IMPL, name, true)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_H_

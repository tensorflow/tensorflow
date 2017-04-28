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

#ifndef TENSORFLOW_FRAMEWORK_OP_H_
#define TENSORFLOW_FRAMEWORK_OP_H_

#include <functional>
#include <unordered_map>

#include <vector>
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/framework/selective_registration.h"
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
  virtual Status LookUp(const string& op_type_name,
                        const OpRegistrationData** op_reg_data) const = 0;

  // Shorthand for calling LookUp to get the OpDef.
  Status LookUpOpDef(const string& op_type_name, const OpDef** op_def) const;
};

// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering ops via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register(
//     [](OpRegistrationData* op_reg_data)->Status {
//       // Populate *op_reg_data here.
//       return Status::OK();
//   });
class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;

  OpRegistry();
  ~OpRegistry() override;

  void Register(OpRegistrationDataFactory op_data_factory);

  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  // Fills *ops with all registered OpDefs (except those with names
  // starting with '_' if include_internal == false).
  void Export(bool include_internal, OpList* ops) const;

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  string DebugString(bool include_internal) const;

  // A singleton available at startup.
  static OpRegistry* Global();

  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);

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
  // Status::OK() otherwise.
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
  bool MustCallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Calls the functions in deferred_ and registers their OpDef's
  // It returns the Status of the first failed op registration or Status::OK()
  // otherwise.
  Status CallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add 'def' to the registry with additional data 'data'. On failure, or if
  // there is already an OpDef with that name registered, returns a non-okay
  // status.
  Status RegisterAlreadyLocked(OpRegistrationDataFactory op_data_factory) const
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ GUARDED_BY(mu_);
  // Values are owned.
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      GUARDED_BY(mu_);
  mutable bool initialized_ GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ GUARDED_BY(mu_);
};

// An adapter to allow an OpList to be used as an OpRegistryInterface.
//
// Note that shape inference functions are not passed in to OpListOpRegistry, so
// it will return an unusable shape inference function for every op it supports;
// therefore, it should only be used in contexts where this is okay.
class OpListOpRegistry : public OpRegistryInterface {
 public:
  // Does not take ownership of op_list, *op_list must outlive *this.
  OpListOpRegistry(const OpList* op_list);
  ~OpListOpRegistry() override;
  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

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

// OpDefBuilderWrapper is a templated class that is used in the REGISTER_OP
// calls. This allows the result of REGISTER_OP to be used in chaining, as in
// REGISTER_OP(a).Attr("...").Input("...");, while still allowing selective
// registration to turn the entire call-chain into a no-op.
template <bool should_register>
class OpDefBuilderWrapper;

// Template specialization that forwards all calls to the contained builder.
template <>
class OpDefBuilderWrapper<true> {
 public:
  typedef OpDefBuilderWrapper<true> WrapperType;
  OpDefBuilderWrapper(const char name[]) : builder_(name) {}
  OpDefBuilderWrapper<true>& Attr(StringPiece spec) {
    builder_.Attr(spec);
    return *this;
  }
  OpDefBuilderWrapper<true>& Input(StringPiece spec) {
    builder_.Input(spec);
    return *this;
  }
  OpDefBuilderWrapper<true>& Output(StringPiece spec) {
    builder_.Output(spec);
    return *this;
  }
  OpDefBuilderWrapper<true>& SetIsCommutative() {
    builder_.SetIsCommutative();
    return *this;
  }
  OpDefBuilderWrapper<true>& SetIsAggregate() {
    builder_.SetIsAggregate();
    return *this;
  }
  OpDefBuilderWrapper<true>& SetIsStateful() {
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper<true>& SetAllowsUninitializedInput() {
    builder_.SetAllowsUninitializedInput();
    return *this;
  }
  OpDefBuilderWrapper<true>& Deprecated(int version, StringPiece explanation) {
    builder_.Deprecated(version, explanation);
    return *this;
  }
  OpDefBuilderWrapper<true>& Doc(StringPiece text) {
    builder_.Doc(text);
    return *this;
  }
  OpDefBuilderWrapper<true>& SetShapeFn(
      Status (*fn)(shape_inference::InferenceContext*)) {
    builder_.SetShapeFn(fn);
    return *this;
  }
  const ::tensorflow::OpDefBuilder& builder() const { return builder_; }

 private:
  mutable ::tensorflow::OpDefBuilder builder_;
};

// Template specialization that turns all calls into no-ops.
template <>
class OpDefBuilderWrapper<false> {
 public:
  constexpr OpDefBuilderWrapper(const char name[]) {}
  OpDefBuilderWrapper<false>& Attr(StringPiece spec) { return *this; }
  OpDefBuilderWrapper<false>& Input(StringPiece spec) { return *this; }
  OpDefBuilderWrapper<false>& Output(StringPiece spec) { return *this; }
  OpDefBuilderWrapper<false>& SetIsCommutative() { return *this; }
  OpDefBuilderWrapper<false>& SetIsAggregate() { return *this; }
  OpDefBuilderWrapper<false>& SetIsStateful() { return *this; }
  OpDefBuilderWrapper<false>& SetAllowsUninitializedInput() { return *this; }
  OpDefBuilderWrapper<false>& Deprecated(int, StringPiece) { return *this; }
  OpDefBuilderWrapper<false>& Doc(StringPiece text) { return *this; }
  OpDefBuilderWrapper<false>& SetShapeFn(
      Status (*fn)(shape_inference::InferenceContext*)) {
    return *this;
  }
};

struct OpDefBuilderReceiver {
  // To call OpRegistry::Global()->Register(...), used by the
  // REGISTER_OP macro below.
  // Note: These are implicitly converting constructors.
  OpDefBuilderReceiver(
      const OpDefBuilderWrapper<true>& wrapper);  // NOLINT(runtime/explicit)
  constexpr OpDefBuilderReceiver(const OpDefBuilderWrapper<false>&) {
  }  // NOLINT(runtime/explicit)
};
}  // namespace register_op

#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                          \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr    \
      TF_ATTRIBUTE_UNUSED =                                                  \
          ::tensorflow::register_op::OpDefBuilderWrapper<SHOULD_REGISTER_OP( \
              name)>(name)

// The `REGISTER_SYSTEM_OP()` macro acts as `REGISTER_OP()` except
// that the op is registered unconditionally even when selective
// registration is used.
#define REGISTER_SYSTEM_OP(name) \
  REGISTER_SYSTEM_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_SYSTEM_OP_UNIQ_HELPER(ctr, name) \
  REGISTER_SYSTEM_OP_UNIQ(ctr, name)
#define REGISTER_SYSTEM_OP_UNIQ(ctr, name)                                \
  static ::tensorflow::register_op::OpDefBuilderReceiver register_op##ctr \
      TF_ATTRIBUTE_UNUSED =                                               \
          ::tensorflow::register_op::OpDefBuilderWrapper<true>(name)

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_H_

/* Copyright 2015 Google Inc. All Rights Reserved.

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

  // Returns nullptr and sets *status if no OpDef is registered under that
  // name, otherwise returns the registered OpDef.
  // Caller must not delete the returned pointer.
  virtual const OpDef* LookUp(const string& op_type_name,
                              Status* status) const = 0;
};

// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering OpDefs via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register([]()->OpDef{
//     OpDef def;
//     // Populate def here.
//     return def;
//   });
class OpRegistry : public OpRegistryInterface {
 public:
  OpRegistry();
  ~OpRegistry() override {}

  // Calls func() and registers the returned OpDef.  Since Register()
  // is normally called during program initialization (before main()),
  // we defer calling func() until the first call to LookUp() or
  // Export() (if one of those has already been called, func() is
  // called immediately).
  void Register(const OpDef& op_def);

  const OpDef* LookUp(const string& op_type_name,
                      Status* status) const override;

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
  // watcher_, if not null, is called every time an op is registered via the
  // Register function. watcher_ is passed the OpDef of the op getting
  // registered.
  typedef std::function<void(const OpDef&)> Watcher;

  // An OpRegistry object has only one watcher. This interface is not thread
  // safe, as different clients are free to set the watcher any time.
  // Clients are expected to atomically perform the following sequence of
  // operations :
  // SetWatcher(a_watcher);
  // Register some ops;
  // SetWatcher(nullptr);
  // Returns a non-OK status if a non-null watcher is over-written by another
  // non-null watcher.
  Status SetWatcher(const Watcher& watcher);

 private:
  // Ensures that all the functions in deferred_ get called, their OpDef's
  // registered, and returns with deferred_ empty.  Returns true the first
  // time it is called.
  bool CallDeferred() const EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Add 'def' to the registry.  On failure, or if there is already an
  // OpDef with that name registered, returns a non-okay status.
  Status RegisterAlreadyLocked(const OpDef& def) const
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpDef> deferred_ GUARDED_BY(mu_);
  mutable std::unordered_map<string, OpDef*> registry_ GUARDED_BY(mu_);
  mutable bool initialized_ GUARDED_BY(mu_);

  // Registry watcher.
  mutable Watcher watcher_ GUARDED_BY(mu_);
};

// An adapter to allow an OpList to be used as an OpRegistryInterface.
class OpListOpRegistry : public OpRegistryInterface {
 public:
  // Does not take ownership of op_list, *op_list must outlive *this.
  OpListOpRegistry(const OpList* op_list);
  ~OpListOpRegistry() override {}
  const OpDef* LookUp(const string& op_type_name,
                      Status* status) const override;

 private:
  std::unordered_map<string, const OpDef*> index_;
};

// Treats 'registry_ptr' as a pointer to OpRegistry, and calls
// registry_ptr->Register(op_def) for each op_def that has been registered with
// the current library's global op registry (obtained by calling
// OpRegistry::Global().
extern "C" void RegisterOps(void* registry_ptr);

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

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_H_

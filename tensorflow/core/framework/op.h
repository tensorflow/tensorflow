#ifndef TENSORFLOW_FRAMEWORK_OP_H_
#define TENSORFLOW_FRAMEWORK_OP_H_

#include <functional>
#include <unordered_map>

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/status.h"

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
  void Register(std::function<OpDef(void)> func);

  const OpDef* LookUp(const string& op_type_name,
                      Status* status) const override;

  // Fills *ops with all registered OpDefss (except those with names
  // starting with '_' if include_internal == false).
  void Export(bool include_internal, OpList* ops) const;

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  string DebugString(bool include_internal) const;

  // A singleton available at startup.
  static OpRegistry* Global();

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
  mutable std::vector<std::function<OpDef(void)>> deferred_ GUARDED_BY(mu_);
  mutable std::unordered_map<string, OpDef*> registry_ GUARDED_BY(mu_);
  mutable bool initialized_ GUARDED_BY(mu_);
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
// To call OpRegistry::Global()->Register(...), used by the
// REGISTER_OP macro below.
OpDefBuilder& RegisterOp(StringPiece name);
}  // namespace register_op

#define REGISTER_OP(name) REGISTER_OP_UNIQ_HELPER(__COUNTER__, name)
#define REGISTER_OP_UNIQ_HELPER(ctr, name) REGISTER_OP_UNIQ(ctr, name)
#define REGISTER_OP_UNIQ(ctr, name)                                         \
  static ::tensorflow::OpDefBuilder& register_op##ctr TF_ATTRIBUTE_UNUSED = \
      ::tensorflow::register_op::RegisterOp(name)

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_OP_H_

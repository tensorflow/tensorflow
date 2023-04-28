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

#ifndef TENSORFLOW_CORE_FRAMEWORK_FUNCTION_H_
#define TENSORFLOW_CORE_FRAMEWORK_FUNCTION_H_

#include <memory>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/optimized_function_graph.pb.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"
#endif  // IS_MOBILE_PLATFORM

namespace tensorflow {

class CollectiveExecutor;
class DeviceSet;
class Graph;
class GraphDef;
class OpKernel;
class ProcessFunctionLibraryRuntime;
class ResourceMgr;
class Rendezvous;
class ScopedStepContainer;
class StepStatsCollectorInterface;
class Node;

// FunctionDefHelper::Create is a convenient helper to construct a
// FunctionDef proto.
// E.g.,
//   FunctionDef my_func = FunctionDefHelper::Create(
//     "my_func_name",
//     {"x:T", "y:T" /* one string per argument */},
//     {"z:T" /* one string per return value */},
//     {"T: {float, double}" /* one string per attribute  */},
//     {
//        {{"o"}, "Mul", {"x", "y"}, {{"T", "$T"}}}
//        /* one entry per function node */
//     },
//     /* Mapping between function returns and function node outputs. */
//     {{"z", "o:z"}});
//
// For the old Function::Node approach, use FunctionDefHelper::Define()
// E.g.,
//   FunctionDef my_func = FunctionDefHelper::Define(
//     "my_func_name",
//     {"x:T", "y:T" /* one string per argument */},
//     {"z:T" /* one string per return value */},
//     {"T: {float, double}" /* one string per attribute  */},
//     {
//        {{"z"}, "Mul", {"x", "y"}, {{"T", "$T"}}}
//        /* one entry per function node */
//     });
class FunctionDefHelper {
 public:
  // AttrValueWrapper has copy constructors for the type T so that
  // it's easy to construct a simple AttrValue proto.
  //
  // If T is a string type (const char*, string, or StringPiece), and
  // it starts with "$", we construct a AttrValue of "placeholder".
  //
  // E.g.,
  //   std::<string, AttrValueWrapper> x = {"T", "$T"}
  // is a named attr value placeholder.
  struct AttrValueWrapper {
    AttrValue proto;

    AttrValueWrapper() {}

    template <typename T>
    AttrValueWrapper(T val) {  // NOLINT(runtime/explicit)
      SetAttrValue(val, &proto);
    }

   private:
    void InitFromString(StringPiece val);
  };

  // Constructs an AttrValue.func given the "name" and "attrs".
  static AttrValueWrapper FunctionRef(
      const std::string& name,
      gtl::ArraySlice<std::pair<string, AttrValueWrapper>> attrs);
  static AttrValueWrapper FunctionRef(const std::string& name) {
    return FunctionRef(name, {});
  }

  // Node is used to construct FunctionDef.Node using initialization
  // lists. E.g.,
  //  Node n = {{"z"}, "Mul", {"x", "y"}, {{"T", "$T"}}};  // z = x * y
  //
  // If the op has no inputs, then name is be specified.
  //  Node n = {{}, "AssignVariable", {"resource", "val"}, {{"dtype",
  //  "DT_FLOAT"},
  //            {"update0"}, "CPU:0", "update1"}}
  struct Node {
    // When constructing a NodeDef, the first entry in ret is used as
    // the node name, the remaining values are ignored.
    std::vector<string> ret;
    std::string op;
    std::vector<string> arg;
    std::vector<std::pair<string, AttrValueWrapper>> attr;
    std::vector<string> dep;
    std::string device;

    // Required if the op has zero outputs. Otherwise, ret[0] used as name if
    // name is left empty.
    std::string name;

    std::string GetName() const {
      if (!name.empty()) return name;
      CHECK(!ret.empty());
      return ret[0];
    }
    std::vector<string> original_node_names;
    std::vector<string> original_func_names;

    NodeDef ToNodeDef() const;
  };

  // Creates a FunctionDef from the given parameters. Node inputs must use
  // function encoding (node_name:output_name[:output_index]).
  // - `ret_def` holds a mapping from the function output names from `out_def`
  //   to the node outputs from `node_def`.
  // - `control_ret_def` holds a mapping from the function control
  //   output names to the nodes from `node_def`.
  static FunctionDef Create(
      const std::string& function_name, gtl::ArraySlice<string> in_def,
      gtl::ArraySlice<string> out_def, gtl::ArraySlice<string> attr_def,
      gtl::ArraySlice<Node> node_def,
      gtl::ArraySlice<std::pair<string, string>> ret_def,
      gtl::ArraySlice<std::pair<string, string>> control_ret_def);

  // Creates a FunctionDef from the given parameters. Node inputs must use
  // function encoding (node_name:output_name[:output_index]).
  // - `ret_def` holds a mapping from the function output names from `out_def`
  //   to the node outputs from `node_def`.
  static FunctionDef Create(const std::string& function_name,
                            gtl::ArraySlice<string> in_def,
                            gtl::ArraySlice<string> out_def,
                            gtl::ArraySlice<string> attr_def,
                            gtl::ArraySlice<Node> node_def,
                            gtl::ArraySlice<std::pair<string, string>> ret_def);

  // TODO(josh11b): Get rid of these and transition to the one above.
  static FunctionDef Define(const std::string& function_name,
                            gtl::ArraySlice<string> arg_def,
                            gtl::ArraySlice<string> ret_def,
                            gtl::ArraySlice<string> attr_def,
                            gtl::ArraySlice<Node> node_def);

  // Defines an anonymous function. I.e., its name is not relevant.
  static FunctionDef Define(gtl::ArraySlice<string> arg_def,
                            gtl::ArraySlice<string> ret_def,
                            gtl::ArraySlice<string> attr_def,
                            gtl::ArraySlice<Node> node_def);

  // Helpers to construct a constant scalar.
  template <typename T>
  static Node Const(const std::string& name, const T& val) {
    Node n = {{name}, "Const"};
    const DataType dtype = DataTypeToEnum<T>::value;
    n.attr.push_back({"dtype", dtype});
    Tensor t(dtype, TensorShape({}));
    t.scalar<T>()() = val;
    n.attr.push_back({"value", t});
    return n;
  }

  template <typename T>
  static Node Const(const std::string& name, gtl::ArraySlice<T> vals) {
    Node n = {{name}, "Const"};
    const DataType dtype = DataTypeToEnum<T>::value;
    n.attr.push_back({"dtype", dtype});
    int64_t num = vals.size();
    Tensor t(dtype, TensorShape({num}));
    for (size_t i = 0; i < vals.size(); ++i) {
      t.flat<T>()(i) = vals[i];
    }
    n.attr.push_back({"value", t});
    return n;
  }
};

template <>
inline FunctionDefHelper::AttrValueWrapper::AttrValueWrapper(const char* val) {
  InitFromString(val);
}

template <>
inline FunctionDefHelper::AttrValueWrapper::AttrValueWrapper(
    const std::string& val) {
  InitFromString(val);
}

template <>
inline FunctionDefHelper::AttrValueWrapper::AttrValueWrapper(StringPiece val) {
  InitFromString(val);
}

// Instantiate a function.
//
// "fdef" encodes a TF function with some attrs in fdef.signature.attr
// containing placeholders.  InstantiateFunction binds these
// placeholders and produces an instantiated function encoded in
// "result.gdef". The value to substitute a placeholder is given by
// "attr_values", which is a map from a placeholder name to an attr
// value.
//
// InstantiateFunction calls "get_function" to find signatures of other
// functions and primitive ops.

// GetFunctionSignature(func name, opdef) returns OK if the func name is found
// and opdef is filled with a pointer to the corresponding signature
// (a OpDef proto). Otherwise, returns an error.
typedef std::function<Status(const string&, const OpDef**)>
    GetFunctionSignature;

struct InstantiationResult {
  DataTypeVector arg_types;
  DataTypeVector ret_types;
  std::vector<NodeDef> nodes;
};
Status InstantiateFunction(const FunctionDef& fdef, AttrSlice attr_values,
                           GetFunctionSignature get_function,
                           InstantiationResult* result);

// Returns a debug string for a function definition.
//
// The returned text is multiple-line. It is intended to be
// human-readable rather than being friendly to parsers. It is _NOT_
// intended to be the canonical string representation of "func_def".
// Particularly, it may not include all information presented in
// "func_def" (e.g., comments, description of the function arguments,
// etc.)
std::string DebugString(const FunctionDef& func_def);
std::string DebugString(const GraphDef& instantiated_func_def);
std::string DebugString(gtl::ArraySlice<NodeDef> instantiated_func_nodes);

// Returns a debug string for a top level graph (the main program and
// its supporting functions defined in its library).
std::string DebugStringWhole(const GraphDef& gdef);

// Returns true if f1 == f2. Compares all fields, including descriptions. Order
// of NodeDefs doesn't matter.
bool FunctionDefsEqual(const FunctionDef& f1, const FunctionDef& f2);

// Return a hash of `fdef` that is consistent with FunctionDefsEqual method.
// In other words, if two fdefs compare equal, their hash values will be the
// same.
uint64 FunctionDefHash(const FunctionDef& fdef);

class CallFrameInterface {
 public:
  virtual ~CallFrameInterface() {}

  virtual size_t num_args() const = 0;
  virtual size_t num_retvals() const = 0;

  virtual Status GetArg(int index, const Tensor** val) = 0;

  // Optimized implementation of `GetArg()` that allows the caller to take
  // ownership of the tensor. This method may only be called once per
  // value of `index` and `CallFrameInterface` instance.
  //
  // REQUIRES: `this->CanConsumeArg(index) == true`.
  virtual void ConsumeArg(int index, Tensor* val) {
    LOG(ERROR) << "This `CallFrameInterface` implementation does not support "
                  "consuming arguments.";
  }
  virtual bool CanConsumeArg(int index) const { return false; }

  virtual Status SetRetval(int index, const Tensor& val) = 0;
};

// Represents a function call frame. I.e., the data structure used to
// pass arguments to a function and retrieve its results.
//
// Runtime must arrange accesses to one FunctionCallFrame s.t.
//   1. SetArgs() happens before any GetArg();
//   2. GetRetvals happens after all SetRetval();
class FunctionCallFrame : public CallFrameInterface {
 public:
  FunctionCallFrame(DataTypeSlice arg_types, DataTypeSlice ret_types);
  ~FunctionCallFrame() override;

  // Caller methods.
  Status SetArgs(gtl::ArraySlice<Tensor> args);
  Status GetRetvals(std::vector<Tensor>* rets) const;

  // Moves the return values from the frame to rets. If allow_dead_tensors is
  // false it will fail if any of the retvals do not have a value.
  Status ConsumeRetvals(std::vector<Tensor>* rets, bool allow_dead_tensors);

  size_t num_args() const override { return arg_types_.size(); }
  size_t num_retvals() const override { return ret_types_.size(); }

  // Callee methods.
  Status GetArg(int index, const Tensor** val) override;
  Status SetRetval(int index, const Tensor& val) override;

 private:
  DataTypeVector arg_types_;
  DataTypeVector ret_types_;
  gtl::InlinedVector<Tensor, 4> args_;
  struct Retval {
    bool has_val = false;
    Tensor val;
  };
  gtl::InlinedVector<Retval, 4> rets_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionCallFrame);
};

// Language agnostic stack traces.
class AbstractStackTrace {
 public:
  struct TracePrintingOptions {
    // Show inline the contents of each stack line.
    bool show_line_contents = false;

    // Drop the common largest prefix of all filenames in stack frames.
    bool filter_common_prefix = false;

    // Do not show internal frames.
    bool drop_internal_frames = false;
  };

  virtual ~AbstractStackTrace() {}

  // The returned span is alive as long as the AbstractStackTrace is alive.
  virtual absl::Span<StackFrame const> ToFrames() const = 0;

  // Returns the last stack frame from user code, attempting to ignore the
  // framework code. Returns an empty frame if no such stack frame was found.
  virtual StackFrame LastUserFrame() const = 0;
  virtual std::string ToString(const TracePrintingOptions& opts) const = 0;
};

using StackTracesMap =
    std::unordered_map<std::string,
                       std::shared_ptr<tensorflow::AbstractStackTrace>>;

// Holds Function information that can be shared in multiple places.
// FunctionRecord must be explicitly finalized before being saved in
// FunctionLibraryDefinition or any other place that expects immutability.
class FunctionRecord : public core::RefCounted {
 public:
  FunctionRecord(const FunctionDef& fdef, const StackTracesMap& stack_traces,
                 bool finalized);
  FunctionRecord(FunctionDef&& fdef, const StackTracesMap& stack_traces,
                 bool finalized);

  // Mark FunctionRecord as finalized (disable mutation).
  void finalize();

  // Get a mutable reference to the FunctionDef owned by the record.
  // Will fail if record is finalized.
  StatusOr<FunctionDef*> mutable_fdef();

  // Get an immutable access to FunctionRecord properties.
  const FunctionDef& fdef() const;
  const StackTracesMap& stack_traces() const;
  const OpRegistrationData& op_registration_data() const;
  const bool finalized() const;

 private:
  bool finalized_ = false;

  FunctionDef fdef_;
  const StackTracesMap stack_traces_;
  const OpRegistrationData op_registration_data_;
};

// Helper to maintain a map between function names in a given
// FunctionDefLibrary and function definitions.
//
// This class is thread-safe.
class FunctionLibraryDefinition : public OpRegistryInterface {
 public:
  // Ops created for function arguments bear the name given by `kArgOp`; those
  // created for return values bear the name given by `kRetOp`.
  static constexpr const char* const kArgOp = "_Arg";
  static constexpr const char* const kDeviceArgOp = "_DeviceArg";
  static constexpr const char* const kRetOp = "_Retval";
  static constexpr const char* const kDeviceRetOp = "_DeviceRetval";
  static constexpr const char* const kIntsOnDeviceAttr =
      "experimental_ints_on_device";
  static constexpr const char* const kSharedRendezvousAttr =
      "shared_rendezvous";

  static constexpr const char* const kGradientOp = "SymbolicGradient";
  static constexpr const char* const kFuncAttr = "f";

  // Note: This constructor grabs `lib_def`'s lock in shared mode.
  FunctionLibraryDefinition(const FunctionLibraryDefinition& lib_def);
  FunctionLibraryDefinition(const OpRegistryInterface* default_registry,
                            const FunctionDefLibrary& lib_def = {});
  ~FunctionLibraryDefinition() override;

  FunctionLibraryDefinition& operator=(const FunctionLibraryDefinition&) =
      delete;
  FunctionLibraryDefinition& operator=(FunctionLibraryDefinition&& other);

  // Returns True if the library contains `func`, False otherwise.
  bool Contains(const std::string& func) const TF_LOCKS_EXCLUDED(mu_);

  // Returns nullptr if "func" is not defined in "lib_def". Otherwise,
  // returns its definition proto.
  //
  // NB: This function returns a borrowed pointer, which can be invalidated by a
  // subsequent call to `ReplaceFunction()` with the given name.
  const FunctionDef* Find(const std::string& func) const TF_LOCKS_EXCLUDED(mu_);

  // Returns nullptr if "func" is not defined in "lib_def". Otherwise,
  // returns a strong reference pointer to the FunctionRecord in the library.
  core::RefCountPtr<FunctionRecord> FindRecord(const std::string& func) const
      TF_LOCKS_EXCLUDED(mu_);

  // Adds function definition 'fdef' to this function library.
  // Returns status 'ok' on success, or error otherwise. This is a no-op if
  // 'fdef' already exists in this function library.
  // If 'fdef' is successfully added to the library, it will be accessible
  // from 'LookUp' and included in the proto returned by 'ToProto'.
  // This operation is atomic.
  //
  // Associates `graph` with a function `func_name`. Lifetime assumption:
  // `graph` has to outlive all instantiated graphs.
  Status AddFunctionDef(const FunctionDef& fdef,
                        const StackTracesMap& stack_traces = {})
      TF_LOCKS_EXCLUDED(mu_);

  // Adds gradient definition 'grad' to this function library.
  // This is a no-op if 'grad' already exists in this function library.
  // If 'grad' is successfully added, it will be accessible via 'FindGradient'
  // and included in the proto returned by 'ToProto'.
  // This operation is atomic.
  Status AddGradientDef(const GradientDef& grad) TF_LOCKS_EXCLUDED(mu_);

  // Replaces the function corresponding to `func` with `fdef`. Returns
  // a non-OK status if "func" was not found in the library, OK otherwise.
  // Please be careful when replacing function: make sure all previous pointers
  // returned by `Find()` are no longer in use.
  Status ReplaceFunction(const std::string& func, const FunctionDef& fdef,
                         const StackTracesMap& stack_traces = {})
      TF_LOCKS_EXCLUDED(mu_);

  // Replaces the gradient corresponding to `grad.function_name()`. Returns
  // a non-OK status if "grad.function_name()" was not found in the library, OK
  // otherwise.
  Status ReplaceGradient(const GradientDef& grad) TF_LOCKS_EXCLUDED(mu_);

  // Removes the function corresponding to 'func'. Returns a non-OK status if
  // 'func' was not found in the library, OK otherwise.
  // Please be careful when removing function: make sure there are no other
  // nodes using the function, and all previous pointers returned by `Find()`
  // are no longer in use.
  Status RemoveFunction(const std::string& func) TF_LOCKS_EXCLUDED(mu_);

  // Removes all the functions and gradient functions.
  void Clear() TF_LOCKS_EXCLUDED(mu_);

  // Adds the functions and gradients in 'other' to this function library.
  // Duplicate functions and gradients are ignored.
  // This operation is atomic.
  Status AddLibrary(const FunctionLibraryDefinition& other)
      TF_LOCKS_EXCLUDED(mu_);
  Status AddLibrary(FunctionLibraryDefinition&& other) TF_LOCKS_EXCLUDED(mu_);

  // Adds the functions and gradients in 'lib_def' to this function library.
  // Duplicate functions and gradients are ignored. This overload adds the
  // functions with no stack traces. This operation is atomic.
  Status AddLibrary(const FunctionDefLibrary& lib_def) TF_LOCKS_EXCLUDED(mu_);
  Status AddLibrary(FunctionDefLibrary&& lib_def) TF_LOCKS_EXCLUDED(mu_);

  // Adds the functions and gradients in 'lib_def' to this function library.
  // Duplicate functions and gradients are ignored.
  // This operation is atomic.
  Status AddLibrary(const FunctionDefLibrary& lib_def,
                    const StackTracesMap& stack_traces) TF_LOCKS_EXCLUDED(mu_);
  Status AddLibrary(FunctionDefLibrary&& lib_def,
                    const StackTracesMap& stack_traces) TF_LOCKS_EXCLUDED(mu_);

  // If the gradient function for 'func' is specified explicitly in
  // the library, returns the gradient function name.  Otherwise,
  // returns an empty string.
  std::string FindGradient(const std::string& func) const
      TF_LOCKS_EXCLUDED(mu_);

  // OpRegistryInterface method. Useful for constructing a Graph.
  //
  // If "op" is defined in the library, returns its signature.
  // Otherwise, assume "op" is a primitive op and returns its op
  // signature and shape inference function.
  //
  // NB: This function outputs a borrowed pointer, which can be invalidated by a
  // subsequent call to `ReplaceFunction()` with the given name.
  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override
      TF_LOCKS_EXCLUDED(mu_);

  // Generates new function name with the specified prefix that is unique
  // across this library.
  std::string UniqueFunctionName(StringPiece prefix) const
      TF_LOCKS_EXCLUDED(mu_);

  // Given a node def 'ndef', inspects attributes of the callee
  // function to derive the attribute 'value' for 'attr'. Returns OK
  // iff the attribute is given by the function's definition.
  // TODO(irving): Remove; keep only the const Node& version.
  template <typename T>
  Status GetAttr(const NodeDef& ndef, const std::string& attr, T* value) const;

  // Given a node, inspects attributes of the callee function to derive the
  // attribute 'value' for 'attr'. Returns OK iff the attribute is given by the
  // function's definition.
  template <typename T>
  Status GetAttr(const Node& node, const std::string& attr, T* value) const;

  // Returns a proto representation of the state of this function library.
  FunctionDefLibrary ToProto() const TF_LOCKS_EXCLUDED(mu_);

  size_t num_functions() const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    return records_.size();
  }

  // Returns all the function names in the FunctionLibraryDefinition.
  std::vector<string> ListFunctionNames() const TF_LOCKS_EXCLUDED(mu_);

  const OpRegistryInterface* default_registry() const {
    return default_registry_;
  }
  void set_default_registry(const OpRegistryInterface* registry) {
    default_registry_ = registry;
  }

  // Returns a copy of `*this` with only the subset of functions that are
  // reachable from the nodes of `graph` or `func`.
  FunctionLibraryDefinition ReachableDefinitions(const GraphDef& graph) const;
  FunctionLibraryDefinition ReachableDefinitions(const FunctionDef& func) const;

  // Copies the function named `func` from `other` to this
  // FunctionLibraryDefinition.
  // REQUIRES: `this->default_registry() == other.default_registry()`.
  // Returns OK on success, or error otherwise. This is a no-op if a function
  // name `func` already exists in this function library, and has the same
  // implementation as in `other`. If the implementations conflict, an invalid
  // argument error is returned.
  Status CopyFunctionDefFrom(const std::string& name,
                             const FunctionLibraryDefinition& other);

  // Returns graph with debug stack traces for the given function, or `nullptr`
  // if none found.
  const StackTracesMap* GetStackTraces(const std::string& func_name) const {
    core::RefCountPtr<FunctionRecord> entry = FindRecord(func_name);
    if (entry.get() != nullptr) {
      return &entry->stack_traces();
    }
    return nullptr;
  }

  // Adds or updates an OptimizedFunctionGraph. Key is `function_name`.
  void AddOptimizedFunctionGraph(const std::string& function_name,
                                 const OptimizedFunctionGraph& graph)
      TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock l(mu_);
    optimized_function_graph_map_.emplace(function_name, graph);
  }

  // Look up for OptimizedFunctionGraph given `function_name`. Returns nullptr
  // if not found.
  OptimizedFunctionGraph* FindOptimizedFunctionGraph(
      const std::string& function_name) const TF_LOCKS_EXCLUDED(mu_) {
    tf_shared_lock l(mu_);
    if (auto it = optimized_function_graph_map_.find(function_name);
        it != optimized_function_graph_map_.end()) {
      return &(it->second);
    }
    return nullptr;
  }

 private:
  core::RefCountPtr<FunctionRecord> FindHelper(const string& func) const
      TF_SHARED_LOCKS_REQUIRED(mu_);
  std::string FindGradientHelper(const std::string& func) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  Status AddHelper(FunctionRecord* registration, bool* added)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Same as AddFunctionDef/AddGradientDef except these methods set
  // `added` to true if the `fdef`/`grad` were actually added to this.
  Status AddFunctionDefHelper(FunctionDef&& fdef,
                              const StackTracesMap& stack_traces, bool* added)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status AddGradientDefHelper(const GradientDef& grad, bool* added)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper function for GetAttr. Returns the FunctionDef* to get the
  // attr from.
  const FunctionDef* GetAttrImpl(const NodeDef& ndef) const
      TF_LOCKS_EXCLUDED(mu_);

  // Remove all functions in `funcs` and all gradients of functions in
  // `funcs_with_grads` from this library.
  Status Remove(const std::vector<string>& funcs,
                const std::vector<string>& funcs_with_grads)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Remove `func` from the library. Returns non-OK Status unless `func` is in
  // the library. This should only be called when there is a guarantee that the
  // function being removed hasn't been retrieved with `Find`.
  Status RemoveFunctionHelper(const std::string& func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Remove gradient of function `func` from the library. Returns non-OK Status
  // unless `func` has a gradient.
  Status RemoveGradient(const std::string& func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  const OpRegistryInterface* default_registry_;
  gtl::FlatMap<string, FunctionRecord*> records_ TF_GUARDED_BY(mu_);
  gtl::FlatMap<string, string> func_grad_ TF_GUARDED_BY(mu_);
  // Maps from function name to optimized function graph.
  gtl::FlatMap<string, OptimizedFunctionGraph> optimized_function_graph_map_
      TF_GUARDED_BY(mu_);
};

// Forward declare. Defined in common_runtime/function.h
struct FunctionBody;

// Forward declare. Defined in common_runtime/device.h
class Device;
// Forward declare. Defined in common_runtime/device_mgr.h
class DeviceMgr;

// Index of an _Arg node.
struct FunctionArgIndex {
  explicit FunctionArgIndex(const int index) : index(index) {}
  FunctionArgIndex(const int index, const int sub_index)
      : index(index), sub_index(sub_index) {}

  // The value of the attribute "Index" of the _Arg node.
  int index;
  // Set only when the _Arg node represents multiple arguments (e.g. an _Arg
  // node is replicated to multiple devices/subgraphs). Use sub-index to
  // distinguish arguments with the same index.
  int sub_index = -1;
};

class FunctionLibraryRuntime {
 public:
  virtual ~FunctionLibraryRuntime() {}

  // Instantiate a function with the given "attrs".
  //
  // Returns OK and fills in "handle" if the instantiation succeeds.
  // Otherwise returns an error and "handle" is undefined.
  struct InstantiateOptions {
    // The canonical device name of the device on which the function
    // should be instantiated. If empty, the function will be
    // instantiated on the local device.
    std::string target;

    // Should the function be instantiated as a multi-device function?
    bool is_multi_device_function = false;

    // If true, graph passes will be skipped when instantiating the function
    // since they have already run on the main function side.
    bool is_component_function = false;

    // For multi-device functions, a vector of canonical device names for
    // function's inputs. The device of resource inputs must be the device
    // backing the resource, not the CPU device backing the resource handle.
    // Must have the same length as number of inputs to the function.
    std::vector<string> input_devices;

    // For multi-device functions, a vector of canonical device names for
    // function's outputs.
    //
    // (a) If specified (must have the same length as number of outputs):
    //
    // Specified devices will be assigned to Retval nodes inserted into the
    // function body graph in place of function outputs. It is allowed to
    // specify output device as empty string, in this case Retval device
    // assignment will be inferred later when function graph will be placed
    // before partitioning (this is required for resource outputs). Placer will
    // respect colocation constraints.
    //
    // (b) If not specified:
    //
    // Function runtime will infer Retval device by following input edges, until
    // it will reach a node with a device specification. This device
    // specification must identify a unique device, i.e. a general specification
    // like "job:foo" matching multiple devices will result in an error.
    //
    // IMPORTANT: Resource outputs
    //
    // Multi device functions might return resources on a devices different from
    // the function call device. If output device is not specified for the
    // resource output, and node producing that resource is a function call,
    // runtime will leave device specification empty and will rely on Placer to
    // infer correct device.
    std::vector<string> output_devices;

    // If set, it indicates the original output indices of a component function.
    absl::optional<std::vector<int>> ret_indices = absl::nullopt;

    // Maps from a CompositeDevice name to a list of underlying physical
    // devices.
    absl::flat_hash_map<string, const std::vector<string>*> composite_devices;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // For multi-device functions, a mapping from _Arg node index to type and
    // shape for input resources.
    // REQUIRES: if input_resource_dtypes_and_shapes.count(i) > 0 then i-th
    // argument type must be DT_RESOURCE.
    std::unordered_map<int, DtypeAndPartialTensorShape>
        input_resource_dtypes_and_shapes;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // If non-null, the runtime will use `lib_def` to resolve function(s) named
    // in `function_name` and `attrs`. Otherwise, the runtime will use its
    // internal library.
    //
    // NOTE(mrry): If provided, all functions defined in `lib_def` must be
    // self-contained, and cannot refer to functions defined in other libraries.
    const FunctionLibraryDefinition* lib_def = nullptr;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // If non-empty, the runtime will use `state_handle` to identify
    // cached state related the instantiated function. Two functions
    // of the same name and attrs, instantiated with the same
    // `state_handle` will have the same handle and share the same
    // state (in stateful kernels); and two functions with different
    // values for `state_handle` will have independent state.
    std::string state_handle;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // Instantiates the function using an executor of the given type. If empty,
    // the default TensorFlow executor will be used.
    std::string executor_type;

    // If true, the runtime will attempt to create kernels for the function at
    // instantiation time, rather than on the first run. This can be used to
    // surface errors earlier.
    bool create_kernels_eagerly = false;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // Instantiates the function with the provided config_proto.
    ConfigProto config_proto;

    // If provided, this optimization function will be invoked before
    // the placer for multi-device functions.
    std::function<Status(std::vector<string> /*ret_node_names*/,
                         std::vector<string> /*keep_node_names*/,
                         FunctionLibraryDefinition*, const DeviceSet&,
                         Device* /*cpu_device*/, std::unique_ptr<Graph>*)>
        optimize_graph_fn;

    // If set, partitioned functions will be added to `graph_collector`.
    // `graph_collector` must be alive during the call to Instantiate.
    GraphCollector* graph_collector = nullptr;

    // Indicates whether the multi-device function backend should default the
    // placement of ops without request device to `target`.
    bool default_device_to_target = true;

    // If true, the optimized Graph will be stored so that
    // `FunctionLibraryRuntime::DebugString(handle)` contains the optimized
    // Graph. Otherwise, the unoptimized function Graph will be returned.
    bool include_optimized_graph_in_debug_string = false;

    // If true, the function library runtime cache the function instantiation.
    bool use_function_cache = false;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // If True, allow optimizations which should be targeted at a limited
    // set of small functions.  For example, running kernels synchronously can
    // be faster under some conditions.
    bool allow_small_function_optimizations = false;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // If True, allow graphs containing control flow nodes to be run on the
    // single threaded executor.
    bool allow_control_flow_sync_execution = false;

    // TODO(b/176491312): Remove this if shape inference on import flag is
    // removed. If True, allows mlir roundtrip to run shape inference on import.
    bool shape_inference_on_tfe_dialect_import = true;

    // Force int32 _Arg and _Retvals nodes to be left on device instead of
    // pinning to host.
    //
    // Note that we do not pin int32 nodes to host for subgraphs running in
    // TPU/XLA devices. So this is mainly used to handle the case of multi-CPU
    // and GPU (non-XLA) graphs.
    bool int_args_and_retvals_on_device = false;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // Instantiates the function for XLA compilation on device_type. If empty,
    // function is not compiled.
    std::string xla_compile_device_type;
  };
  typedef uint64 Handle;
  virtual Status Instantiate(const std::string& function_name, AttrSlice attrs,
                             const InstantiateOptions& options,
                             Handle* handle) = 0;
  Status Instantiate(const std::string& function_name, AttrSlice attrs,
                     Handle* handle) {
    auto opts = absl::make_unique<InstantiateOptions>();
    return Instantiate(function_name, attrs, *opts, handle);
  }

  // Releases state associated with the handle.
  virtual Status ReleaseHandle(Handle handle) = 0;

  // Returns the function body for the instantiated function given its
  // handle 'h'. Returns nullptr if "h" is not found.
  //
  // *this keeps the ownership of the returned object, which remains alive
  // as long as *this.
  virtual const FunctionBody* GetFunctionBody(Handle h) = 0;

  // Returns the return types for the function identified by handle `h`.
  virtual Status GetRetTypes(Handle h, DataTypeVector* ret_types) = 0;

  // Asynchronously invokes the instantiated function identified by
  // "handle".
  //
  // If function execution succeeds, "done" is called with OK and
  // "*rets" is filled with the function's return values. Otherwise,
  // "done" is called with an error status.
  //
  // Does not take ownership of "rets".
  // In the cross-process scenario, runner isn't used for making the Async
  // RPC calls.
  struct Options {
    Options() {}
    explicit Options(const int64_t step_id) : step_id(step_id) {}

    // Choose a step ID that is guaranteed not to clash with any
    // Session-generated step ID. DirectSession only generates
    // non-negative step IDs (contiguous, starting from 0), and
    // MasterSession generates 56-bit random step IDs whose MSB is
    // always 0, so a negative random step ID should suffice.
    const int64_t step_id = -std::abs(static_cast<int64_t>(random::New64()));

    // op_id of the function running in eager mode. Set when we want to copy
    // remote outputs lazily. All components of a remote multi-device function
    // should use the same op_id, in order to correctly map remote output
    // tensors to the remote TensorHandles in the default device.
    absl::optional<int64_t> op_id = absl::nullopt;

    // Not owned. Caller makes sure that the rendezvous outlives this Options.
    RendezvousInterface* rendezvous = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    CollectiveExecutor* collective_executor = nullptr;
    ScopedStepContainer* step_container = nullptr;
    StepStatsCollectorInterface* stats_collector = nullptr;
    tsl::CoordinationServiceAgent* coordination_service_agent = nullptr;

    absl::optional<ManagedStackTrace> stack_trace = absl::nullopt;

    std::function<void(std::function<void()>)>* runner = nullptr;

    // Parameters for remote function execution.
    bool remote_execution = false;
    std::string source_device = "";  // Fully specified device name.

    // Allocator attributes specifying where the args are / rets should be put.
    // These should either be {} or match the length of args / retvals. If {},
    // the default allocator attributes will be assumed for all args / retvals.
    std::vector<AllocatorAttributes> args_alloc_attrs;
    std::vector<AllocatorAttributes> rets_alloc_attrs;

    // If true, we create a new IntraProcessRendezvous, else use the existing
    // one.
    bool create_rendezvous = false;

    // If True, allow returning dead tensors.
    bool allow_dead_tensors = false;

    // If True, hint that all kernels should be treated as "inexpensive", and
    // hence executed on the scheduling thread.
    bool run_all_kernels_inline = false;

    // If not null, use this thread pool for intra op scheduling.
    thread::ThreadPoolInterface* user_intra_op_threadpool = nullptr;

    // Returns a human readable representation of this.
    std::string DebugString() const;
  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void Run(const Options& opts, Handle handle,
                   gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                   DoneCallback done) = 0;
  virtual void Run(const Options& opts, Handle handle,
                   CallFrameInterface* call_frame, DoneCallback done) = 0;

  virtual Status RunSync(Options opts, Handle handle,
                         gtl::ArraySlice<Tensor> args,
                         std::vector<Tensor>* rets) = 0;
  virtual Status RunSync(Options opts, Handle handle,
                         CallFrameInterface* call_frame) = 0;

  // Creates a "kernel" for the given NodeProperties "props".
  //
  // If succeeds, returns OK and the caller takes the ownership of the
  // returned "*kernel". Otherwise, returns an error.
  virtual Status CreateKernel(
      const std::shared_ptr<const NodeProperties>& props,
      OpKernel** kernel) = 0;

  // Returns true iff the function named `function_name` is stateful.
  //
  // NOTE(mrry): This method assumes that the runtime is associated with a
  // default function library, and looks up `function_name` in that library.
  // It does not support overriding the function library.
  virtual bool IsStateful(const std::string& function_name) const = 0;

  // Returns the device on which the function executes.
  virtual Device* device() = 0;
  virtual const Device* device() const = 0;

  // Returns the default runner in which the ops should be launched. If the
  // device on which the function executes has a private thread pool, return
  // runner on the device local thread pool.
  virtual std::function<void(std::function<void()>)>* runner() = 0;

  // Get the DeviceMgr from which the device was obtained.
  virtual const DeviceMgr* device_mgr() const = 0;

  // Returns the function library definition that backs this runtime.
  //
  // NOTE(mrry): The returned library definition is the default function library
  // for this runtime. The caller may override the function library used by the
  // runtime to instantiate functions, which will not be reflected in the return
  // value of this function.
  virtual const FunctionLibraryDefinition* GetFunctionLibraryDefinition()
      const = 0;

  // Returns the environment on which the function executes.
  virtual Env* env() = 0;

  // Returns the ConfigProto passed to the session used to create the function.
  virtual const ConfigProto* const config_proto() = 0;

  // Returns a debug string showing the definition of the function of
  // 'handle'.
  virtual std::string DebugString(Handle handle) = 0;

  // Returns the graph version number.
  virtual int graph_def_version() const = 0;

  typedef uint64 LocalHandle;

  // Creates a copy of ProcessFunctionLibraryRuntime (transferring ownership to
  // the caller), FunctionLibraryRuntime (owned by the returned
  // ProcessFunctionLibraryRuntime), FunctionLibraryDefinition (transferring
  // ownership to the caller). Note that both the ProcessFunctionLibraryRuntime
  // and FunctionLibraryRuntime borrow a pointer to the
  // FunctionLibraryDefinition and so the FunctionLibraryDefinition should
  // outlive both.
  //
  // The `skip_flib_def` argument controls whether the method should clone the
  // FunctionLibraryDefinition (default behavior) or return an empty function
  // library. The latter is used by tf.data, which manages
  // FunctionLibraryDefinitions for its functions independently (and passes
  // these into the FunctionLibraryRuntime through an overlay), to avoid linear
  // runtime w.r.t. to number of functions in the current function library.
  virtual Status Clone(std::unique_ptr<FunctionLibraryDefinition>* out_lib_def,
                       std::unique_ptr<ProcessFunctionLibraryRuntime>* out_pflr,
                       FunctionLibraryRuntime** out_flr,
                       bool skip_flib_def = false) = 0;

  // Returns the name of the executor class (in the sense of
  // `ExecutorFactory::GetFactory()`) that will be used based on the given
  // dynamic `options` and static `attrs`. If none is specified, this method
  // will return an empty string, which leaves the decision up to the runtime.
  static std::string ExecutorType(const InstantiateOptions& options,
                                  AttrSlice attrs);
};

// Returns the device of the `arg_index`-th function input. Update
// `composite_devices` if the input device is a composite device.
std::string GetFunctionResourceInputDevice(
    const Tensor& input, const int arg_index, const FunctionDef& function_def,
    absl::flat_hash_map<string, std::vector<string>>* composite_devices);

// Returns a canonicalized string for the instantiation of the function of the
// given "name", attributes "attrs", and "options".
//
// The returned string is guaranteed to be stable within one address space. But
// it may be change as the implementation evolves. Therefore, it should not be
// persisted or compared across address spaces.
std::string Canonicalize(
    const std::string& funcname, AttrSlice attrs,
    const FunctionLibraryRuntime::InstantiateOptions& options);
std::string Canonicalize(const std::string& funcname, AttrSlice attrs);

const FunctionLibraryRuntime::Handle kInvalidHandle = -1;
const FunctionLibraryRuntime::LocalHandle kInvalidLocalHandle = -1;

class CustomKernelCreator {
 public:
  virtual ~CustomKernelCreator() {}

  // Given a NodeDef 'node_def' and the function library runtime 'flr',
  // validate if the class supports creating such a kernel.
  virtual bool CanCreateKernel(
      const FunctionLibraryRuntime& flr,
      const std::shared_ptr<const NodeProperties>& props) const = 0;

  // Given a supported NodeDef, returns a kernel that computes the node.
  virtual Status CreateKernel(
      FunctionLibraryRuntime* flr,
      const std::shared_ptr<const NodeProperties>& props,
      std::unique_ptr<OpKernel>* kernel) const = 0;
};

typedef
#if !defined(IS_MOBILE_PLATFORM)
    absl::variant<Tensor, eager::RemoteTensorHandle*>
        FunctionArg;
#else
    absl::variant<Tensor>
        FunctionArg;
#endif

// Either a local tensor or the shape of a remote tensor.
typedef absl::variant<Tensor, TensorShape> FunctionRet;

// Used to instantiate and run functions in a distributed system.
class DistributedFunctionLibraryRuntime {
 public:
  virtual ~DistributedFunctionLibraryRuntime() {}

  // Instantiate a function on a remote target specified in `options.target`, by
  // sending the name and definition of the function to the remote worker. The
  // local `handle` is filled for the instantiated function data and can be used
  // for subsequent run function calls on the remote target.
  virtual void Instantiate(
      const std::string& function_name,
      const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::LocalHandle* handle,
      FunctionLibraryRuntime::DoneCallback done) = 0;

  // Run an instantiated remote function (specified by `handle`) with a list of
  // input Tensors in `args` and get its output Tensors in `rets`. The input
  // tensor data will be sent with the function execution request, and must be
  // available on the current caller side.
  // opts.runner isn't used for execution.
  virtual void Run(const FunctionLibraryRuntime::Options& opts,
                   FunctionLibraryRuntime::LocalHandle handle,
                   gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                   FunctionLibraryRuntime::DoneCallback done) = 0;

  // Run an instantiated remote function (specified by `handle`) with a list of
  // input Tensors or RemoteTensorHandles as `args` and get its output Tensors
  // or TensorShapes in `rets`. When using RemoteTensorHandles as function
  // inputs or TensorShapes as outputs, the corresponding tensor data will be
  // resolved on the remote worker, so it is not required to be locally
  // available on the caller side. Using RemoteTensorHandle inputs is not
  // supported in TensorFlow v1 runtime.
  virtual void Run(const FunctionLibraryRuntime::Options& opts,
                   FunctionLibraryRuntime::LocalHandle handle,
                   gtl::ArraySlice<FunctionArg> args,
                   std::vector<FunctionRet>* rets,
                   FunctionLibraryRuntime::DoneCallback done) = 0;

  // Clean up a previously instantiated function on remote worker.
  virtual void CleanUp(uint64 step_id,
                       FunctionLibraryRuntime::LocalHandle handle,
                       FunctionLibraryRuntime::DoneCallback done) = 0;

  // DeviceMgr with *all* available devices (i.e., local and remote).
  virtual DeviceMgr* remote_device_mgr() const = 0;
};

// Extracts the actual type from "attr_values" based on its definition
// "arg_def".
//
// If "arg_def" is a N*T type, *is_type_list is set to false, and
// *dtypes is set to be a vector of size N and each element is T.
//
// If "arg_def" is a list(type), *is_type_list is set to true, and
// *dtypes is set to be a vector of types specified in attrs for
// arg_def.
//
// Otherwise (arg_def is a simple type T), *is_type_list is set to
// false, and *dtypes is set to a single element vector, whose only
// element is T.
Status ArgNumType(AttrSlice attrs, const OpDef::ArgDef& arg_def,
                  bool* is_type_list, DataTypeVector* dtypes);

// To register a gradient function for a builtin op, one should use
//   REGISTER_OP_GRADIENT(<op_name>, <c++ grad factory>);
//
// Typically, the c++ grad factory is a plan function that can be
// converted into ::tensorflow::gradient::Creator, which is
//   std::function<Status(const AttrSlice&, FunctionDef*)>.
//
// A ::tensorflow::gradient::Creator should populate in FunctionDef* with a
// definition of a brain function which compute the gradient for the
// <op_name> when the <op_name> is instantiated with the given attrs.
//
// E.g.,
//
// Status MatMulGrad(const AttrSlice& attrs, FunctionDef* g) {
//   bool transpose_a;
//   TF_RETURN_IF_ERROR(attrs.Get("transpose_a", &transpose_a));
//   bool transpose_b;
//   TF_RETURN_IF_ERROR(attrs.Get("transpose_b", &transpose_b));
//   DataType dtype;
//   TF_RETURN_IF_ERROR(attrs.Get("dtype", &dtype));
//   if (!transpose_a && !transpose_b) {
//     *g = FunctionDefHelper::Define(
//       "MatMulGrad",
//       {"x:T ", "y:T", "dz:T"},    // Inputs to this function
//       {"dx:T", "dy:T"},           // Outputs from this function
//       {"T: {float, double}"},     // Attributes needed by this function
//       {
//         {{"x_t"}, "Transpose", {"x"}, {{"T", "$T"}}},
//         {{"y_t"}, "Transpose", {"y"}, {{"T", "$T"}}},
//         {{"dx"}, "MatMul", {"dz", "y_t"}, {{"T", "$T"}}},
//         {{"dy"}, "MatMul", {"x_", "dz"}, {{"T", "$T"}}},
//       });
//   } else {
//     ... ...
//   }
//   return OkStatus();
// }
//
// NOTE: $T is substituted with the type variable "T" when the
// gradient function MatMul is instantiated.
//
// TODO(zhifengc): Better documentation somewhere.

// Macros to define a gradient function factory for a primitive
// operation.
#define REGISTER_OP_GRADIENT(name, fn) \
  REGISTER_OP_GRADIENT_UNIQ_HELPER(__COUNTER__, name, fn)

#define REGISTER_OP_NO_GRADIENT(name) \
  REGISTER_OP_GRADIENT_UNIQ_HELPER(__COUNTER__, name, nullptr)

#define REGISTER_OP_GRADIENT_UNIQ_HELPER(ctr, name, fn) \
  REGISTER_OP_GRADIENT_UNIQ(ctr, name, fn)

#define REGISTER_OP_GRADIENT_UNIQ(ctr, name, fn)      \
  static bool unused_grad_##ctr TF_ATTRIBUTE_UNUSED = \
      SHOULD_REGISTER_OP_GRADIENT &&                  \
      ::tensorflow::gradient::RegisterOp(name, fn)

namespace gradient {
// Register a gradient creator for the "op".
typedef std::function<Status(const AttrSlice& attrs, FunctionDef*)> Creator;
bool RegisterOp(const std::string& op, Creator func);

// Returns OK the gradient creator for the "op" is found (may be
// nullptr if REGISTER_OP_NO_GRADIENT is used.
Status GetOpGradientCreator(const std::string& op, Creator* creator);
};  // namespace gradient

// Declare explicit instantiations of GetAttr
#define GET_ATTR(T)                                          \
  extern template Status FunctionLibraryDefinition::GetAttr( \
      const Node&, const string&, T*) const;                 \
  extern template Status FunctionLibraryDefinition::GetAttr( \
      const NodeDef&, const string&, T*) const;
GET_ATTR(string)
GET_ATTR(bool)
#undef GET_ATTR

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_FUNCTION_H_

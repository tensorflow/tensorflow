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

#ifndef TENSORFLOW_FRAMEWORK_FUNCTION_H_
#define TENSORFLOW_FRAMEWORK_FUNCTION_H_

#include <vector>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

class CancellationManager;
class GraphDef;
class OpKernel;
class ResourceMgr;
class Rendezvous;
class ScopedStepContainer;
class StepStatsCollector;
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
      const string& name,
      gtl::ArraySlice<std::pair<string, AttrValueWrapper>> attrs);
  static AttrValueWrapper FunctionRef(const string& name) {
    return FunctionRef(name, {});
  }

  // Node is used to construct FunctionDef.Node using initialization
  // lists. E.g.,
  //  Node n = {{"z"}, "Mul", {"x", "y"}, {{"T", "$T"}}};  // z = x * y
  struct Node {
    // When constructing a NodeDef, the first entry in ret is used as
    // the node name, the remaining values are ignored.
    std::vector<string> ret;
    string op;
    std::vector<string> arg;
    std::vector<std::pair<string, AttrValueWrapper>> attr;
    std::vector<string> dep;

    NodeDef ToNodeDef() const;
  };

  // The Create() function uses the new NodeDef field.  `ret_def`
  // holds a mapping from the function output names from `out_def` to
  // the node outputs from `node_def`.
  static FunctionDef Create(const string& function_name,
                            gtl::ArraySlice<string> in_def,
                            gtl::ArraySlice<string> out_def,
                            gtl::ArraySlice<string> attr_def,
                            gtl::ArraySlice<Node> node_def,
                            gtl::ArraySlice<std::pair<string, string>> ret_def);

  // The two Define() functions use the old FunctionDef::Node field.
  // TODO(josh11b): Get rid of these and transition to the one above.
  static FunctionDef Define(const string& function_name,
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
  static Node Const(const string& name, const T& val) {
    Node n = {{name}, "Const"};
    const DataType dtype = DataTypeToEnum<T>::value;
    n.attr.push_back({"dtype", dtype});
    Tensor t(dtype, TensorShape({}));
    t.scalar<T>()() = val;
    n.attr.push_back({"value", t});
    return n;
  }

  template <typename T>
  static Node Const(const string& name, gtl::ArraySlice<T> vals) {
    Node n = {{name}, "Const"};
    const DataType dtype = DataTypeToEnum<T>::value;
    n.attr.push_back({"dtype", dtype});
    int64 num = vals.size();
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
    const string& val) {
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
string DebugString(const FunctionDef& func_def);
string DebugString(const GraphDef& instantiated_func_def);
string DebugString(gtl::ArraySlice<NodeDef> instantiated_func_nodes);

// Returns a debug string for a top level graph (the main program and
// its supporting functions defined in its library).
string DebugStringWhole(const GraphDef& gdef);

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

  virtual Status GetArg(int index, Tensor* val) const = 0;
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
  ~FunctionCallFrame();

  // Caller methods.
  Status SetArgs(gtl::ArraySlice<Tensor> args);
  Status GetRetvals(std::vector<Tensor>* rets) const;
  Status ConsumeRetvals(std::vector<Tensor>* rets);

  size_t num_args() const override { return arg_types_.size(); }
  size_t num_retvals() const override { return ret_types_.size(); }

  // Callee methods.
  Status GetArg(int index, Tensor* val) const override;
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

// Helper to maintain a map between function names in a given
// FunctionDefLibrary and function definitions.
class FunctionLibraryDefinition : public OpRegistryInterface {
 public:
  explicit FunctionLibraryDefinition(const FunctionLibraryDefinition& lib_def);
  FunctionLibraryDefinition(const OpRegistryInterface* default_registry,
                            const FunctionDefLibrary& lib_def);
  ~FunctionLibraryDefinition() override;

  FunctionLibraryDefinition& operator=(const FunctionLibraryDefinition&) =
      delete;

  // Returns nullptr if "func" is not defined in "lib_def". Otherwise,
  // returns its definition proto.
  const FunctionDef* Find(const string& func) const;

  // Adds function definition 'fdef' to this function library.
  // Returns status 'ok' on success, or error otherwise. This is a no-op if
  // 'fdef' already exists in this function library.
  // If 'fdef' is successfully added to the library, it will be accessible
  // from 'LookUp' and included in the proto returned by 'ToProto'.
  // This operation is atomic.
  Status AddFunctionDef(const FunctionDef& fdef);

  // Adds gradient definition 'grad' to this function library.
  // This is a no-op if 'grad' already exists in this function library.
  // If 'grad' is successfully added, it will be accessible via 'FindGradient'
  // and included in the proto returned by 'ToProto'.
  // This operation is atomic.
  Status AddGradientDef(const GradientDef& grad);

  // Remove function `func` from the library. Returns non-OK Status unless
  // `func` is in the library.
  Status RemoveFunction(const string& func);

  // Remove gradient of function `func` from the library. Returns non-OK Status
  // unless `func` has a gradient.
  Status RemoveGradient(const string& func);

  // Adds the functions and gradients in 'other' to this function library.
  // Duplicate functions and gradients are ignored.
  // This operation is atomic.
  Status AddLibrary(const FunctionLibraryDefinition& other);

  // Adds the functions and gradients in 'lib_def' to this function library.
  // Duplicate functions and gradients are ignored.
  // This operation is atomic.
  Status AddLibrary(const FunctionDefLibrary& lib_def);

  // If the gradient function for 'func' is specified explicitly in
  // the library, returns the gradient function name.  Otherwise,
  // returns an empty string.
  string FindGradient(const string& func) const;

  // OpRegistryInterface method. Useful for constructing a Graph.
  //
  // If "op" is defined in the library, returns its signature.
  // Otherwise, assume "op" is a primitive op and returns its op
  // signature and shape inference function.
  Status LookUp(const string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;

  static constexpr const char* const kGradientOp = "SymbolicGradient";
  static constexpr const char* const kFuncAttr = "f";

  // Given a node def 'ndef', inspects attributes of the callee
  // function to derive the attribute 'value' for 'attr'. Returns OK
  // iff the attribute is given by the function's definition.
  // TODO(irving): Remove; keep only the const Node& version.
  template <typename T>
  Status GetAttr(const NodeDef& ndef, const string& attr, T* value) const;

  // Given a node, inspects attributes of the callee function to derive the
  // attribute 'value' for 'attr'. Returns OK iff the attribute is given by the
  // function's definition.
  template <typename T>
  Status GetAttr(const Node& node, const string& attr, T* value) const;

  // Returns a proto representation of the state of this function library.
  FunctionDefLibrary ToProto() const;

  size_t num_functions() const { return function_defs_.size(); }

  const OpRegistryInterface* default_registry() const {
    return default_registry_;
  }

 private:
  // Shape inference for functions is handled separately by ShapeRefiner.

  struct FunctionDefAndOpRegistration {
    FunctionDefAndOpRegistration(const FunctionDef& fdef_in);

    FunctionDef fdef;
    OpRegistrationData op_registration_data;
  };

  // Same as AddFunctionDef/AddGradientDef except these methods set
  // `added` to true if the `fdef`/`grad` were actually added to this.
  Status AddFunctionDefHelper(const FunctionDef& fdef, bool* added);
  Status AddGradientDefHelper(const GradientDef& grad, bool* added);

  const OpRegistryInterface* const default_registry_;
  gtl::FlatMap<string, std::unique_ptr<FunctionDefAndOpRegistration>>
      function_defs_;
  gtl::FlatMap<string, string> func_grad_;

  // Helper function for GetAttr. Returns the FunctionDef* to get the
  // attr from.
  const FunctionDef* GetAttrImpl(const NodeDef& ndef) const;

  // Remove all functions in `funcs` and all gradients of
  // functions in `funcs_with_grads` from this library.
  void Remove(const std::vector<string>& funcs,
              const std::vector<string>& funcs_with_grads);
};

// Forward declare. Defined in common_runtime/function.h
struct FunctionBody;

// Forward declare. Defined in common_runtime/device.h
class Device;

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
    string target;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // If non-null, the runtime will use `overlay_lib` to resolve
    // function(s) named in `function_name` and `attrs`. Otherwise,
    // the runtime will use its internal library.
    // NOTE(mrry): If provided, all functions defined in `overlay_lib`
    // must be self-contained, and cannot refer to functions defined
    // in other libraries.
    // TODO(mrry): Provide a mechanism for sharing core functions
    // between a set of libraries (e.g. by allowing a
    // `FunctionLibraryDefinition` to store an `outer_scope` pointer
    // and implementing name resolution across libraries).
    const FunctionLibraryDefinition* overlay_lib = nullptr;

    // This interface is EXPERIMENTAL and subject to change.
    //
    // If non-empty, the runtime will use `state_handle` to identify
    // cached state related the instantiated function. Two functions
    // of the same name and attrs, instantiated with the same
    // `state_handle` will have the same handle and share the same
    // state (in stateful kernels); and two functions with different
    // values for `state_handle` will have independent state.
    string state_handle;
  };
  typedef uint64 Handle;
  virtual Status Instantiate(const string& function_name, AttrSlice attrs,
                             const InstantiateOptions& options,
                             Handle* handle) = 0;
  Status Instantiate(const string& function_name, AttrSlice attrs,
                     Handle* handle) {
    return Instantiate(function_name, attrs, {}, handle);
  }

  // Releases state associated with the handle.
  virtual Status ReleaseHandle(Handle handle) = 0;

  // Returns the function body for the instantiated function given its
  // handle 'h'. Returns nullptr if "h" is not found.
  //
  // *this keeps the ownership of the returned object, which remains alive
  // as long as *this.
  virtual const FunctionBody* GetFunctionBody(Handle h) = 0;

  // Asynchronously invokes the instantiated function identified by
  // "handle".
  //
  // If function execution succeeds, "done" is called with OK and
  // "*rets" is filled with the function's return values. Otheriwse,
  // "done" is called with an error status.
  //
  // Does not take ownership of "rets".
  // In the cross-process scenario, runner isn't used for making the Async
  // RPC calls.
  struct Options {
    // The id of the step that is calling this function.
    int64 step_id = 0;
    Rendezvous* rendezvous = nullptr;
    CancellationManager* cancellation_manager = nullptr;
    ScopedStepContainer* step_container = nullptr;
    StepStatsCollector* stats_collector = nullptr;

    std::function<void(std::function<void()>)>* runner = nullptr;

    // Parameters for remote function execution.
    bool remote_execution = false;
    string source_device = "";  // Fully specified device name.

    // Allocator attributes specifying where the args are / rets should be put.
    // These should either be {} or match the length of args / retvals. If {},
    // the default allocator attributes will be assumed for all args / retvals.
    std::vector<AllocatorAttributes> args_alloc_attrs;
    std::vector<AllocatorAttributes> rets_alloc_attrs;

    // If true, we create a new IntraProcessRendezvous, else use the existing
    // one.
    bool create_rendezvous = false;
  };
  typedef std::function<void(const Status&)> DoneCallback;
  virtual void Run(const Options& opts, Handle handle,
                   gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                   DoneCallback done) = 0;
  virtual void Run(const Options& opts, Handle handle,
                   CallFrameInterface* call_frame, DoneCallback done) = 0;

  // Creates a "kernel" for the given node def "ndef".
  //
  // If succeeds, returns OK and the caller takes the ownership of the
  // returned "*kernel". Otherwise, returns an error.
  virtual Status CreateKernel(const NodeDef& ndef, OpKernel** kernel) = 0;

  // Returns true iff the function named `function_name` is stateful.
  // NOTE(mrry): This method assumes that the runtime is associated with a
  // default function library, and looks up `function_name` in that library.
  // It does not support overlay libraries.
  virtual bool IsStateful(const string& function_name) = 0;

  // Returns the device on which the function executes.
  virtual Device* device() = 0;

  // Returns the function library definition that backs this runtime.
  // NOTE(mrry): The returned library definition is the default function library
  // for this runtime. The runtime may instantiate functions from separate
  // overlay libraries, which are not returned by this function.
  virtual const FunctionLibraryDefinition* GetFunctionLibraryDefinition()
      const = 0;

  // Returns the environment on which the function executes.
  virtual Env* env() = 0;

  // Returns a debug string showing the definition of the function of
  // 'handle'.
  virtual string DebugString(Handle handle) = 0;

  // Returns the graph version number.
  virtual int graph_def_version() = 0;

  typedef uint64 LocalHandle;
};

// Returns a canonicalized string for the instantiation of the
// function of the given "name", attributes "attrs", and "options".
//
// The returned string is guaranteed to be stable within one address
// space. But it may be change as the implementation
// evolves. Therefore, it should not be persisted or compared across
// address spaces.
string Canonicalize(const string& funcname, AttrSlice attrs,
                    const FunctionLibraryRuntime::InstantiateOptions& options);
inline string Canonicalize(const string& funcname, AttrSlice attrs) {
  return Canonicalize(funcname, attrs, {});
}

const FunctionLibraryRuntime::Handle kInvalidHandle = -1;
const FunctionLibraryRuntime::LocalHandle kInvalidLocalHandle = -1;
typedef std::function<Status(FunctionLibraryRuntime*, const NodeDef&,
                             std::unique_ptr<OpKernel>*)>
    CustomKernelCreator;

// Used to instantiate and run functions in a distributed system.
class DistributedFunctionLibraryRuntime {
 public:
  virtual ~DistributedFunctionLibraryRuntime() {}

  // The _target attr in attrs determines where the function is instantiated.
  virtual Status Instantiate(
      const string& function_name, const FunctionLibraryDefinition& lib_def,
      AttrSlice attrs,
      const FunctionLibraryRuntime::InstantiateOptions& options,
      FunctionLibraryRuntime::LocalHandle* handle) = 0;

  // opts.runner isn't used for execution.
  virtual void Run(const FunctionLibraryRuntime::Options& opts,
                   FunctionLibraryRuntime::LocalHandle handle,
                   gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
                   FunctionLibraryRuntime::DoneCallback done) = 0;
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
//   return Status::OK();
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

#define REGISTER_OP_GRADIENT_UNIQ(ctr, name, fn)                 \
  static bool unused_grad_##ctr = SHOULD_REGISTER_OP_GRADIENT && \
                                  ::tensorflow::gradient::RegisterOp(name, fn)

namespace gradient {
// Register a gradient creator for the "op".
typedef std::function<Status(const AttrSlice& attrs, FunctionDef*)> Creator;
bool RegisterOp(const string& op, Creator func);

// Returns OK the gradient creator for the "op" is found (may be
// nullptr if REGISTER_OP_NO_GRADIENT is used.
Status GetOpGradientCreator(const string& op, Creator* creator);
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

#endif  // TENSORFLOW_FRAMEWORK_FUNCTION_H_

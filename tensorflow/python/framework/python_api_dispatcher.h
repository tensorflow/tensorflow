/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
// Support for API dispatch at the Python level.
//
// The dispatcher is implemented in c++ for efficiency.
//
// * PythonAPIDispatcher: Class that handles dispatch for a single Python API.
//   Contains a mapping from PySignatureCheckers to dispatch targets (python
//   functions).
//
// * PySignatureChecker: Class to efficiently check whether dispatch should be
//   invoked for a given set of parameters.  Contains a collection of
//   PyTypeCheckers.
//
// * PyTypeChecker: Class to efficiently check whether a Python value matches
//   a type annotation.  Three subclasses (PyInstanceChecker, PyListChecker,
//   and PyUnionChecker) handle the different kinds of type annotation.

#ifndef TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_DISPATCHER_H_
#define TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_DISPATCHER_H_

#include <Python.h>

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/function_parameter_canonicalizer.h"

namespace tensorflow {

namespace py_dispatch {

class PyTypeChecker;
class PySignatureChecker;

// Dispatcher for a single TensorFlow Python API (e.g. `tf.add` or `tf.concat`).
//
// A separate `PythonAPIDispatcher` object is created for each API, and handles
// dispatch for that API.  The `Register` method can be used to add new
// "dispatch targets", which override the default behavior of the API when it
// is called with parameters matching a given signature.  The `Dispatch` method
// checks if any registered target matches parameters, and if so, then calls
// that target.
//
// This class is *not* thread-safe.  It is assumed that the Python Global
// Interpreter Lock (GIL) will be held when any method is called.
class PythonAPIDispatcher {
  // TODO(b/196369143) Add benchmarking for this class.
 public:
  // Creates a new PythonAPIDispatcher for the named API.
  //
  // Args:
  //   api_name: The name of the API (used for error messages).
  //   arg_names: The argument names (used for parameter canonicalization).
  //   defaults: The argument defaults, as returned by `inspect.getargspec`
  //       (used for parameter canonicalization).
  PythonAPIDispatcher(const std::string& api_name,
                      absl::Span<const char*> arg_names,
                      absl::Span<PyObject*> defaults);

  // Registers a new dispatch target for this dispatcher.  If the API is
  // called with parameters that match `signature_checker`, then
  // `dispatch_target` will be called instead of the default API implementation.
  void Register(PySignatureChecker signature_checker,
                PyObject* dispatch_target);

  // Performs dispatch with the given set of parameters.
  //
  // * If a single target matches the parameters, then that target is called.
  // * If multiple targets match the parameters, then an exception is raised.
  // * If no targets match the parameters, then returns `Py_NotImplemented`.
  //
  // On error, returns nullptr and sets a Python exception.
  Safe_PyObjectPtr Dispatch(PyObject* args, PyObject* kwargs);

  // Remove a dispatch target from this dispatcher.  If the target was
  // registered with multiple signatures, then all entries will be removed.
  // (This method is primarily intended for regression tests.)
  void Unregister(PyObject* func);

  std::string DebugString() const;

 private:
  // Name of the API.
  std::string api_name_;

  // Mapping from signature checkers to dispatch targets.
  std::vector<std::pair<PySignatureChecker, Safe_PyObjectPtr>> targets_;

  // Parameter canonicalizer.
  FunctionParameterCanonicalizer canonicalizer_;

  // Target storage for canonicalization.  (Note: for efficiency, `Dispatch`
  // writes to this pre-allocated storage, rather than allocating new storage
  // each time it is called.)
  std::vector<PyObject*> canonicalized_args_storage_;
  absl::Span<PyObject*> canonicalized_args_span_;
};

// Registers a type for use with dispatch.  Dispatch will only occur if at least
// one parameter value matches an annotation corresponding to a registered
// dispatchable type.
//
// Returns true on success; or sets a Python exception and returns false
// on error.
//
// Must be called before any PyInstanceChecker object is created from py_class.
//
// (Note: the CompositeTensor class is automatically registered for dispatch,
// so you do not need to use this method for any class that is a subclass of
// CompositeTensor or ExtensionType.)
bool RegisterDispatchableType(PyObject* py_class);

// Class used by dispatch to check if parameters' values match a signature.
//
// Currently only supports checking parameters with kind POSITIONAL_ONLY or
// POSITIONAL_OR_KEYWORD.  (Does not support checking parameters with kind
// VAR_POSITIONAL, VAR_KEYWORD, or KEYWORD_ONLY.)
class PySignatureChecker {
 public:
  // A parameter index and a TypeChecker for the parameter at that index.
  using ParamChecker = std::pair<int, std::shared_ptr<PyTypeChecker>>;

  // Constructs a signature checker that will check the specified positional
  // parameters.
  explicit PySignatureChecker(std::vector<ParamChecker> parameter_checkers);

  // Returns true if the given canonicalized arguments match this signature
  // checker.
  bool CheckCanonicalizedArgs(absl::Span<PyObject*> canon_args) const;

  std::string DebugString() const;

 private:
  // Type checkers for individual parameters.  Only annotated parameters will
  // be checked.  This list is sorted to perform less expensive checks first.
  // E.g., we check simple values before list values.
  std::vector<ParamChecker> positional_parameter_checkers_;
};

// Abstract base class that checks if a Python value matches a type annotation.
//
// Subclasses of PyTypeChecker are defined for different annotations (List,
// Union, etc). Currently, we support the minimum set of type checkers that are
// required for CompositeTensor dispatch -- namely, `List`, `Union`, and simple
// types (`IsInstance`).  Support for additional annotations may be added in the
// future.
class PyTypeChecker {
 public:
  using PyTypeChecker_ptr = std::shared_ptr<PyTypeChecker>;
  PyTypeChecker() = default;
  PyTypeChecker(const PyTypeChecker&) = delete;
  PyTypeChecker(PyTypeChecker&&) = delete;
  virtual ~PyTypeChecker() {}

  // Enumeration used to indicate whether a Python value matches a type
  // annotation. MATCH and NO_MATCH simply indicate whether a value matches the
  // annotation.
  //
  // MATCH_DISPATCHABLE indicates that a value matches the annotation, and
  // additionally that the value (or one of its nested values) matched a type
  // that has been registered for dispatch. This is important information
  // because we only want to perform dispatch if at least one such value
  // matches. Otherwise, we would end up using dispatch in undesirable cases.
  // Examples:
  //
  // @tf.dispatch_for(tf.concat)(x=List[MyType])
  //
  //    We should not dispatch to `my_concat` when the user calls
  //    `tf.concat([])` (even though it's technically true that the empty
  //    list satisfies the type annotation `List[MyType]`).
  //
  // @tf.dispatch_for(tf.add)(x=Union[MyType, Tensor], y=Union[MyType, Tensor])
  //
  //   We should not dispatch to `my_add` when the user calls
  //   `tf.add(tf.constant(1), tf.constant(2))` (even though this technically
  //   matches the annotated types).
  enum class MatchType { NO_MATCH, MATCH, MATCH_DISPATCHABLE };

  // Returns a value indicating how this type checker matched with the given
  // value.
  virtual MatchType Check(PyObject* value) = 0;

  // Approximate cost of calling this type checker, so we can perform less
  // expensive checks first.  (E.g., checking if every element in a list has a
  // given type is more costly than checking a single value.)
  virtual int cost() const = 0;

  virtual std::string DebugString() const = 0;
};

// PyTypeChecker that checks if a value is an instance of a given Python type.
class PyInstanceChecker : public PyTypeChecker {
 public:
  explicit PyInstanceChecker(const std::vector<PyObject*>& py_classes);
  ~PyInstanceChecker() override;
  MatchType Check(PyObject* value) override;
  int cost() const override;
  std::string DebugString() const override;

  // Size of the cache (for regression testing).
  size_t cache_size() const { return py_class_cache_.size(); }

 private:
  // Python class to check values against.
  std::vector<Safe_PyObjectPtr> py_classes_;

  // Cache to avoid having to call PyObject_IsInstance.  Note: we rely on the
  // Python GIL (global interpreter lock) to avoid concurrent writes to this
  // cache, since `Check()` is always called from Python (via pybind11).
  absl::flat_hash_map<PyTypeObject*, MatchType> py_class_cache_;

  // Maximum cache size.  In typical user programs, the cache will never become
  // full, but we use a maximum size in case the user creates types dynamically,
  // to avoid having an unbounded number of items in the cache.
  // TODO(b/194903203) Consider switching to an LRU cache.
  static constexpr int kMaxItemsInCache = 1024;
};

// PyTypeChecker that checks if a value is a list whose elements all match a
// nested PyTypeChecker.
class PyListChecker : public PyTypeChecker {
 public:
  explicit PyListChecker(PyTypeChecker_ptr element_type)
      : element_type_(element_type) {}
  MatchType Check(PyObject* value) override;
  int cost() const override;
  std::string DebugString() const override;

 private:
  PyTypeChecker_ptr element_type_;
};

// PyTypeChecker that checks if a value matches at least one nested
// PyTypeChecker.
class PyUnionChecker : public PyTypeChecker {
 public:
  explicit PyUnionChecker(std::vector<PyTypeChecker_ptr> options)
      : options_(options) {}
  MatchType Check(PyObject* value) override;
  int cost() const override;
  std::string DebugString() const override;

 private:
  std::vector<PyTypeChecker_ptr> options_;
};

}  // namespace py_dispatch
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_FRAMEWORK_PYTHON_API_DISPATCHER_H_

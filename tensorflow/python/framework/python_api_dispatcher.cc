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

#include "tensorflow/python/framework/python_api_dispatcher.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace tensorflow {
namespace py_dispatch {

namespace {

PyObject* ImportTypeFromModule(const char* module_name, const char* type_name) {
  static PyObject* given_type = [module_name, type_name]() {
    PyObject* module = PyImport_ImportModule(module_name);
    PyObject* attr =
        module ? PyObject_GetAttrString(module, type_name) : nullptr;
    if (attr == nullptr) {
      PyErr_WriteUnraisable(nullptr);
      PyErr_Clear();
    }
    if (module) Py_DECREF(module);
    return attr;
  }();
  return given_type;
}

struct RegisteredDispatchableTypes {
  absl::Mutex mu;
  std::vector<Safe_PyObjectPtr> types;
  uint64_t version = 0;
};

RegisteredDispatchableTypes& GetRegisteredDispatchableTypes() {
  static auto* registered_dispatchable_types =
      new RegisteredDispatchableTypes();

  // Keep Python import and reference-count operations outside registry.mu.
  static PyObject* composite_tensor = ImportTypeFromModule(
      "tensorflow.python.framework.composite_tensor",
      "CompositeTensor");

  {
    absl::MutexLock lock(&registered_dispatchable_types->mu);
    if (registered_dispatchable_types->types.empty()) {
      if (composite_tensor != nullptr) {
        Py_INCREF(composite_tensor);
        registered_dispatchable_types->types.push_back(
            Safe_PyObjectPtr(composite_tensor));
        ++registered_dispatchable_types->version;
      }
    }
  }

  return *registered_dispatchable_types;
}

struct RegisteredDispatchableTypesSnapshot {
  std::vector<Safe_PyObjectPtr> types;
  uint64_t version;
};

RegisteredDispatchableTypesSnapshot GetRegisteredDispatchableTypesSnapshot() {
  auto& registry = GetRegisteredDispatchableTypes();

  RegisteredDispatchableTypesSnapshot snapshot;
  {
    absl::MutexLock lock(&registry.mu);
    snapshot.version = registry.version;
    snapshot.types.reserve(registry.types.size());

    for (const auto& registered_type : registry.types) {
      PyObject* obj = registered_type.get();
      Py_INCREF(obj);
      snapshot.types.emplace_back(obj);
    }
  }

  return snapshot;
}

// Returns true if `py_class` is a registered dispatchable type.
bool IsRegisteredDispatchableType(
    PyObject* py_class, const std::vector<Safe_PyObjectPtr>& registered_types) {
  DCheckPyGilState();

  for (const auto& registered_type : registered_types) {
    int result = PyObject_IsSubclass(py_class, registered_type.get());
    if (result > 0) return true;
    if (result < 0) PyErr_Clear();
  }

  return false;
}

// Returns true if `py_class` is a registered dispatchable type.
bool IsRegisteredDispatchableType(PyObject* py_class) {
  auto snapshot = GetRegisteredDispatchableTypesSnapshot();
  return IsRegisteredDispatchableType(py_class, snapshot.types);
}

// Raises an exception indicating that multiple dispatch targets matched.
Safe_PyObjectPtr RaiseDispatchConflictError(const std::string& api_name,
                                            PyObject* selected,
                                            PyObject* target) {
  Safe_PyObjectPtr s1(PyObject_Str(selected));
  Safe_PyObjectPtr s2(PyObject_Str(target));
  PyErr_SetString(PyExc_ValueError,
                  absl::StrCat("Multiple dispatch targets that were "
                               "registered with tf.dispatch_for (",
                               s1 ? PyUnicode_AsUTF8(s1.get()) : "?", " and ",
                               s2 ? PyUnicode_AsUTF8(s2.get()) : "?",
                               ") match the arguments to ", api_name)
                      .c_str());
  return nullptr;
}

}  // namespace

bool RegisterDispatchableType(PyObject* py_class) {
  DCheckPyGilState();
  if (!PyType_Check(py_class)) {
    PyErr_SetString(
        PyExc_ValueError,
        absl::StrCat("Expected a type object; got object with type ",
                     py_class->ob_type->tp_name)
            .c_str());
    return false;
  }

  auto& registry = GetRegisteredDispatchableTypes();

  while (true) {
    auto snapshot = GetRegisteredDispatchableTypesSnapshot();

    if (IsRegisteredDispatchableType(py_class, snapshot.types)) {
      Safe_PyObjectPtr s(PyObject_Str(py_class));
      PyErr_SetString(PyExc_ValueError,
                      absl::StrCat("Type ", s ? PyUnicode_AsUTF8(s.get()) : "?",
                                   " (or one of its bases classes) has "
                                   "already been registered")
                          .c_str());
      return false;
    }

    // Own the new reference before entering the critical section.
    Py_INCREF(py_class);
    Safe_PyObjectPtr owned_py_class(py_class);

    {
      absl::MutexLock lock(&registry.mu);

      // A concurrent registration happened after our snapshot. Retry against
      // the new registry state rather than committing a stale decision.
      if (registry.version != snapshot.version) {
        continue;
      }

      registry.types.push_back(std::move(owned_py_class));
      ++registry.version;
      return true;
    }
  }
}

PythonAPIDispatcher::PythonAPIDispatcher(const std::string& api_name,
                                         absl::Span<const char*> arg_names,
                                         absl::Span<PyObject*> defaults)
    : api_name_(api_name),
      targets_mu_(std::make_unique<absl::Mutex>()),
      canonicalizer_(arg_names, defaults) {}

void PythonAPIDispatcher::Register(PySignatureChecker signature_checker,
                                   PyObject* dispatch_target) {
  DCheckPyGilState();
  Py_INCREF(dispatch_target);
  Safe_PyObjectPtr owned_target(dispatch_target);

  absl::MutexLock lock(targets_mu_.get());
  targets_.emplace_back(std::move(signature_checker), std::move(owned_target));
}

Safe_PyObjectPtr PythonAPIDispatcher::Dispatch(PyObject* args,
                                               PyObject* kwargs) {
  DCheckPyGilState();
  if (kwargs == Py_None) {
    kwargs = nullptr;
  }

#ifdef Py_GIL_DISABLED
  // Keep one stable kwargs snapshot for both signature matching and
  // dispatch target execution.
  Safe_PyObjectPtr kwargs_snapshot;
  if (kwargs != nullptr) {
    kwargs_snapshot = make_safe(PyDict_Copy(kwargs));
    if (!kwargs_snapshot) {
      return nullptr;
    }
    kwargs = kwargs_snapshot.get();
  }
#endif  // Py_GIL_DISABLED

  // Canonicalize args (so we don't need to deal with kwargs). Keep strong
  // references alive for the entire dispatch operation.
  absl::InlinedVector<Safe_PyObjectPtr, 8> canonicalized_args_storage(
      canonicalizer_.GetArgSize());

  if (!canonicalizer_.Canonicalize(
          args, kwargs, absl::MakeSpan(canonicalized_args_storage))) {
    return nullptr;
  }

  absl::InlinedVector<PyObject*, 8> canonicalized_args_raw;
  canonicalized_args_raw.reserve(canonicalized_args_storage.size());
  for (const auto& arg : canonicalized_args_storage) {
    canonicalized_args_raw.push_back(arg.get());
  }

  absl::Span<PyObject*> canonicalized_args_span(canonicalized_args_raw.data(),
                                                canonicalized_args_raw.size());

  // Make a copy of targets to avoid iterator invalidation if
  // Register/Unregister are called concurrently or re-entrantly during
  // CheckCanonicalizedArgs. Do not hold targets_mu_ while calling Python.
  std::vector<std::pair<PySignatureChecker, Safe_PyObjectPtr>> targets_snapshot;
  {
    absl::MutexLock lock(targets_mu_.get());
    targets_snapshot.reserve(targets_.size());
    for (const auto& target : targets_) {
      PyObject* obj = target.second.get();
      Py_INCREF(obj);
      targets_snapshot.emplace_back(target.first, make_safe(obj));
    }
  }

  PyObject* selected = nullptr;
  for (auto& target : targets_snapshot) {
    if (target.first.CheckCanonicalizedArgs(canonicalized_args_span)) {
      if (selected && selected != target.second.get()) {
        return RaiseDispatchConflictError(api_name_, selected,
                                          target.second.get());
      }
      selected = target.second.get();
    }
  }
  if (selected) {
    return Safe_PyObjectPtr(PyObject_Call(selected, args, kwargs));
  } else {
    Py_INCREF(Py_NotImplemented);
    return Safe_PyObjectPtr(Py_NotImplemented);
  }
}

// TODO(b/194903203) Raise an error if `func` is not registered.
void PythonAPIDispatcher::Unregister(PyObject* func) {
  DCheckPyGilState();
  using DispatchTargetPair = std::pair<PySignatureChecker, Safe_PyObjectPtr>;

  // Keep removed references alive until after targets_mu_ is released, since
  // DECREF may run arbitrary Python finalization code.
  std::vector<DispatchTargetPair> removed;
  {
    absl::MutexLock lock(targets_mu_.get());

    std::vector<DispatchTargetPair> kept;
    kept.reserve(targets_.size());
    removed.reserve(targets_.size());

    for (auto& target : targets_) {
      if (target.second.get() == func) {
        removed.emplace_back(std::move(target));
      } else {
        kept.emplace_back(std::move(target));
      }
    }

    targets_ = std::move(kept);
  }
}

std::string PythonAPIDispatcher::DebugString() const {
  DCheckPyGilState();
  std::string out = absl::StrCat("<Dispatch(", api_name_, "): ");

  std::vector<std::pair<PySignatureChecker, Safe_PyObjectPtr>> targets_snapshot;
  {
    absl::MutexLock lock(targets_mu_.get());
    targets_snapshot.reserve(targets_.size());
    for (const auto& target : targets_) {
      PyObject* obj = target.second.get();
      Py_INCREF(obj);
      targets_snapshot.emplace_back(target.first, make_safe(obj));
    }
  }

  const char* sep = "";
  for (const auto& target : targets_snapshot) {
    Safe_PyObjectPtr target_str(PyObject_Str(target.second.get()));
    absl::StrAppend(&out, sep, target.first.DebugString(), " -> ",
                    target_str ? PyUnicode_AsUTF8(target_str.get()) : "?");
    sep = ", ";
  }
  absl::StrAppend(&out, ">");
  return out;
}

PySignatureChecker::PySignatureChecker(
    std::vector<ParamChecker> parameter_checkers)
    : positional_parameter_checkers_(std::move(parameter_checkers)) {
  // Check less expensive parameters first, preserving argument order for equal
  // costs.
  std::stable_sort(positional_parameter_checkers_.begin(),
                   positional_parameter_checkers_.end(),
                   [](const ParamChecker& a, const ParamChecker& b) {
                     return a.second->cost() < b.second->cost();
                   });
}

bool PySignatureChecker::CheckCanonicalizedArgs(
    absl::Span<PyObject*> canon_args) const {
  DCheckPyGilState();
  bool matched_dispatchable_type = false;
  for (auto& c : positional_parameter_checkers_) {
    int index = c.first;
    auto& param_checker = c.second;
    if (index >= canon_args.size()) {
      return false;
    }
    PyObject* arg = canon_args[index];
    if (arg == nullptr) {
      return false;
    }
    switch (param_checker->Check(arg)) {
      case PyTypeChecker::MatchType::NO_MATCH:
        return false;
      case PyTypeChecker::MatchType::MATCH_DISPATCHABLE:
        matched_dispatchable_type = true;
        break;
      case PyTypeChecker::MatchType::MATCH:
        break;
    }
  }
  return matched_dispatchable_type;
}

std::string PySignatureChecker::DebugString() const {
  return absl::StrJoin(positional_parameter_checkers_, ", ",
                       [](std::string* out, ParamChecker p) {
                         absl::StrAppend(out, "args[", p.first,
                                         "]:", p.second->DebugString());
                       });
}

PyInstanceChecker::PyInstanceChecker(const std::vector<PyObject*>& py_classes)
    : py_class_cache_mu_(std::make_unique<absl::Mutex>()) {
  DCheckPyGilState();
  py_classes_.reserve(py_classes.size());
  for (PyObject* py_class : py_classes) {
    py_classes_.emplace_back(py_class);
    Py_INCREF(py_class);
  }
}

PyInstanceChecker::~PyInstanceChecker() {
  DCheckPyGilState();

  std::vector<PyTypeObject*> cached_types;
  {
    absl::MutexLock lock(py_class_cache_mu_.get());
    cached_types.reserve(py_class_cache_.size());
    for (const auto& pair : py_class_cache_) {
      cached_types.push_back(pair.first);
    }
    py_class_cache_.clear();
  }

  for (PyTypeObject* type : cached_types) {
    Py_DECREF(type);
  }
}

PyTypeChecker::MatchType PyInstanceChecker::Check(PyObject* value) {
  DCheckPyGilState();
  auto* type = Py_TYPE(value);

  {
    absl::MutexLock lock(py_class_cache_mu_.get());
    auto it = py_class_cache_.find(type);
    if (it != py_class_cache_.end()) {
      return it->second;
    }
  }

  MatchType result = MatchType::NO_MATCH;
  for (const auto& py_class : py_classes_) {
    int is_instance = PyObject_IsInstance(value, py_class.get());
    if (is_instance == 1) {
      if (IsRegisteredDispatchableType(py_class.get())) {
        result = MatchType::MATCH_DISPATCHABLE;
        break;
      } else {
        result = MatchType::MATCH;
      }
    } else if (is_instance < 0) {
      PyErr_Clear();
      return MatchType::NO_MATCH;
    }
  }

  {
    absl::MutexLock lock(py_class_cache_mu_.get());

    auto existing = py_class_cache_.find(type);
    if (existing != py_class_cache_.end()) {
      return existing->second;
    }

    if (py_class_cache_.size() < kMaxItemsInCache) {
      Py_INCREF(type);
      py_class_cache_.insert({type, result});
    }
  }

  return result;
}

int PyInstanceChecker::cost() const { return py_classes_.size(); }

std::string PyInstanceChecker::DebugString() const {
  DCheckPyGilState();
  std::vector<const char*> type_names;
  type_names.reserve(py_classes_.size());
  for (const auto& py_class : py_classes_) {
    type_names.push_back(
        reinterpret_cast<PyTypeObject*>(py_class.get())->tp_name);
  }
  return absl::StrJoin(
      py_classes_, ", ", [](std::string* out, const Safe_PyObjectPtr& v) {
        out->append(reinterpret_cast<PyTypeObject*>(v.get())->tp_name);
      });
}

PyTypeChecker::MatchType PyListChecker::Check(PyObject* value) {
  DCheckPyGilState();
  if (!(PyList_Check(value) || PyTuple_Check(value))) {
    return MatchType::NO_MATCH;
  }

  Safe_PyObjectPtr seq(PySequence_Fast(value, ""));
  if (!seq) {
    PyErr_Clear();
    return MatchType::NO_MATCH;
  }

#ifdef Py_GIL_DISABLED
  // Snapshot mutable list elements under the container lock, keeping strong
  // references alive after the critical section is released.
  std::vector<Safe_PyObjectPtr> elements;

  if (PyList_Check(seq.get())) {
    const Py_ssize_t size = PyList_GET_SIZE(seq.get());
    elements.reserve(size);

    Py_BEGIN_CRITICAL_SECTION(seq.get());

    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject* item = PyList_GET_ITEM(seq.get(), i);
      Py_INCREF(item);
      elements.emplace_back(item);
    }

    Py_END_CRITICAL_SECTION();
  } else {
    const Py_ssize_t size = PyTuple_GET_SIZE(seq.get());
    elements.reserve(size);

    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject* item = PyTuple_GET_ITEM(seq.get(), i);
      Py_INCREF(item);
      elements.emplace_back(item);
    }
  }

  MatchType result = MatchType::MATCH;
  for (const auto& item : elements) {
    switch (element_type_->Check(item.get())) {
      case MatchType::NO_MATCH:
        return MatchType::NO_MATCH;
      case MatchType::MATCH_DISPATCHABLE:
        result = MatchType::MATCH_DISPATCHABLE;
        break;
      case MatchType::MATCH:
        break;
    }
  }
  return result;

#else

  MatchType result = MatchType::MATCH;
  for (int i = 0; i < PySequence_Fast_GET_SIZE(seq.get()); ++i) {
    switch (element_type_->Check(PySequence_Fast_GET_ITEM(seq.get(), i))) {
      case MatchType::NO_MATCH:
        return MatchType::NO_MATCH;
      case MatchType::MATCH_DISPATCHABLE:
        result = MatchType::MATCH_DISPATCHABLE;
        break;
      case MatchType::MATCH:
        break;
    }
  }
  return result;
#endif  // Py_GIL_DISABLED
}

int PyListChecker::cost() const { return 10 * element_type_->cost(); }

std::string PyListChecker::DebugString() const {
  return absl::StrCat("List[", element_type_->DebugString(), "]");
}

PyTypeChecker::MatchType PyUnionChecker::Check(PyObject* value) {
  DCheckPyGilState();
  MatchType result = MatchType::NO_MATCH;
  for (auto& type_option : options_) {
    switch (type_option->Check(value)) {
      case MatchType::MATCH:
        result = MatchType::MATCH;
        break;
      case MatchType::MATCH_DISPATCHABLE:
        return MatchType::MATCH_DISPATCHABLE;
      case MatchType::NO_MATCH:
        break;
    }
  }
  return result;
}

int PyUnionChecker::cost() const {
  int cost = 1;
  for (auto& type_option : options_) {
    cost += type_option->cost();
  }
  return cost;
}

std::string PyUnionChecker::DebugString() const {
  return absl::StrCat("Union[",
                      absl::StrJoin(options_, ", ",
                                    [](std::string* out, PyTypeChecker_ptr v) {
                                      out->append(v->DebugString());
                                    }),
                      "]");
}

}  // namespace py_dispatch
}  // namespace tensorflow

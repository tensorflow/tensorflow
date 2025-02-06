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

#include <set>

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

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

std::vector<Safe_PyObjectPtr>& GetRegisteredDispatchableTypes() {
  static std::vector<Safe_PyObjectPtr>* registered_dispatchable_types =
      new std::vector<Safe_PyObjectPtr>();
  if (registered_dispatchable_types->empty()) {
    static PyObject* composite_tensor = ImportTypeFromModule(
        "tensorflow.python.framework.composite_tensor",
        "CompositeTensor");
    Py_INCREF(composite_tensor);
    registered_dispatchable_types->push_back(
        Safe_PyObjectPtr(composite_tensor));
  }
  return *registered_dispatchable_types;
}

// Returns true if `py_class` is a registered dispatchable type.
bool IsRegisteredDispatchableType(PyObject* py_class) {
  DCheckPyGilState();
  for (const auto& registered_type : GetRegisteredDispatchableTypes()) {
    int result = PyObject_IsSubclass(py_class, registered_type.get());
    if (result > 0) return true;
    if (result < 0) PyErr_Clear();
  }
  return false;
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
  if (IsRegisteredDispatchableType(py_class)) {
    Safe_PyObjectPtr s(PyObject_Str(py_class));
    PyErr_SetString(PyExc_ValueError,
                    absl::StrCat("Type ", s ? PyUnicode_AsUTF8(s.get()) : "?",
                                 " (or one of its bases clases) has "
                                 "already been registered")
                        .c_str());
    return false;
  }
  Py_INCREF(py_class);
  GetRegisteredDispatchableTypes().push_back(Safe_PyObjectPtr(py_class));
  return true;
}

PythonAPIDispatcher::PythonAPIDispatcher(const std::string& api_name,
                                         absl::Span<const char*> arg_names,
                                         absl::Span<PyObject*> defaults)
    : api_name_(api_name),
      canonicalizer_(arg_names, defaults),
      canonicalized_args_storage_(canonicalizer_.GetArgSize()),
      canonicalized_args_span_(canonicalized_args_storage_) {}

void PythonAPIDispatcher::Register(PySignatureChecker signature_checker,
                                   PyObject* dispatch_target) {
  DCheckPyGilState();
  Py_INCREF(dispatch_target);
  targets_.emplace_back(std::move(signature_checker),
                        Safe_PyObjectPtr(dispatch_target));
}

Safe_PyObjectPtr PythonAPIDispatcher::Dispatch(PyObject* args,
                                               PyObject* kwargs) {
  DCheckPyGilState();
  if (kwargs == Py_None) {
    kwargs = nullptr;
  }
  // Canonicalize args (so we don't need to deal with kwargs).
  if (!canonicalizer_.Canonicalize(args, kwargs, canonicalized_args_span_)) {
    return nullptr;
  }

  PyObject* selected = nullptr;
  for (auto& target : targets_) {
    if (target.first.CheckCanonicalizedArgs(canonicalized_args_span_)) {
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
  targets_.erase(std::remove_if(targets_.begin(), targets_.end(),
                                [func](const DispatchTargetPair& t) {
                                  return t.second.get() == func;
                                }),
                 targets_.end());
}

std::string PythonAPIDispatcher::DebugString() const {
  DCheckPyGilState();
  std::string out = absl::StrCat("<Dispatch(", api_name_, "): ");

  const char* sep = "";
  for (const auto& target : targets_) {
    Safe_PyObjectPtr target_str(PyObject_Str(target.second.get()));
    absl::StrAppend(&out, sep, target.first.DebugString(), " -> ",
                    target_str ? PyUnicode_AsUTF8(target_str.get()) : "?");
    sep = ", ";
  }
  return out;
}

PySignatureChecker::PySignatureChecker(
    std::vector<ParamChecker> parameter_checkers)
    : positional_parameter_checkers_(std::move(parameter_checkers)) {
  // Check less expensive parameters first.
  std::sort(positional_parameter_checkers_.begin(),
            positional_parameter_checkers_.end(),
            [](ParamChecker a, ParamChecker b) {
              return a.second->cost() < b.second->cost();
            });
}

bool PySignatureChecker::CheckCanonicalizedArgs(
    absl::Span<PyObject*> canon_args) const {
  bool matched_dispatchable_type = false;
  for (auto& c : positional_parameter_checkers_) {
    int index = c.first;
    auto& param_checker = c.second;
    if (index >= canon_args.size()) {
      return false;
    }
    switch (param_checker->Check(canon_args[index])) {
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

PyInstanceChecker::PyInstanceChecker(const std::vector<PyObject*>& py_classes) {
  DCheckPyGilState();
  py_classes_.reserve(py_classes.size());
  for (PyObject* py_class : py_classes) {
    py_classes_.emplace_back(py_class);
    Py_INCREF(py_class);
  }
}

PyInstanceChecker::~PyInstanceChecker() {
  DCheckPyGilState();
  for (const auto& pair : py_class_cache_) {
    Py_DECREF(pair.first);
  }
}

PyTypeChecker::MatchType PyInstanceChecker::Check(PyObject* value) {
  DCheckPyGilState();
  auto* type = Py_TYPE(value);
  auto it = py_class_cache_.find(type);
  if (it != py_class_cache_.end()) {
    return it->second;
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

  if (py_class_cache_.size() < kMaxItemsInCache) {
    Py_INCREF(type);
    auto insert_result = py_class_cache_.insert({type, result});
    if (!insert_result.second) {
      Py_DECREF(type);  // Result was added by a different thread.
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
    return MatchType::NO_MATCH;  // value is not a sequence.
  }

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
}

int PyListChecker::cost() const { return 10 * element_type_->cost(); }

std::string PyListChecker::DebugString() const {
  return absl::StrCat("List[", element_type_->DebugString(), "]");
}

PyTypeChecker::MatchType PyUnionChecker::Check(PyObject* value) {
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

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
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace tensorflow {
namespace py_dispatch {

namespace {

// Returns true if `py_class` is a subclass of tf.CompositeTensor.
bool IsSubclassOfCompositeTensor(PyObject* py_class) {
  static PyObject* composite_tensor =
      swig::GetRegisteredPyObject("CompositeTensor");
  int result = PyObject_IsSubclass(py_class, composite_tensor);
  if (result < 0) PyErr_Clear();
  return result > 0;
}

// Raises an exception indicating that multiple dispatch targets matched.
Safe_PyObjectPtr RaiseDispatchConflictError(const std::string& api_name,
                                            PyObject* selected,
                                            PyObject* target) {
  Safe_PyObjectPtr s1(PyObject_Str(selected));
  if (!s1) return nullptr;
  Safe_PyObjectPtr s2(PyObject_Str(target));
  if (!s2) return nullptr;
  PyErr_SetString(PyExc_ValueError,
                  absl::StrCat("Multiple dispatch targets that were "
                               "registered with tf.dispatch_for (",
                               PyUnicode_AsUTF8(s1.get()), " and ",
                               PyUnicode_AsUTF8(s2.get()),
                               ") match the arguments to ", api_name)
                      .c_str());
  return nullptr;
}

}  // namespace

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
  targets_.emplace_back(std::move(signature_checker), dispatch_target);
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
  std::string out = absl::StrCat("<Disptach(", api_name_, "): ");

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
  bool matched_composite_tensor = false;
  for (auto& c : positional_parameter_checkers_) {
    int index = c.first;
    auto& param_checker = c.second;
    if (index >= canon_args.size()) {
      return false;
    }
    switch (param_checker->Check(canon_args[index])) {
      case PyTypeChecker::MatchType::NO_MATCH:
        return false;
      case PyTypeChecker::MatchType::MATCH_COMPOSITE:
        matched_composite_tensor = true;
        break;
      case PyTypeChecker::MatchType::MATCH:
        break;
    }
  }
  return matched_composite_tensor;
}

std::string PySignatureChecker::DebugString() const {
  return absl::StrJoin(positional_parameter_checkers_, ", ",
                       [](std::string* out, ParamChecker p) {
                         absl::StrAppend(out, "args[", p.first,
                                         "]:", p.second->DebugString());
                       });
}

PyInstanceChecker::PyInstanceChecker(PyObject* py_class) : py_class_(py_class) {
  DCheckPyGilState();
  Py_INCREF(py_class);
  match_type_ = IsSubclassOfCompositeTensor(py_class)
                    ? MatchType::MATCH_COMPOSITE
                    : MatchType::MATCH;
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
    return it->second ? match_type_ : MatchType::NO_MATCH;
  }

  int result = PyObject_IsInstance(value, py_class_.get());
  if (result < 0) {
    PyErr_Clear();
    return MatchType::NO_MATCH;
  }

  if (py_class_cache_.size() < kMaxItemsInCache) {
    Py_INCREF(type);
    auto insert_result = py_class_cache_.insert({type, result});
    DCHECK(insert_result.second);
  }
  return result ? match_type_ : MatchType::NO_MATCH;
}

std::string PyInstanceChecker::DebugString() const {
  DCheckPyGilState();
  return reinterpret_cast<PyTypeObject*>(py_class_.get())->tp_name;
}

PyTypeChecker::MatchType PyListChecker::Check(PyObject* value) {
  DCheckPyGilState();
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
      case MatchType::MATCH_COMPOSITE:
        result = MatchType::MATCH_COMPOSITE;
        break;
      case MatchType::MATCH:
        break;
    }
  }
  return result;
}

int PyListChecker::cost() { return 10 * element_type_->cost(); }

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
      case MatchType::MATCH_COMPOSITE:
        return MatchType::MATCH_COMPOSITE;
      case MatchType::NO_MATCH:
        break;
    }
  }
  return result;
}

int PyUnionChecker::cost() {
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

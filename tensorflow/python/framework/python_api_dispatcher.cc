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

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace tensorflow {

using ParamInfo = PythonAPIDispatcher::ParamInfo;

// List of python types to check for dispatch.  In most cases, this vector
// will have size zero or one; and sizes greater than 3 should be rare.
using TypeList = absl::InlinedVector<PyTypeObject*, 3>;

namespace {

// Returns the __tf__dispatch__ attribute of `obj`.
Safe_PyObjectPtr GetAttr_TFDispatch(PyObject* obj) {
#if PY_MAJOR_VERSION < 3
  // Python 2.x:
  static PyObject* attr = PyString_InternFromString("__tf_dispatch__");
#else
  // Python 3.x:
  static PyObject* attr = PyUnicode_InternFromString("__tf_dispatch__");
#endif
  return Safe_PyObjectPtr(PyObject_GetAttr(obj, attr));
}

// Searches `params` for dispatchable types, and returns a vector of borrowed
// references to those types.  Removes consecutive duplicates (i.e., if a
// dispatchable parameter has the same type as the previously encountered
// dispatcahble parameter, then it's type is not added again), so the result
// will usually have a length of zero or one; but in the general case, it may be
// longer, and may contain (nonconsecutive) duplicates.
//
// Assumes that `params` is a tuple, and that all parameter indices in
// `dispatch_params` and `dispatch_list_params` are valid.
TypeList FindDispatchTypes(PyObject* params,
                           const std::vector<ParamInfo>& dispatchable_params) {
  TypeList dispatch_types;
  for (const auto& param : dispatchable_params) {
    DCHECK_GE(param.index, 0);
    DCHECK_LT(param.index, PyTuple_GET_SIZE(params));
    PyObject* value = PyTuple_GET_ITEM(params, param.index);
    if (param.is_list) {
      DCHECK(PyList_Check(value));
      Py_ssize_t num_items = PyList_Size(value);
      for (Py_ssize_t i = 0; i < num_items; ++i) {
        PyObject* item = PyList_GET_ITEM(value, i);
        // TODO(b/164980194) Consider changing IsDispatchable to not use a
        // cache.  This may impact efficiency (needs to be measured), but would
        // allow us to support monkey-patching classes to be dispatchable.
        if (swig::IsDispatchable(item)) {
          if (dispatch_types.empty() ||
              value->ob_type != dispatch_types.back()) {
            dispatch_types.push_back(item->ob_type);
          }
        }
      }
    } else {
      if (swig::IsDispatchable(value)) {
        if (dispatch_types.empty() || value->ob_type != dispatch_types.back()) {
          dispatch_types.push_back(value->ob_type);
        }
      }
    }
  }

  return dispatch_types;
}

// Removes duplicates from `dispatch_types`, and moves any subtypes to
// before their supertypes.  Note: this method is only called when
// `dispatch_types.size() > 1`.
void SortDispatchTypes(TypeList& dispatch_types) {
  // Remove duplicates.  Note: this is O(n^2) in the number of dispatchable
  // types, but we expect this number to be very small in almost every case
  // (usually zero, sometimes one, and rarely larger than two).
  for (int i = 0; i < dispatch_types.size() - 1; ++i) {
    if (dispatch_types[i] == nullptr) continue;
    for (int j = i + 1; j < dispatch_types.size(); ++j) {
      if (dispatch_types[i] == dispatch_types[j]) {
        dispatch_types[j] = nullptr;  // mark duplicate
      }
    }
  }
  dispatch_types.erase(
      std::remove_if(dispatch_types.begin(), dispatch_types.end(),
                     [](PyTypeObject* t) { return t == nullptr; }),
      dispatch_types.end());

  // Move subclasses before superclasses.  As above, this is O(n^2), but we
  // expect n to be small.
  TypeList sorted;
  TypeList subtypes;
  for (int i = 0; i < dispatch_types.size(); ++i) {
    if (dispatch_types[i] == nullptr) continue;
    subtypes.clear();
    for (int j = i + 1; j < dispatch_types.size(); ++j) {
      if (dispatch_types[j] == nullptr) continue;
      if (PyType_IsSubtype(dispatch_types[j], dispatch_types[i])) {
        subtypes.push_back(dispatch_types[j]);
        dispatch_types[j] = nullptr;  // mark as already added.
      }
    }
    if (!subtypes.empty()) {
      std::sort(subtypes.begin(), subtypes.end(), PyType_IsSubtype);
      sorted.insert(sorted.end(), subtypes.begin(), subtypes.end());
    }
    sorted.push_back(dispatch_types[i]);
  }
  DCHECK_EQ(dispatch_types.size(), sorted.size());
  dispatch_types.swap(sorted);
}

}  // namespace

PythonAPIDispatcher::PythonAPIDispatcher(const std::string& api_name,
                                         PyObject* api_func, int num_params,
                                         bool right_to_left)
    : api_name_(PyUnicode_FromStringAndSize(api_name.c_str(), api_name.size())),
      api_func_(api_func),
      num_params_(num_params),
      right_to_left_(right_to_left) {
  Py_INCREF(api_func);
}

bool PythonAPIDispatcher::Initialize(
    std::vector<ParamInfo> dispatchable_params) {
  dispatchable_params_.swap(dispatchable_params);
  std::sort(dispatchable_params_.begin(), dispatchable_params_.end(),
            [](const ParamInfo& a, const ParamInfo& b) -> bool {
              return a.index < b.index;
            });
  if (right_to_left_) {
    std::reverse(dispatchable_params_.begin(), dispatchable_params_.end());
  }

  for (const auto& p : dispatchable_params_) {
    if (p.index < 0 || p.index >= num_params_) {
      PyErr_SetString(
          PyExc_ValueError,
          absl::StrCat("PythonAPIDispatcher: dispatchable parameter index out ",
                       "of range: ", p.index, " not in [0, ", num_params_, ")")
              .c_str());
      return false;
    }
  }
  return true;
}

PyObject* PythonAPIDispatcher::Dispatch(PyObject* params) const {
  DCHECK(PyTuple_Check(params));

  // TODO(b/164980194) Consider removing this check, if the caller is also
  // checking/guaranteeing it (once dispatch has been integrated w/ the Python
  // API handlers).
  if (num_params_ != PyTuple_Size(params)) {
#if PY_MAJOR_VERSION < 3
    // Python 2.x:
    Safe_PyObjectPtr api_name_str(PyUnicode_AsUTF8String(api_name_.get()));
    if (!api_name_str) return nullptr;
    const char* api_name = PyString_AsString(api_name_str.get());
#else
    // Python 3.x:
    const char* api_name = PyUnicode_AsUTF8AndSize(api_name_.get(), nullptr);
#endif
    PyErr_SetString(
        PyExc_TypeError,
        absl::StrCat(api_name ? api_name : "unknown PythonAPIDispatcher",
                     " expected ", num_params_, " parameters, but got ",
                     PyTuple_Size(params))
            .c_str());
    return nullptr;
  }

  TypeList dispatch_types = FindDispatchTypes(params, dispatchable_params_);

  if (dispatch_types.empty()) {
    return Py_NotImplemented;
  }

  if (dispatch_types.size() > 1) {
    SortDispatchTypes(dispatch_types);
  }

  for (PyTypeObject* dispatch_type : dispatch_types) {
    Safe_PyObjectPtr dispatcher =
        GetAttr_TFDispatch(reinterpret_cast<PyObject*>(dispatch_type));
    if (!dispatcher) return nullptr;
    PyObject* result = PyObject_CallFunctionObjArgs(
        dispatcher.get(), api_name_.get(), api_func_.get(), params, nullptr);
    if (result != Py_NotImplemented) {
      return result;
    }
  }

  return Py_NotImplemented;
}

}  // namespace tensorflow

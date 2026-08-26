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

#include "tensorflow/python/util/function_parameter_canonicalizer.h"

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"

namespace {
inline const char* PyUnicodeAsUtf8Compat(PyObject* obj) {
#if PY_MAJOR_VERSION < 3
  return PyString_AS_STRING(obj);
#else
  return PyUnicode_AsUTF8(obj);
#endif
}

inline PyObject* PyUnicodeInternFromStringCompat(const char* str) {
#if PY_MAJOR_VERSION < 3
  return PyString_InternFromString(str);
#else
  return PyUnicode_InternFromString(str);
#endif
}

inline void PyUnicodeInternInPlaceCompat(PyObject** obj) {
#if PY_MAJOR_VERSION < 3
  PyString_InternInPlace(obj);
#else
  PyUnicode_InternInPlace(obj);
#endif
}

}  // namespace

namespace tensorflow {

FunctionParameterCanonicalizer::FunctionParameterCanonicalizer(
    absl::Span<const char*> arg_names, absl::Span<PyObject*> defaults)
    : positional_args_size_(arg_names.size() - defaults.size()) {
  DCheckPyGilState();
  DCHECK_GE(positional_args_size_, 0);

  interned_arg_names_.reserve(arg_names.size());
  for (const char* obj : arg_names)
    interned_arg_names_.emplace_back(PyUnicodeInternFromStringCompat(obj));

  DCHECK(AreInternedArgNamesUnique());

  for (PyObject* obj : defaults) Py_INCREF(obj);
  defaults_ = std::vector<Safe_PyObjectPtr>(defaults.begin(), defaults.end());
}

bool FunctionParameterCanonicalizer::Canonicalize(
    PyObject* args, PyObject* kwargs, absl::Span<Safe_PyObjectPtr> result) {
  // TODO(kkb): Closely follow `Python/ceval.c`'s logic and error handling.

  DCheckPyGilState();
  DCHECK(PyTuple_CheckExact(args));
  DCHECK(kwargs == nullptr || PyDict_CheckExact(kwargs));
  DCHECK_EQ(result.size(), interned_arg_names_.size());

  const int args_size = Py_SIZE(args);
  int remaining_positional_args_count = positional_args_size_ - args_size;

  // Check if the number of input arguments are too many.
  if (TF_PREDICT_FALSE(args_size > interned_arg_names_.size())) {
    PyErr_SetString(
        PyExc_TypeError,
        absl::StrCat("Too many arguments were given. Expected ",
                     interned_arg_names_.size(), " but got ", args_size, ".")
            .c_str());
    return false;
  }

  // Fill positional arguments with strong references.
  for (int i = 0; i < args_size; ++i) {
    PyObject* value = PyTuple_GET_ITEM(args, i);
    Py_INCREF(value);
    result[i] = Safe_PyObjectPtr(value);
  }

  // Fill default arguments with strong references.
  for (int i = std::max(positional_args_size_, args_size);
       i < interned_arg_names_.size(); ++i) {
    PyObject* value = defaults_[i - positional_args_size_].get();
    Py_INCREF(value);
    result[i] = Safe_PyObjectPtr(value);
  }

  // Fill keyword arguments.
  if (kwargs != nullptr) {
#ifdef Py_GIL_DISABLED
    // PyDict_Next() requires external synchronization in free-threaded
    // CPython. Snapshot strong references while holding the dictionary
    // critical section, then process them after releasing it.
    using OwnedKwarg = std::pair<Safe_PyObjectPtr, Safe_PyObjectPtr>;
    std::vector<OwnedKwarg> kwargs_snapshot;
    kwargs_snapshot.reserve(PyDict_Size(kwargs));

    Py_BEGIN_CRITICAL_SECTION(kwargs);

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      Py_INCREF(key);
      Py_INCREF(value);
      kwargs_snapshot.emplace_back(Safe_PyObjectPtr(key),
                                   Safe_PyObjectPtr(value));
    }

    Py_END_CRITICAL_SECTION();

    for (auto& item : kwargs_snapshot) {
      PyObject* key = item.first.get();

      std::size_t index = InternedArgNameLinearSearch(key);

      if (TF_PREDICT_FALSE(index == interned_arg_names_.size())) {
        PyObject* interned_key = key;
        Py_INCREF(interned_key);
        PyUnicodeInternInPlaceCompat(&interned_key);

        index = InternedArgNameLinearSearch(interned_key);

        if (TF_PREDICT_FALSE(index == interned_arg_names_.size())) {
          PyErr_Format(PyExc_TypeError,
                       "Got an unexpected keyword argument '%s'",
                       PyUnicodeAsUtf8Compat(interned_key));
          Py_DECREF(interned_key);
          return false;
        }

        Py_DECREF(interned_key);
      }

      if (TF_PREDICT_FALSE(index < args_size)) {
        PyErr_Format(PyExc_TypeError, "Got multiple values for argument '%s'",
                     PyUnicodeAsUtf8Compat(key));
        return false;
      }

      if (TF_PREDICT_FALSE(index < positional_args_size_))
        --remaining_positional_args_count;

      result[index] = std::move(item.second);
    }
#else
    // With the GIL enabled, preserve the original direct dictionary iteration
    // fast path while returning strong references in `result`.
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      std::size_t index = InternedArgNameLinearSearch(key);

      if (TF_PREDICT_FALSE(index == interned_arg_names_.size())) {
        Py_INCREF(key);
        PyUnicodeInternInPlaceCompat(&key);
        index = InternedArgNameLinearSearch(key);

        if (TF_PREDICT_FALSE(index == interned_arg_names_.size())) {
          PyErr_Format(PyExc_TypeError,
                       "Got an unexpected keyword argument '%s'",
                       PyUnicodeAsUtf8Compat(key));
          Py_DECREF(key);
          return false;
        }

        Py_DECREF(key);
      }

      if (TF_PREDICT_FALSE(index < args_size)) {
        PyErr_Format(PyExc_TypeError, "Got multiple values for argument '%s'",
                     PyUnicodeAsUtf8Compat(key));
        return false;
      }

      if (TF_PREDICT_FALSE(index < positional_args_size_))
        --remaining_positional_args_count;

      Py_INCREF(value);
      result[index] = Safe_PyObjectPtr(value);
    }
#endif  // Py_GIL_DISABLED
  }

  // Check if all the arguments are filled.
  // Example failure, not enough number of arguments passed: `matmul(x)`
  if (TF_PREDICT_FALSE(remaining_positional_args_count > 0)) {
    // TODO(kkb): Report what arguments are missing.
    PyErr_SetString(PyExc_TypeError, "Missing required positional argument");
    return false;
  }

  return true;
}

ABSL_MUST_USE_RESULT
ABSL_ATTRIBUTE_HOT
inline std::size_t FunctionParameterCanonicalizer::InternedArgNameLinearSearch(
    PyObject* name) {
  std::size_t result = interned_arg_names_.size();

  for (std::size_t i = 0; i < interned_arg_names_.size(); ++i)
    if (TF_PREDICT_FALSE(name == interned_arg_names_[i].get())) return i;

  return result;
}

bool FunctionParameterCanonicalizer::AreInternedArgNamesUnique() {
  absl::flat_hash_set<PyObject*> interned_arg_names_set;
  for (const Safe_PyObjectPtr& obj : interned_arg_names_)
    interned_arg_names_set.emplace(obj.get());

  return interned_arg_names_set.size() == interned_arg_names_.size();
}
}  // namespace tensorflow

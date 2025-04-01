/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/util/nest.h"

#include <Python.h>

#include <cstddef>
#include <string>

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace tensorflow {

namespace {

// Gets a string representation of the input object.
//
// Args:
//   o: a python object.
//   length: If set to negative, the whole string is returned. Otherwise, the
//       string gets clipped to 'length' in size.
//
// Returns:
//   A string representation.
std::string PyObject_ToString(PyObject* o, int length = -1) {
  auto str_o = make_safe(PyObject_Str(o));
  std::string str = PyUnicode_AsUTF8(str_o.get());
  if (length < 0 || str.size() <= length) {
    return str;
  }
  absl::string_view str_piece(str);
  return tensorflow::strings::StrCat(str_piece.substr(length), "...");
}

// Gets a list of keys from a dict or mapping type object.
//
// Args:
//   o: a dictionary or mapping type object.
//
// Returns:
//   A new reference to a list.
//
// Raises:
//   TypeError: if `o` is not a dict or mapping type object.
PyObject* GetKeysFromDictOrMapping(PyObject* o) {
  if (PyDict_Check(o)) {
    return PyDict_Keys(o);
  } else if (PyMapping_Check(o)) {
    return PyMapping_Keys(o);
  } else {
    auto* o_type = Py_TYPE(o);
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a type compatible with dict or mapping, got '",
            o_type->tp_name, "'")
            .c_str());
    return nullptr;
  }
}

}  // namespace

PyObject* FlattenDictItems(PyObject* dict) {
  if (!PyDict_Check(dict) && !swig::IsMapping(dict)) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "FlattenDictItems: 'dict' must be a dictionary or ",
                        "collection.Mapping type object, instead of '",
                        Py_TYPE(dict)->tp_name, "'.")
                        .c_str());
    return nullptr;
  }
  PyObject* flat_dictionary = PyDict_New();
  auto keys = make_safe(GetKeysFromDictOrMapping(dict));
  for (size_t i = 0; i < PyList_Size(keys.get()); ++i) {
    auto* key = PyList_GetItem(keys.get(), i);
    // We use a general approach in case 'dict' is a PyMapping type,
    // but not a PyDict type.
    auto* value = PyObject_GetItem(dict, key);
    if (swig::IsNested(key)) {
      // The dict might contain list - list pairs.
      auto flat_keys = make_safe(swig::Flatten(key, false));
      auto flat_values = make_safe(swig::Flatten(value, false));
      size_t flat_keys_sz = PyList_Size(flat_keys.get());
      size_t flat_values_sz = PyList_Size(flat_values.get());
      if (flat_keys_sz != flat_values_sz) {
        PyErr_SetString(
            PyExc_ValueError,
            tensorflow::strings::StrCat(
                "Could not flatten dictionary. Key had ", flat_keys_sz,
                " elements, but value had ", flat_values_sz,
                " elements. Key: ", PyObject_ToString(flat_keys.get()),
                ", value: ", PyObject_ToString(flat_values.get()), ".")
                .c_str());
        Py_DecRef(flat_dictionary);
        return nullptr;
      }
      for (size_t i = 0; i < flat_keys_sz; ++i) {
        auto* flat_key = PyList_GetItem(flat_keys.get(), i);
        auto* flat_value = PyList_GetItem(flat_values.get(), i);
        if (PyDict_GetItem(flat_dictionary, flat_key) != nullptr) {
          PyErr_SetString(
              PyExc_ValueError,
              tensorflow::strings::StrCat(
                  "Cannot flatten dict because this key is not unique: ",
                  PyObject_ToString(flat_key))
                  .c_str());
          Py_DecRef(flat_dictionary);
          return nullptr;
        }
        PyDict_SetItem(flat_dictionary, flat_key, flat_value);
      }
    } else {
      if (PyDict_GetItem(flat_dictionary, key) != nullptr) {
        PyErr_SetString(
            PyExc_ValueError,
            tensorflow::strings::StrCat(
                "Cannot flatten dict because this key is not unique: ",
                PyObject_ToString(key))
                .c_str());
        Py_DecRef(flat_dictionary);
        return nullptr;
      }
      PyDict_SetItem(flat_dictionary, key, value);
    }
    // Manually decrease because PyObject_GetItem() returns a new reference.
    Py_DECREF(value);
  }
  return flat_dictionary;
}

}  // namespace tensorflow

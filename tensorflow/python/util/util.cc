/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/util/util.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace swig {

namespace {

// Type object for collections.Sequence. This is set by RegisterSequenceClass.
PyObject* CollectionsSequenceType = nullptr;

bool WarnedThatSetIsNotSequence = false;

// Returns 1 if `o` is considered a sequence for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occured.
int IsSequenceHelper(PyObject* o) {
  if (PyDict_Check(o)) return true;
  if (PySet_Check(o) && !WarnedThatSetIsNotSequence) {
    LOG(WARNING) << "Sets are not currently considered sequences, "
                    "but this may change in the future, "
                    "so consider avoiding using them.";
    WarnedThatSetIsNotSequence = true;
  }
  if (CollectionsSequenceType == nullptr) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat(
            "collections.Sequence type has not been set. "
            "Please call RegisterSequenceClass before using this module")
            .c_str());
    return -1;
  }
  int is_instance = PyObject_IsInstance(o, CollectionsSequenceType);
  if (is_instance == -1) return -1;
  return static_cast<int>(is_instance != 0 && !PyBytes_Check(o) &&
#if PY_MAJOR_VERSION < 3
                          !PyString_Check(o) &&
#endif
                          !PyUnicode_Check(o));
}

bool FlattenHelper(PyObject* nested, PyObject* list) {
  // if nested is not a sequence, append itself and exit
  int is_seq = IsSequenceHelper(nested);
  if (is_seq == -1) return false;
  if (!is_seq) {
    return PyList_Append(list, nested) != -1;
  }

  // if nested if dictionary, sort it by key and recurse on each value
  if (PyDict_Check(nested)) {
    PyObject* keys = PyDict_Keys(nested);
    if (PyList_Sort(keys) == -1) return false;
    Py_ssize_t size = PyList_Size(keys);
    for (Py_ssize_t i = 0; i < size; ++i) {
      // We know that key and val will not be deleted because nested owns
      // a reference to them and callers of flatten must not modify nested
      // while the method is running.
      PyObject* key = PyList_GET_ITEM(keys, i);
      PyObject* val = PyDict_GetItem(nested, key);
      if (Py_EnterRecursiveCall(" in Flatten")) {
        Py_DECREF(keys);
        return false;
      }
      FlattenHelper(val, list);
      Py_LeaveRecursiveCall();
    }
    Py_DECREF(keys);
    return true;
  }

  // iterate and recurse
  PyObject* item;
  PyObject* iterator = PyObject_GetIter(nested);
  while ((item = PyIter_Next(iterator)) != nullptr) {
    FlattenHelper(item, list);
    Py_DECREF(item);
  }
  Py_DECREF(iterator);
  return true;
}

}  // anonymous namespace

void RegisterSequenceClass(PyObject* sequence_class) {
  if (!PyType_Check(sequence_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `collections.Sequence`. Got ",
            Py_TYPE(sequence_class)->tp_name)
            .c_str());
    return;
  }
  CollectionsSequenceType = sequence_class;
}

bool IsSequence(PyObject* o) { return IsSequenceHelper(o) == 1; }

PyObject* Flatten(PyObject* nested) {
  PyObject* list = PyList_New(0);
  if (FlattenHelper(nested, list)) {
    return list;
  } else {
    Py_DECREF(list);
    return nullptr;
  }
}
}  // namespace swig
}  // namespace tensorflow

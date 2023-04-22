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
#include <Python.h>

#include "absl/container/flat_hash_map.h"
#include "pybind11/pybind11.h"
#include "tensorflow/core/platform/logging.h"

namespace py = pybind11;
constexpr int PY_MODULE_TYPE_TP_BASIC_SIZE = 56;

struct FastModuleObject {
  // A dummy array that ensures enough size is reserved for FastModuleObject,
  // because it's inherited from PyModuleObject.
  const std::array<char, PY_MODULE_TYPE_TP_BASIC_SIZE> opaque_base_fields;
  // A cache that helps reduce attribute lookup overhead.
  absl::flat_hash_map<PyObject *, PyObject *> attr_map;
  // pointer to the external getattribute function
  PyObject *cb_getattribute = nullptr;
  // pointer to the external getattr function
  PyObject *cb_getattr = nullptr;
  // static PyTypeObject type;

  FastModuleObject() = delete;
  ~FastModuleObject() = delete;
  static FastModuleObject *UncheckedCast(PyObject *obj);
};

static int FastModule_init(FastModuleObject *self, PyObject *args,
                           PyObject *kwds) {
  DCHECK_EQ(PY_MODULE_TYPE_TP_BASIC_SIZE, PyModule_Type.tp_basicsize);
  if (PyModule_Type.tp_init(reinterpret_cast<PyObject *>(self), args, kwds) < 0)
    return -1;
  new (&(self->attr_map)) absl::flat_hash_map<PyObject *, PyObject *>();
  return 0;
}

// Parses the input as a callable and checks the result.
static PyObject *ParseFunc(PyObject *args) {
  PyObject *func;
  if (!PyArg_ParseTuple(args, "O:set_callback", &func)) return nullptr;
  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_TypeError, "input args must be callable");
    return nullptr;
  }
  Py_INCREF(func);  // Add a reference to new callback
  return func;
}

// Sets the pointer 'cb_getattribute' in the FastModuleObject object
// corresponding to 'self'.
static PyObject *SetGetattributeCallback(PyObject *self, PyObject *args) {
  PyObject *func = ParseFunc(args);
  // Dispose of previous callback
  Py_XDECREF(FastModuleObject::UncheckedCast(self)->cb_getattribute);
  // Remember new callback
  FastModuleObject::UncheckedCast(self)->cb_getattribute = func;
  Py_RETURN_NONE;
}

// Sets the pointer 'cb_getattr' in the FastModuleObject object
// corresponding to 'self'.
static PyObject *SetGetattrCallback(PyObject *self, PyObject *args) {
  PyObject *func = ParseFunc(args);
  // Dispose of previous callback
  Py_XDECREF(FastModuleObject::UncheckedCast(self)->cb_getattr);
  // Remember new callback
  FastModuleObject::UncheckedCast(self)->cb_getattr = func;
  Py_RETURN_NONE;
}

// Inserts or updates a key-value pair in the cache 'attr_map'
// of the FastModuleObject object corresponding to 'self'.
static PyObject *FastDictInsert(FastModuleObject *self, PyObject *args) {
  PyObject *name, *value;
  if (!PyArg_ParseTuple(args, "OO", &name, &value)) {
    PyErr_SetString(PyExc_TypeError, "_fastdict_insert: incorrect inputs");
    return nullptr;
  }
  auto &attr_map = self->attr_map;
  if (attr_map.find(name) != attr_map.end()) {
    Py_DECREF(name);
    Py_DECREF(value);
  }
  attr_map.insert_or_assign(name, value);
  // Increment the reference count
  Py_INCREF(name);
  Py_INCREF(value);
  // Properly handle returning Py_None
  Py_RETURN_NONE;
}

// Gets a value from a key in the cache 'attr_map'
// of the FastModuleObject object corresponding to 'self'.
static PyObject *FastDictGet(FastModuleObject *self, PyObject *args) {
  PyObject *name;
  if (!PyArg_ParseTuple(args, "O", &name)) {
    PyErr_SetString(PyExc_TypeError, "_fastdict_get: incorrect inputs");
    return nullptr;
  }
  auto &attr_map = self->attr_map;
  auto result = attr_map.find(name);
  if (result != attr_map.end()) {
    PyObject *value = result->second;
    Py_INCREF(value);
    return value;
  }
  // Copied from CPython's moduleobject.c
  PyErr_Format(PyExc_KeyError, "module has no attribute '%U'", name);
  return nullptr;
}

// Returns true if a key exists in the cache 'attr_map'
// of the FastModuleObject object corresponding to 'self',
// otherwise returns false.
static PyObject *FastDictContains(FastModuleObject *self, PyObject *args) {
  PyObject *name;
  if (!PyArg_ParseTuple(args, "O", &name)) {
    PyErr_SetString(PyExc_TypeError, "_fastdict_key_in: incorrect inputs");
    return nullptr;
  }
  const auto &attr_map = self->attr_map;
  const auto result = attr_map.contains(name);
  if (result) {
    // Properly handle returning Py_True
    Py_RETURN_TRUE;
  }
  // Properly handle returning Py_False
  Py_RETURN_FALSE;
}

// Calls a function 'func' with inputs 'self' and 'args'.
static PyObject *CallFunc(FastModuleObject *self, PyObject *args,
                          PyObject *func) {
  if (func == nullptr) {
    PyErr_SetString(PyExc_NameError,
                    "Attempting to call a callback that was not defined");
    return nullptr;
  }
  PyObject *name;
  if (!PyArg_ParseTuple(args, "O", &name)) {
    PyErr_SetString(PyExc_TypeError, "CallFunc: incorrect inputs");
    return nullptr;
  }
  PyObject *arglist = Py_BuildValue("(OO)", self, name);
  auto result = PyObject_CallObject(func, arglist);
  Py_DECREF(arglist);
  return result;
}

static PyMethodDef FastModule_methods[] = {
    {"_fastdict_insert", reinterpret_cast<PyCFunction>(FastDictInsert),
     METH_VARARGS, "Registers a method to the fast lookup table."},
    {"_fastdict_get", reinterpret_cast<PyCFunction>(FastDictGet), METH_VARARGS,
     "Gets a method from the fast lookup table."},
    {"_fastdict_key_in", reinterpret_cast<PyCFunction>(FastDictContains),
     METH_VARARGS, "Checks if a method exists in the fast lookup table."},
    {"set_getattribute_callback", SetGetattributeCallback, METH_VARARGS,
     "Defines the callback function to replace __getattribute__"},
    {"set_getattr_callback", SetGetattrCallback, METH_VARARGS,
     "Defines the callback function to replace __getattr__"},
    {nullptr, nullptr, 0, nullptr},
};

// Attempts to get the attribute based on 'name' as the key in cache 'attr_map'
// of the FastModuleObject object corresponding to 'module'.
// If the lookup fails in the cache, either uses
// a user-defined callback 'cb_getattribute'
// or the default 'tp_getattro' function to look for the attribute.
static PyObject *FastTpGetattro(PyObject *module, PyObject *name) {
  FastModuleObject *fast_module = FastModuleObject::UncheckedCast(module);
  auto &attr_map = fast_module->attr_map;
  auto it = attr_map.find(name);
  // If the attribute lookup is successful in the cache, directly return it.
  if (it != attr_map.end()) {
    PyObject *value = it->second;
    Py_INCREF(value);
    return value;
  }
  PyObject *arglist = Py_BuildValue("(O)", name);
  PyObject *result;
  // Prefer the customized callback function over the default function.
  if (fast_module->cb_getattribute != nullptr) {
    result = CallFunc(fast_module, arglist, fast_module->cb_getattribute);
  } else {
    result = PyModule_Type.tp_getattro(module, name);
  }
  // Return result if it's found
  if (result != nullptr) {
    return result;
  }
  // If the default lookup fails and an AttributeError is raised,
  // clear the error status before using the __getattr__ callback function.
  auto is_error = PyErr_Occurred();
  if (is_error && PyErr_ExceptionMatches(PyExc_AttributeError) &&
      fast_module->cb_getattr != nullptr) {
    PyErr_Clear();
    return CallFunc(fast_module, arglist, fast_module->cb_getattr);
  }
  // If all options were used up
  return result;
}

// Customized destructor for FastModuleType.tp_dealloc
// In addition to default behavior it also clears up the contents in attr_map.
static void FastModuleObjectDealloc(PyObject *module) {
  auto &attr_map = FastModuleObject::UncheckedCast(module)->attr_map;
  for (auto &it : attr_map) {
    Py_DECREF(it.first);
    Py_DECREF(it.second);
  }
  attr_map.~flat_hash_map<PyObject *, PyObject *>();
  Py_TYPE(module)->tp_free(module);
}

static PyTypeObject FastModuleType = []() {
  PyTypeObject obj = {PyVarObject_HEAD_INIT(&PyType_Type, 0)};
  obj.tp_name = "fast_module_type.FastModuleType";
  obj.tp_basicsize = sizeof(FastModuleObject);
  obj.tp_itemsize = 0;
  obj.tp_dealloc = FastModuleObjectDealloc;
  obj.tp_getattro = FastTpGetattro;
  obj.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  obj.tp_doc = "FastModuleType objects";
  obj.tp_methods = FastModule_methods;
  obj.tp_init = reinterpret_cast<initproc>(FastModule_init);
  return obj;
}();

// Returns true if the type of 'obj' or any of its parent class
// is equal to 'target'. Otherwise returns false.
bool IsAnyBaseSameType(const PyObject *obj, const PyTypeObject *target) {
  auto *tp = Py_TYPE(obj);
  while (true) {
    if (tp == target) return true;
    // If the default type is found, there is no need to search further
    if (tp == &PyBaseObject_Type) break;
    tp = tp->tp_base;
  }
  return false;
}

// Casts 'obj' to 'FastModuleObject *'.
// Conducts a check only in non-optimized builds.
FastModuleObject *FastModuleObject::UncheckedCast(PyObject *obj) {
  DCHECK(IsAnyBaseSameType(obj, &FastModuleType));
  return reinterpret_cast<FastModuleObject *>(obj);
}

PYBIND11_MODULE(fast_module_type, m) {
  FastModuleType.tp_base = &PyModule_Type;
  FastModuleType.tp_setattro = [](PyObject *module, PyObject *name,
                                  PyObject *value) -> int {
    auto &attr_map = FastModuleObject::UncheckedCast(module)->attr_map;
    if (attr_map.find(name) != attr_map.end()) {
      Py_DECREF(name);
      Py_DECREF(value);
    }
    attr_map.insert_or_assign(name, value);
    // Increment the reference count
    Py_INCREF(name);
    Py_INCREF(value);
    PyObject_GenericSetAttr(module, name, value);
    return 0;
  };

  m.doc() = R"pbdoc(
    fast_module_type
    -----
  )pbdoc";
  // Use getter function to hold attributes rather than pybind11's m.attr due to
  // b/145559202.
  m.def(
      "get_fast_module_type_class",
      []() {
        return py::cast<py::object>(
            reinterpret_cast<PyObject *>(&FastModuleType));
      },
      py::return_value_policy::reference);
}

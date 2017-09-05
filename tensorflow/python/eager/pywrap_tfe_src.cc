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

// Must be included first.
#include "tensorflow/python/lib/core/numpy.h"

#include "tensorflow/python/eager/pywrap_tfe.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"

using tensorflow::string;

namespace {

#define PARSE_VALUE(fn_name, type, check_fn, parse_fn)                       \
  bool fn_name(const string& key, PyObject* py_value, TF_Status* status,     \
               type* value) {                                                \
    if (check_fn(py_value)) {                                                \
      *value = static_cast<type>(parse_fn(py_value));                        \
      return true;                                                           \
    } else {                                                                 \
      TF_SetStatus(status, TF_INVALID_ARGUMENT,                              \
                   tensorflow::strings::StrCat(                              \
                       "Expecting " #type " value for attr ", key, ", got ", \
                       py_value->ob_type->tp_name)                           \
                       .c_str());                                            \
      return false;                                                          \
    }                                                                        \
  }

#if PY_MAJOR_VERSION >= 3
PARSE_VALUE(ParseIntValue, int, PyLong_Check, PyLong_AsLong)
PARSE_VALUE(ParseInt64Value, int64_t, PyLong_Check, PyLong_AsLong)
#else
PARSE_VALUE(ParseIntValue, int, PyInt_Check, PyInt_AsLong)
PARSE_VALUE(ParseInt64Value, int64_t, PyInt_Check, PyInt_AsLong)
#endif
PARSE_VALUE(ParseFloatValue, float, PyFloat_Check, PyFloat_AsDouble)
#undef PARSE_VALUE

bool ParseStringValue(const string& key, PyObject* py_value, TF_Status* status,
                      const char** value) {
  if (PyBytes_Check(py_value)) {
    *value = PyBytes_AsString(py_value);
    return true;
  }
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(py_value)) {
    *value = PyUnicode_AsUTF8(py_value);
    return true;
  }
#endif
  TF_SetStatus(
      status, TF_INVALID_ARGUMENT,
      tensorflow::strings::StrCat("Expecting a const char* value for attr ",
                                  key, ", got ", py_value->ob_type->tp_name)
          .c_str());
  return false;
}

bool ParseBoolValue(const string& key, PyObject* py_value, TF_Status* status,
                    unsigned char* value) {
  *value = PyObject_IsTrue(py_value);
  return true;
}

const char* ParseProtoValue(const string& key, const char* proto_name,
                            PyObject* py_value, size_t* size,
                            TF_Status* status) {
  char* output = nullptr;
  Py_ssize_t py_size;
  if (PyBytes_Check(py_value) &&
      PyBytes_AsStringAndSize(py_value, &output, &py_size) >= 0) {
    *size = static_cast<size_t>(py_size);
    return output;
  }
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(py_value) &&
      (output = PyUnicode_AsUTF8AndSize(py_value, &py_size)) != nullptr) {
    *size = static_cast<size_t>(py_size);
    return output;
  }
#endif
  TF_SetStatus(status, TF_INVALID_ARGUMENT,
               tensorflow::strings::StrCat("Expecting a string (serialized ",
                                           proto_name, ") value for attr ", key)
                   .c_str());
  return nullptr;
}

bool SetOpAttrList(TFE_Op* op, const char* key, PyObject* py_list,
                   TF_AttrType type, TF_Status* status) {
  if (!PySequence_Check(py_list)) {
    TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        tensorflow::strings::StrCat("Expecting sequence value for attr ", key,
                                    ", got ", py_list->ob_type->tp_name)
            .c_str());
    return false;
  }
  const int num_values = PySequence_Size(py_list);

#define PARSE_LIST(c_type, parse_fn)                                \
  std::unique_ptr<c_type[]> values(new c_type[num_values]);         \
  for (int i = 0; i < num_values; ++i) {                            \
    auto py_value = PySequence_ITEM(py_list, i);                    \
    if (!parse_fn(key, py_value, status, &values[i])) return false; \
  }

  if (type == TF_ATTR_STRING) {
    PARSE_LIST(const char*, ParseStringValue);
    TFE_OpSetAttrStringList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_INT) {
    PARSE_LIST(int64_t, ParseInt64Value);
    TFE_OpSetAttrIntList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_FLOAT) {
    PARSE_LIST(float, ParseFloatValue);
    TFE_OpSetAttrFloatList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_BOOL) {
    PARSE_LIST(unsigned char, ParseBoolValue);
    TFE_OpSetAttrBoolList(op, key, values.get(), num_values);
  } else if (type == TF_ATTR_TYPE) {
    PARSE_LIST(int, ParseIntValue);
    TFE_OpSetAttrTypeList(op, key,
                          reinterpret_cast<const TF_DataType*>(values.get()),
                          num_values);
  } else if (type == TF_ATTR_SHAPE) {
    // Make one pass through the input counting the total number of
    // dims across all the input lists.
    int total_dims = 0;
    for (int i = 0; i < num_values; ++i) {
      auto py_value = PySequence_ITEM(py_list, i);
      if (py_value != Py_None) {
        if (!PySequence_Check(py_value)) {
          TF_SetStatus(
              status, TF_INVALID_ARGUMENT,
              tensorflow::strings::StrCat(
                  "Expecting None or sequence value for element", i,
                  " of attr ", key, ", got ", py_value->ob_type->tp_name)
                  .c_str());
          return false;
        }
        const auto size = PySequence_Size(py_value);
        total_dims += size;
      }
    }
    // Allocate a buffer that can fit all of the dims together.
    std::unique_ptr<int64_t[]> buffer(new int64_t[total_dims]);
    // Copy the input dims into the buffer and set dims to point to
    // the start of each list's dims.
    std::unique_ptr<const int64_t* []> dims(new const int64_t*[num_values]);
    std::unique_ptr<int[]> num_dims(new int[num_values]);
    int64_t* offset = buffer.get();
    for (int i = 0; i < num_values; ++i) {
      auto py_value = PySequence_ITEM(py_list, i);
      if (py_value == Py_None) {
        dims[i] = nullptr;
        num_dims[i] = -1;
      } else {
        const auto size = PySequence_Size(py_value);
        dims[i] = offset;
        num_dims[i] = size;
        for (int j = 0; j < size; ++j) {
          auto inner_py_value = PySequence_ITEM(py_value, j);
          if (inner_py_value == Py_None) {
            *offset = -1;
          } else if (!ParseInt64Value(key, inner_py_value, status, offset)) {
            return false;
          }
          ++offset;
        }
      }
    }
    TFE_OpSetAttrShapeList(op, key, dims.get(), num_dims.get(), num_values,
                           status);
    if (TF_GetCode(status) != TF_OK) return false;
  } else {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 tensorflow::strings::StrCat("Attr ", key,
                                             " has unhandled list type ", type)
                     .c_str());
    return false;
  }
#undef PARSE_LIST
  return true;
}

bool SetOpAttrScalar(TFE_Op* op, const char* key, PyObject* py_value,
                     TF_AttrType type, TF_Status* status) {
  if (type == TF_ATTR_STRING) {
    const char* value;
    if (!ParseStringValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrString(op, key, value);
  } else if (type == TF_ATTR_INT) {
    int64_t value;
    if (!ParseInt64Value(key, py_value, status, &value)) return false;
    TFE_OpSetAttrInt(op, key, value);
  } else if (type == TF_ATTR_FLOAT) {
    float value;
    if (!ParseFloatValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrFloat(op, key, value);
  } else if (type == TF_ATTR_BOOL) {
    unsigned char value;
    if (!ParseBoolValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrBool(op, key, value);
  } else if (type == TF_ATTR_TYPE) {
    int value;
    if (!ParseIntValue(key, py_value, status, &value)) return false;
    TFE_OpSetAttrType(op, key, static_cast<TF_DataType>(value));
  } else if (type == TF_ATTR_SHAPE) {
    if (py_value == Py_None) {
      TFE_OpSetAttrShape(op, key, nullptr, -1, status);
    } else {
      if (!PySequence_Check(py_value)) {
        TF_SetStatus(status, TF_INVALID_ARGUMENT,
                     tensorflow::strings::StrCat(
                         "Expecting None or sequence value for attr", key,
                         ", got ", py_value->ob_type->tp_name)
                         .c_str());
        return false;
      }
      const auto num_dims = PySequence_Size(py_value);
      std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
      for (int i = 0; i < num_dims; ++i) {
        auto inner_py_value = PySequence_ITEM(py_value, i);
        if (inner_py_value == Py_None) {
          dims[i] = -1;
        } else if (!ParseInt64Value(key, inner_py_value, status, &dims[i])) {
          return false;
        }
      }
      TFE_OpSetAttrShape(op, key, dims.get(), num_dims, status);
    }
    if (TF_GetCode(status) != TF_OK) return false;
  } else {
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        tensorflow::strings::StrCat("Attr ", key, " has unhandled type ", type)
            .c_str());
    return false;
  }
  return true;
}

void SetOpAttrs(TFE_Op* op, PyObject* attrs, TF_Status* out_status) {
  if (attrs == Py_None) return;
  if (!PyTuple_Check(attrs)) {
    TF_SetStatus(out_status, TF_INVALID_ARGUMENT, "Expecting an attrs tuple.");
    return;
  }
  Py_ssize_t len = PyTuple_GET_SIZE(attrs);
  if ((len & 1) != 0) {
    TF_SetStatus(out_status, TF_INVALID_ARGUMENT,
                 "Expecting attrs tuple to have even length.");
    return;
  }
  // Parse attrs
  for (Py_ssize_t i = 0; i < len; i += 2) {
    PyObject* py_key = PyTuple_GET_ITEM(attrs, i);
    PyObject* py_value = PyTuple_GET_ITEM(attrs, i + 1);
#if PY_MAJOR_VERSION >= 3
    const char* key = PyBytes_Check(py_key) ? PyBytes_AsString(py_key)
                                            : PyUnicode_AsUTF8(py_key);
#else
    const char* key = PyBytes_AsString(py_key);
#endif
    unsigned char is_list = 0;
    const TF_AttrType type = TFE_OpGetAttrType(op, key, &is_list, out_status);
    if (TF_GetCode(out_status) != TF_OK) return;
    if (is_list != 0) {
      if (!SetOpAttrList(op, key, py_value, type, out_status)) return;
    } else {
      if (!SetOpAttrScalar(op, key, py_value, type, out_status)) return;
    }
  }
}
}  // namespace

void TFE_Py_Execute(TFE_Context* ctx, const char* device_name,
                    const char* op_name, TFE_InputTensorHandles* inputs,
                    PyObject* attrs, TFE_OutputTensorHandles* outputs,
                    TF_Status* out_status) {
  TFE_Op* op = TFE_NewOp(ctx, op_name, out_status);
  if (TF_GetCode(out_status) != TF_OK) return;
  TFE_OpSetDevice(op, ctx, device_name, out_status);
  if (TF_GetCode(out_status) == TF_OK) {
    for (int i = 0; i < inputs->size() && TF_GetCode(out_status) == TF_OK;
         ++i) {
      TFE_OpAddInput(op, inputs->at(i), out_status);
    }
  }
  if (TF_GetCode(out_status) == TF_OK) {
    SetOpAttrs(op, attrs, out_status);
  }
  if (TF_GetCode(out_status) == TF_OK) {
    int num_outputs = outputs->size();
    TFE_Execute(op, outputs->data(), &num_outputs, out_status);
    outputs->resize(num_outputs);
  }
  if (TF_GetCode(out_status) != TF_OK) {
    TF_SetStatus(out_status, TF_GetCode(out_status),
                 tensorflow::strings::StrCat(TF_Message(out_status),
                                             " [Op:", op_name, "]")
                     .c_str());
  }
  TFE_DeleteOp(op);
}

PyObject* TFE_Py_TensorHandleToNumpy(TFE_TensorHandle* h, TF_Status* status) {
  const tensorflow::Tensor* t =
      TFE_TensorHandleUnderlyingTensorInHostMemory(h, status);
  if (TF_GetCode(status) != TF_OK) {
    Py_RETURN_NONE;
  }
  PyObject* ret = nullptr;
  auto cppstatus = tensorflow::TensorToNdarray(*t, &ret);
  if (!cppstatus.ok()) {
    TF_SetStatus(status, TF_Code(cppstatus.code()),
                 cppstatus.error_message().c_str());
  }
  if (ret != nullptr) return ret;
  Py_RETURN_NONE;
}

namespace {
// Python subclass of Exception that is created on not ok Status.
tensorflow::mutex exception_class_mutex(tensorflow::LINKER_INITIALIZED);
PyObject* exception_class GUARDED_BY(exception_class_mutex) = nullptr;
}  // namespace

TFE_TensorHandle* TFE_Py_NumpyToTensorHandle(PyObject* obj) {
  tensorflow::Tensor t;
  auto cppstatus = tensorflow::NdarrayToTensor(obj, &t);
  if (cppstatus.ok()) {
    return TFE_NewTensorHandle(t);
  } else {
    tensorflow::mutex_lock l(exception_class_mutex);
    auto msg = tensorflow::strings::StrCat(
        "failed to convert numpy ndarray to a Tensor (",
        cppstatus.error_message(), ")");
    if (exception_class != nullptr) {
      PyErr_SetObject(exception_class,
                      Py_BuildValue("si", msg.c_str(), TF_INVALID_ARGUMENT));
    } else {
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());
    }
  }
  return nullptr;
}

PyObject* TFE_Py_RegisterExceptionClass(PyObject* e) {
  tensorflow::mutex_lock l(exception_class_mutex);
  if (exception_class != nullptr) {
    Py_DECREF(exception_class);
  }
  if (PyObject_IsSubclass(e, PyExc_Exception) <= 0) {
    exception_class = nullptr;
    PyErr_SetString(PyExc_TypeError,
                    "TFE_Py_RegisterExceptionClass: "
                    "Registered class should be subclass of Exception.");
    return nullptr;
  } else {
    Py_INCREF(e);
    exception_class = e;
    Py_RETURN_NONE;
  }
}

int TFE_Py_MayBeRaiseException(TF_Status* status) {
  if (TF_GetCode(status) == TF_OK) return 0;
  tensorflow::mutex_lock l(exception_class_mutex);
  if (exception_class != nullptr) {
    PyErr_SetObject(exception_class, Py_BuildValue("si", TF_Message(status),
                                                   TF_GetCode(status)));
  } else {
    PyErr_SetString(PyExc_RuntimeError, TF_Message(status));
  }
  return -1;
}

char* TFE_GetPyThonString(PyObject* o) {
  if (PyBytes_Check(o)) {
    return PyBytes_AsString(o);
  }
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(o)) {
    return PyUnicode_AsUTF8(o);
  }
#endif
  return nullptr;
}

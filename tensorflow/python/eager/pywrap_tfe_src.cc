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

#include "tensorflow/python/eager/pywrap_tfe.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/tape.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/eager/pywrap_tensor.h"

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
      tensorflow::strings::StrCat("Expecting a string value for attr ", key,
                                  ", got ", py_value->ob_type->tp_name)
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

bool SetOpAttrScalar(TFE_Context* ctx, TFE_Op* op, const char* key,
                     PyObject* py_value, TF_AttrType type, TF_Status* status) {
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
  } else if (type == TF_ATTR_FUNC) {
    // Allow:
    // (1) String function name, OR
    // (2) A Python object with a .name attribute
    //     (A crude test for being a
    //     tensorflow.python.framework.function._DefinedFunction)
    //     (which is what the various "defun" or "Defun" decorators do).
    // And in the future also allow an object that can encapsulate
    // the function name and its attribute values.
    const char* func_name = nullptr;
    if (!ParseStringValue(key, py_value, status, &func_name)) {
      PyObject* name_attr = PyObject_GetAttrString(py_value, "name");
      if (name_attr == nullptr ||
          !ParseStringValue(key, name_attr, status, &func_name)) {
        TF_SetStatus(
            status, TF_INVALID_ARGUMENT,
            tensorflow::strings::StrCat(
                "unable to set function value attribute from a ",
                py_value->ob_type->tp_name,
                " object. If you think this is an error, please file an issue "
                "at https://github.com/tensorflow/tensorflow/issues/new")
                .c_str());
        return false;
      }
    }
    TFE_Op* func = TFE_NewOp(ctx, func_name, status);
    if (TF_GetCode(status) != TF_OK) return false;
    TFE_OpSetAttrFunction(op, key, func);
    TFE_DeleteOp(func);
  } else {
    TF_SetStatus(
        status, TF_UNIMPLEMENTED,
        tensorflow::strings::StrCat("Attr ", key, " has unhandled type ", type)
            .c_str());
    return false;
  }
  return true;
}

void SetOpAttrs(TFE_Context* ctx, TFE_Op* op, PyObject* attrs,
                TF_Status* out_status) {
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
      if (!SetOpAttrScalar(ctx, op, key, py_value, type, out_status)) return;
    }
  }
}

// Python subclass of Exception that is created on not ok Status.
tensorflow::mutex exception_class_mutex(tensorflow::LINKER_INITIALIZED);
PyObject* exception_class GUARDED_BY(exception_class_mutex) = nullptr;

static tensorflow::mutex _uid_mutex(tensorflow::LINKER_INITIALIZED);
static tensorflow::int64 _uid GUARDED_BY(_uid_mutex) = 0;

}  // namespace

void TFE_Py_Execute(TFE_Context* ctx, const char* device_name,
                    const char* op_name, TFE_InputTensorHandles* inputs,
                    PyObject* attrs, TFE_OutputTensorHandles* outputs,
                    TF_Status* out_status) {
  TFE_Op* op = TFE_NewOp(ctx, op_name, out_status);
  if (TF_GetCode(out_status) != TF_OK) return;
  TFE_OpSetDevice(op, device_name, out_status);
  if (TF_GetCode(out_status) == TF_OK) {
    for (int i = 0; i < inputs->size() && TF_GetCode(out_status) == TF_OK;
         ++i) {
      TFE_OpAddInput(op, inputs->at(i), out_status);
    }
  }
  if (TF_GetCode(out_status) == TF_OK) {
    SetOpAttrs(ctx, op, attrs, out_status);
  }
  Py_BEGIN_ALLOW_THREADS;
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
  Py_END_ALLOW_THREADS;
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

int MaybeRaiseExceptionFromTFStatus(TF_Status* status, PyObject* exception) {
  if (TF_GetCode(status) == TF_OK) return 0;
  const char* msg = TF_Message(status);
  if (exception == nullptr) {
    tensorflow::mutex_lock l(exception_class_mutex);
    if (exception_class != nullptr) {
      PyErr_SetObject(exception_class,
                      Py_BuildValue("si", msg, TF_GetCode(status)));
      return -1;
    } else {
      exception = PyExc_RuntimeError;
    }
  }
  // May be update already set exception.
  PyErr_SetString(exception, msg);
  return -1;
}

int MaybeRaiseExceptionFromStatus(const tensorflow::Status& status,
                                  PyObject* exception) {
  if (status.ok()) return 0;
  const char* msg = status.error_message().c_str();
  if (exception == nullptr) {
    tensorflow::mutex_lock l(exception_class_mutex);
    if (exception_class != nullptr) {
      PyErr_SetObject(exception_class, Py_BuildValue("si", msg, status.code()));
      return -1;
    } else {
      exception = PyExc_RuntimeError;
    }
  }
  // May be update already set exception.
  PyErr_SetString(exception, msg);
  return -1;
}

char* TFE_GetPythonString(PyObject* o) {
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

int64_t get_uid() {
  tensorflow::mutex_lock l(_uid_mutex);
  return _uid++;
}

PyObject* TFE_Py_UID() { return PyLong_FromLongLong(get_uid()); }

void TFE_DeleteContextCapsule(PyObject* context) {
  TF_Status* status = TF_NewStatus();
  TFE_Context* ctx =
      reinterpret_cast<TFE_Context*>(PyCapsule_GetPointer(context, nullptr));
  TFE_DeleteContext(ctx, status);
  TF_DeleteStatus(status);
}

using GradientTape = tensorflow::eager::GradientTape<PyObject, PyObject>;

typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      GradientTape* tape;
} TFE_Py_Tape;

static void TFE_Py_Tape_Delete(PyObject* tape) {
  delete reinterpret_cast<TFE_Py_Tape*>(tape)->tape;
  Py_TYPE(tape)->tp_free(tape);
}

static PyTypeObject TFE_Py_Tape_Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "tfe.Tape", /* tp_name */
    sizeof(TFE_Py_Tape),                          /* tp_basicsize */
    0,                                            /* tp_itemsize */
    &TFE_Py_Tape_Delete,                          /* tp_dealloc */
    nullptr,                                      /* tp_print */
    nullptr,                                      /* tp_getattr */
    nullptr,                                      /* tp_setattr */
    nullptr,                                      /* tp_reserved */
    nullptr,                                      /* tp_repr */
    nullptr,                                      /* tp_as_number */
    nullptr,                                      /* tp_as_sequence */
    nullptr,                                      /* tp_as_mapping */
    nullptr,                                      /* tp_hash  */
    nullptr,                                      /* tp_call */
    nullptr,                                      /* tp_str */
    nullptr,                                      /* tp_getattro */
    nullptr,                                      /* tp_setattro */
    nullptr,                                      /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                           /* tp_flags */
    "TFE_Py_Tape objects",                        /* tp_doc */
};

PyObject* TFE_Py_NewTape() {
  TFE_Py_Tape_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&TFE_Py_Tape_Type) < 0) return nullptr;
  TFE_Py_Tape* tape = PyObject_NEW(TFE_Py_Tape, &TFE_Py_Tape_Type);
  tape->tape = new GradientTape();
  return reinterpret_cast<PyObject*>(tape);
}

static tensorflow::int64 MakeInt(PyObject* integer) {
#if PY_MAJOR_VERSION >= 3
  return PyLong_AsLong(integer);
#else
  return PyInt_AsLong(integer);
#endif
}

static std::vector<tensorflow::int64> MakeIntList(PyObject* list) {
  if (list == Py_None) {
    return {};
  }
  PyObject* seq = PySequence_Fast(list, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Size(list);
  std::vector<tensorflow::int64> tensor_ids;
  tensor_ids.reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    if (PyLong_Check(item)) {
      tensorflow::int64 id = MakeInt(item);
      tensor_ids.push_back(id);
    } else {
      tensor_ids.push_back(-1);
    }
  }
  Py_DECREF(seq);
  return tensor_ids;
}

PyObject* TFE_Py_TapeShouldRecord(PyObject* py_tape, PyObject* tensors) {
  if (tensors == Py_None) {
    Py_RETURN_FALSE;
  }
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return nullptr;
  }
  int len = PySequence_Fast_GET_SIZE(seq);
  // TODO(apassos) consider not building a list and changing the API to check
  // each tensor individually.
  std::vector<tensorflow::int64> tensor_ids;
  tensor_ids.reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    if (EagerTensor_CheckExact(item)) {
      tensor_ids.push_back(EagerTensor_id(item));
    } else {
      PyObject* id_field = PyObject_GetAttrString(item, "_id");
      if (id_field == nullptr) {
        return nullptr;
      }
      tensor_ids.push_back(MakeInt(id_field));
      Py_DECREF(id_field);
    }
  }
  Py_DECREF(seq);
  TFE_Py_Tape* tape = reinterpret_cast<TFE_Py_Tape*>(py_tape);
  if (tape->tape->ShouldRecord(tensor_ids)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

void TFE_Py_TapeWatch(PyObject* tape, tensorflow::int64 tensor_id) {
  reinterpret_cast<TFE_Py_Tape*>(tape)->tape->Watch(tensor_id);
}

static tensorflow::eager::TapeTensor TapeTensorFromTensor(PyObject* tensor) {
  if (EagerTensor_CheckExact(tensor)) {
    TFE_TensorHandle* t = EagerTensor_Handle(tensor);
    tensorflow::int64 id = EagerTensor_id(tensor);
    return tensorflow::eager::TapeTensor{id, t->t.dtype(), t->t.shape()};
  }
  PyObject* id_field = PyObject_GetAttrString(tensor, "_id");
  tensorflow::int64 id = MakeInt(id_field);
  Py_DECREF(id_field);
  if (PyErr_Occurred() != nullptr) {
    return tensorflow::eager::TapeTensor{
        id, static_cast<tensorflow::DataType>(0), tensorflow::TensorShape({})};
  }
  PyObject* dtype_object = PyObject_GetAttrString(tensor, "dtype");
  PyObject* dtype_enum = PyObject_GetAttrString(dtype_object, "_type_enum");
  Py_DECREF(dtype_object);
  tensorflow::DataType dtype =
      static_cast<tensorflow::DataType>(MakeInt(dtype_enum));
  Py_DECREF(dtype_enum);
  if (PyErr_Occurred() != nullptr) {
    return tensorflow::eager::TapeTensor{id, dtype,
                                         tensorflow::TensorShape({})};
  }
  static char _shape_tuple[] = "_shape_tuple";
  PyObject* shape_tuple = PyObject_CallMethod(tensor, _shape_tuple, nullptr);
  if (PyErr_Occurred() != nullptr) {
    return tensorflow::eager::TapeTensor{id, dtype,
                                         tensorflow::TensorShape({})};
  }
  auto l = MakeIntList(shape_tuple);
  Py_DECREF(shape_tuple);
  // Replace -1, which represents accidental Nones which can occur in graph mode
  // and can cause errors in shape cosntruction with 0s.
  for (auto& c : l) {
    if (c < 0) {
      c = 0;
    }
  }
  tensorflow::TensorShape shape(l);
  return tensorflow::eager::TapeTensor{id, dtype, shape};
}

std::vector<tensorflow::int64> MakeTensorIDList(PyObject* tensors) {
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Fast_GET_SIZE(seq);
  std::vector<tensorflow::int64> list;
  list.reserve(len);
  for (int i = 0; i < len; ++i) {
    PyObject* tensor = PySequence_Fast_GET_ITEM(seq, i);
    if (EagerTensor_CheckExact(tensor)) {
      list.push_back(EagerTensor_id(tensor));
    } else {
      PyObject* id_field = PyObject_GetAttrString(tensor, "_id");
      list.push_back(MakeInt(id_field));
      Py_DECREF(id_field);
    }
  }
  Py_DECREF(seq);
  return list;
}

void TFE_Py_TapeRecordOperation(PyObject* tape, PyObject* op_type,
                                PyObject* output_tensors,
                                PyObject* input_tensors,
                                PyObject* backward_function) {
  std::vector<tensorflow::int64> input_ids = MakeTensorIDList(input_tensors);
  std::vector<tensorflow::eager::TapeTensor> output_info;
  PyObject* seq = PySequence_Fast(output_tensors,
                                  "expected a sequence of integer tensor ids");
  int len = PySequence_Size(output_tensors);
  output_info.reserve(len);
  for (int i = 0; i < len; ++i) {
    output_info.push_back(
        TapeTensorFromTensor(PySequence_Fast_GET_ITEM(seq, i)));
    if (PyErr_Occurred() != nullptr) {
      Py_DECREF(seq);
      return;
    }
  }
  Py_DECREF(seq);
  char* op_type_str = nullptr;
  if (PyBytes_Check(op_type)) {
    op_type_str = PyBytes_AsString(op_type);
  } else if (PyUnicode_Check(op_type)) {
#if PY_MAJOR_VERSION >= 3
    op_type_str = PyUnicode_AsUTF8(op_type);
#else
    PyObject* py_str = PyUnicode_AsUTF8String(op_type);
    if (py_str == nullptr) return;
    op_type_str = PyBytes_AS_STRING(py_str);
    Py_DECREF(py_str);
#endif
  } else {
    PyErr_SetString(PyExc_RuntimeError, "op_type should be a string.");
    return;
  }

  Py_INCREF(backward_function);
  reinterpret_cast<TFE_Py_Tape*>(tape)->tape->RecordOperation(
      op_type_str, output_info, input_ids, backward_function,
      [backward_function]() { Py_DECREF(backward_function); });
}

void TFE_Py_TapeDeleteTrace(PyObject* tape, tensorflow::int64 tensor_id) {
  reinterpret_cast<TFE_Py_Tape*>(tape)->tape->DeleteTrace(tensor_id);
}

class PyVSpace : public tensorflow::eager::VSpace<PyObject, PyObject> {
 public:
  explicit PyVSpace(PyObject* py_vspace) : py_vspace_(py_vspace) {}

  tensorflow::Status Initialize() {
    num_elements_ = PyObject_GetAttrString(py_vspace_, "num_elements_fn");
    if (num_elements_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    aggregate_fn_ = PyObject_GetAttrString(py_vspace_, "aggregate_fn");
    if (aggregate_fn_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    zeros_ = PyObject_GetAttrString(py_vspace_, "zeros");
    if (zeros_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    ones_ =
        PyObject_GetAttrString(reinterpret_cast<PyObject*>(py_vspace_), "ones");
    if (ones_ == nullptr) {
      return tensorflow::errors::InvalidArgument("invalid vspace");
    }
    return tensorflow::Status::OK();
  }

  ~PyVSpace() override {
    Py_XDECREF(num_elements_);
    Py_XDECREF(aggregate_fn_);
    Py_XDECREF(zeros_);
    Py_XDECREF(ones_);
  }

  tensorflow::int64 NumElements(PyObject* tensor) const final {
    PyObject* arglist =
        Py_BuildValue("(O)", reinterpret_cast<PyObject*>(tensor));
    PyObject* result = PyEval_CallObject(num_elements_, arglist);
    tensorflow::int64 r = MakeInt(result);
    Py_DECREF(result);
    Py_DECREF(arglist);
    return r;
  }

  PyObject* AggregateGradients(
      tensorflow::gtl::ArraySlice<PyObject*> gradient_tensors) const final {
    PyObject* list = PyList_New(gradient_tensors.size());
    for (int i = 0; i < gradient_tensors.size(); ++i) {
      // Note: stealing a reference to the gradient tensors.
      CHECK(gradient_tensors[i] != nullptr);
      CHECK(gradient_tensors[i] != Py_None);
      PyList_SET_ITEM(list, i,
                      reinterpret_cast<PyObject*>(gradient_tensors[i]));
    }
    PyObject* arglist = Py_BuildValue("(O)", list);
    CHECK(arglist != nullptr);
    PyObject* result = PyEval_CallObject(aggregate_fn_, arglist);
    Py_DECREF(arglist);
    Py_DECREF(list);
    return result;
  }

  PyObject* Zeros(tensorflow::TensorShape shape,
                  tensorflow::DataType dtype) const final {
    PyObject* py_shape = PyTuple_New(shape.dims());
    for (int i = 0; i < shape.dims(); ++i) {
      PyTuple_SET_ITEM(py_shape, i, PyLong_FromLong(shape.dim_size(i)));
    }
    PyObject* py_dtype = PyLong_FromLong(static_cast<int>(dtype));
    PyObject* arg_list = Py_BuildValue("OO", py_shape, py_dtype);
    PyObject* result = PyEval_CallObject(zeros_, arg_list);
    Py_DECREF(arg_list);
    Py_DECREF(py_dtype);
    Py_DECREF(py_shape);
    return reinterpret_cast<PyObject*>(result);
  }

  PyObject* Ones(tensorflow::TensorShape shape,
                 tensorflow::DataType dtype) const final {
    PyObject* py_shape = PyTuple_New(shape.dims());
    for (int i = 0; i < shape.dims(); ++i) {
      PyTuple_SET_ITEM(py_shape, i, PyLong_FromLong(shape.dim_size(i)));
    }
    PyObject* py_dtype = PyLong_FromLong(static_cast<int>(dtype));
    PyObject* arg_list = Py_BuildValue("OO", py_shape, py_dtype);
    PyObject* result = PyEval_CallObject(ones_, arg_list);
    Py_DECREF(arg_list);
    Py_DECREF(py_dtype);
    Py_DECREF(py_shape);
    return result;
  }

  tensorflow::Status CallBackwardFunction(
      PyObject* backward_function,
      tensorflow::gtl::ArraySlice<PyObject*> output_gradients,
      std::vector<PyObject*>* result) const final {
    PyObject* grads = PyTuple_New(output_gradients.size());
    for (int i = 0; i < output_gradients.size(); ++i) {
      if (output_gradients[i] == nullptr) {
        Py_INCREF(Py_None);
        PyTuple_SET_ITEM(grads, i, Py_None);
      } else {
        PyTuple_SET_ITEM(grads, i,
                         reinterpret_cast<PyObject*>(output_gradients[i]));
      }
    }
    PyObject* py_result = PyEval_CallObject(
        reinterpret_cast<PyObject*>(backward_function), grads);
    Py_DECREF(grads);
    Py_DECREF(backward_function);
    if (py_result == nullptr) {
      VLOG(1) << "Gradient function threw exceptions";
      if (VLOG_IS_ON(1)) {
        PyErr_Print();
      }
      return tensorflow::errors::Internal("gradient function threw exceptions");
    }
    result->clear();
    PyObject* seq =
        PySequence_Fast(py_result, "expected a sequence of gradients");
    if (seq == nullptr) {
      return tensorflow::errors::InvalidArgument(
          "gradient function did not return a list");
    }
    int len = PySequence_Fast_GET_SIZE(seq);
    VLOG(1) << "Gradient length is " << len;
    result->reserve(len);
    for (int i = 0; i < len; ++i) {
      PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
      if (item == Py_None) {
        result->push_back(nullptr);
      } else {
        Py_INCREF(item);
        result->push_back(item);
      }
    }
    Py_DECREF(seq);
    Py_DECREF(py_result);
    return tensorflow::Status::OK();
  }

  void DeleteGradient(PyObject* tensor) const final { Py_XDECREF(tensor); }

 private:
  PyObject* py_vspace_;

  PyObject* num_elements_;
  PyObject* aggregate_fn_;
  PyObject* zeros_;
  PyObject* ones_;
};

std::vector<PyObject*> MakeTensorList(PyObject* tensors) {
  PyObject* seq = PySequence_Fast(tensors, "expected a sequence");
  if (seq == nullptr) {
    return {};
  }
  int len = PySequence_Fast_GET_SIZE(seq);
  std::vector<PyObject*> list;
  list.reserve(len);
  for (int i = 0; i < len; ++i) {
    list.push_back(PySequence_Fast_GET_ITEM(seq, i));
  }
  Py_DECREF(seq);
  return list;
}


PyObject* TFE_Py_TapeGradient(PyObject* tape, PyObject* vspace,
                              PyObject* target, PyObject* sources,
                              PyObject* output_gradients, TF_Status* status) {
  PyVSpace c_vspace(vspace);
  if (!c_vspace.Initialize().ok()) {
    return nullptr;
  }

  std::vector<tensorflow::int64> target_vec = MakeTensorIDList(target);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  std::vector<tensorflow::int64> sources_vec = MakeTensorIDList(sources);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  std::vector<PyObject*> outgrad_vec;
  if (output_gradients != Py_None) {
    outgrad_vec = MakeTensorList(output_gradients);
    if (PyErr_Occurred()) {
      return nullptr;
    }
    for (PyObject* tensor : outgrad_vec) {
      // Calling the backward function will eat a reference to the tensors in
      // outgrad_vec, so we need to increase their reference count.
      Py_INCREF(tensor);
    }
  }
  TFE_Py_Tape* tape_obj = reinterpret_cast<TFE_Py_Tape*>(tape);
  std::vector<PyObject*> result;
  status->status = tape_obj->tape->ComputeGradient(
      c_vspace, target_vec, sources_vec, outgrad_vec, &result);
  if (!status->status.ok()) {
    return nullptr;
  }
  if (!result.empty()) {
    PyObject* py_result = PyList_New(result.size());
    for (int i = 0; i < result.size(); ++i) {
      if (result[i] == nullptr) {
        Py_INCREF(Py_None);
        result[i] = Py_None;
      }
      PyList_SET_ITEM(py_result, i, reinterpret_cast<PyObject*>(result[i]));
    }
    return py_result;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

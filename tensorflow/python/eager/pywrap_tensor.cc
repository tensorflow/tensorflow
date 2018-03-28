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

#include <stdlib.h>

#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/numpy.h"
#include "tensorflow/python/lib/core/py_seq_tensor.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"

namespace {

TFE_Context* GetContext(PyObject* ctx) {
  TFE_Context* context =
      reinterpret_cast<TFE_Context*>(PyCapsule_GetPointer(ctx, nullptr));
  if (context == nullptr) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "Expecting a PyCapsule encoded context handle. Got ",
                        Py_TYPE(ctx)->tp_name)
                        .c_str());
  }
  return context;
}

// Convert a Python numpy.ndarray object to a TFE_TensorHandle.
// The two may share underlying storage so changes to one may reflect in the
// other.
TFE_TensorHandle* NumpyToTensorHandle(PyObject* obj) {
  tensorflow::Tensor t;
  auto cppstatus = tensorflow::NdarrayToTensor(obj, &t);
  if (cppstatus.ok()) {
    return TFE_NewTensorHandle(t);
  } else {
    PyErr_SetString(PyExc_ValueError,
                    tensorflow::strings::StrCat(
                        "Failed to convert numpy ndarray to a Tensor (",
                        cppstatus.error_message(), ").")
                        .c_str());
    return nullptr;
  }
}

// Casts data referred to by `handle` from type `src_type_enum` to type
// `dst_type_enum`.
TFE_TensorHandle* EagerCast(TFE_Context* ctx, TFE_TensorHandle* handle,
                            TF_DataType src_type_enum,
                            TF_DataType dst_type_enum, TF_Status* out_status) {
  if (ctx == nullptr) return nullptr;
  const char* op_name = "Cast";
  const char* device_name = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_Op* op = TFE_NewOp(ctx, op_name, out_status);
#define RETURN_ERROR  \
  {                   \
    TFE_DeleteOp(op); \
    return nullptr;   \
  }
  if (TF_GetCode(out_status) != TF_OK) RETURN_ERROR
  TFE_OpSetDevice(op, device_name, out_status);
  if (TF_GetCode(out_status) != TF_OK) RETURN_ERROR
  TFE_OpAddInput(op, handle, out_status);
  if (TF_GetCode(out_status) != TF_OK) RETURN_ERROR
  TFE_OpSetAttrType(op, "SrcT", src_type_enum);
  TFE_OpSetAttrType(op, "DstT", dst_type_enum);
  TFE_TensorHandle* output = nullptr;
  int num_outputs = 1;
  TFE_Execute(op, &output, &num_outputs, out_status);
  if (TF_GetCode(out_status) != TF_OK || num_outputs != 1 ||
      output == nullptr) {
    if (output != nullptr) {
      TFE_DeleteTensorHandle(output);
    }
    RETURN_ERROR
  }
  TFE_DeleteOp(op);
  return output;
#undef RETURN_ERROR
}

TFE_TensorHandle* CopyToDevice(TFE_TensorHandle* handle, PyObject* ctx,
                               PyObject* dev) {
  const char* device = "";
  if (dev != nullptr && dev != Py_None) {
    device = PyBytes_AsString(dev);
#if PY_MAJOR_VERSION >= 3
    if (device == nullptr) {
      PyErr_Clear();
      device = PyUnicode_AsUTF8(dev);
    }
#endif
    if (device == nullptr) {
      PyErr_SetString(PyExc_TypeError,
                      "Error parsing device argument to CopyToDevice");
      return nullptr;
    }
  }
  TFE_Context* context = GetContext(ctx);
  if (context == nullptr) {  // PyErr already set by GetContext
    return nullptr;
  }
  auto status = tensorflow::make_safe(TF_NewStatus());
  TFE_TensorHandle* new_handle =
      TFE_TensorHandleCopyToDevice(handle, context, device, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat("Error copying tensor to device: ", device,
                                    ". ", TF_Message(status.get()))
            .c_str());
    return nullptr;
  }
  return new_handle;
}

// Helper function to convert `v` to an int and store it in `*out`. Returns true
// on success, false otherwise.
// Note that we assume that v is a python int (not long) representing a
// TF_DataType value.
bool PyIntToDataType(PyObject* v, int* out) {
#if PY_MAJOR_VERSION < 3
  if (PyInt_Check(v)) {
    *out = PyInt_AS_LONG(v);
    return true;
  }
#else
  if (PyLong_Check(v)) {
    *out = PyLong_AsLong(v);
    return true;
  }
#endif
  return false;
}

// Helper function to create a python integer from TF_DataType.
PyObject* PyIntFromDataType(TF_DataType l) {
#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(l);
#else
  return PyLong_FromLong(l);
#endif
}

}  // namespace

extern "C" {

static const int kMaxEagerTensorParentSize = 64;

// TODO(agarwal): store context handle in EagerTensor.
typedef struct EagerTensor {
  PyObject_HEAD;
  // Note that we leave kMaxEagerTensorParentSize bytes here for use by the
  // parent class. The parent class is set at runtime, so we don't know the
  // exact size at compile time.
  char unused[kMaxEagerTensorParentSize];
  TFE_TensorHandle* handle;
  int64_t id;
  // This mirrors tensorflow.core.framework.ops.Tensor._handle_data Which will
  // be None for tensors of type other than DT_REOSURCE. For DT_RESOURCE
  // tensors, this will contain a serialized HandleData proto with shape
  // inference metadata about shapes and dtypes of resources accessible from
  // this handle.
  // Note that we assume that handle_data cannot participate in reference
  // cycles, and hence don't provide GC support for it.
  PyObject* handle_data;

  // This stores `_keras_mask` object and is set by Tensorflow layers.
  PyObject* keras_mask;

  // This stores `_tensor_shape`, a cached `TensorShape` object, and is set the
  // first time that `_EagerTensorBase`'s `shape` property is called.
  PyObject* tensor_shape;

  // We store a status object here as an optimization to avoid allocating a new
  // Status objects on different functions that operate on EagerTensor and need
  // to use a TF_Status object. However note that accesses to `status` are not
  // thread-safe.
  TF_Status* status;
} EagerTensor;

// tp_init for EagerTensor.
int EagerTensor_init(EagerTensor* self, PyObject* args, PyObject* kwds) {
  self->id = get_uid();
  self->handle = nullptr;
  Py_INCREF(Py_None);
  self->handle_data = Py_None;
  Py_INCREF(Py_None);
  self->keras_mask = Py_None;
  Py_INCREF(Py_None);
  self->tensor_shape = Py_None;
  self->status = TF_NewStatus();
  PyObject* value;
  PyObject* context = nullptr;
  PyObject* device = nullptr;
  PyObject* dtype = Py_None;
  const char* kwlist[] = {"value", "context", "device", "dtype", nullptr};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|O",
                                   const_cast<char**>(kwlist), &value, &context,
                                   &device, &dtype)) {
    return -1;
  }
  // Extract dtype
  int desired_dtype = -1;
  if (dtype != Py_None) {
    if (!PyIntToDataType(dtype, &desired_dtype)) {
      PyErr_SetString(PyExc_TypeError,
                      tensorflow::strings::StrCat(
                          "Expecting a DataType value for dtype. Got ",
                          Py_TYPE(dtype)->tp_name)
                          .c_str());
      return -1;
    }
  }
  tensorflow::Safe_TFE_TensorHandlePtr handle =
      tensorflow::make_safe(static_cast<TFE_TensorHandle*>(nullptr));
  PyErr_Clear();
  if (PyArray_Check(value)) {
    int desired_np_dtype = -1;
    if (desired_dtype >= 0) {
      if (!tensorflow::TF_DataType_to_PyArray_TYPE(
               static_cast<TF_DataType>(desired_dtype), &desired_np_dtype)
               .ok()) {
        PyErr_SetString(PyExc_TypeError,
                        tensorflow::strings::StrCat(
                            "Invalid dtype argument value ", desired_dtype)
                            .c_str());
        return -1;
      }
    }
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(value);
    int current_np_dtype = PyArray_TYPE(array);
    auto safe_value = tensorflow::make_safe(static_cast<PyObject*>(nullptr));
    if ((desired_np_dtype >= 0 && desired_np_dtype != current_np_dtype) ||
        !PyArray_ISCARRAY(array)) {
      int new_dtype =
          desired_np_dtype >= 0 ? desired_np_dtype : current_np_dtype;
      safe_value = tensorflow::make_safe(
          PyArray_FromAny(value, PyArray_DescrFromType(new_dtype), 0, 0,
                          NPY_ARRAY_CARRAY | NPY_ARRAY_FORCECAST, nullptr));
      if (PyErr_Occurred()) return -1;
      if (safe_value == nullptr) {
        PyErr_SetString(PyExc_ValueError, "Error while casting a numpy value");
        return -1;
      }
      value = safe_value.get();
    }
    handle = tensorflow::make_safe(NumpyToTensorHandle(value));
  } else {
    tensorflow::Tensor t;
    // TODO(josh11b): Have PySeqToTensor set python errors instead of
    // returning Status.
    auto cppstatus = tensorflow::PySeqToTensor(value, dtype, &t);
    if (!cppstatus.ok()) {
      PyErr_SetString(PyExc_ValueError, cppstatus.error_message().c_str());
      return -1;
    }
    handle = tensorflow::make_safe(TFE_NewTensorHandle(t));
  }
  if (PyErr_Occurred()) return -1;
  if (handle == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Error while creating an EagerTensor");
    return -1;
  }
  TF_DataType handle_dtype = TFE_TensorHandleDataType(handle.get());
  if (desired_dtype >= 0 && desired_dtype != handle_dtype) {
    handle = tensorflow::make_safe(
        EagerCast(GetContext(context), handle.get(), handle_dtype,
                  static_cast<TF_DataType>(desired_dtype), self->status));
    if (TF_GetCode(self->status) != TF_OK) {
      PyErr_SetString(PyExc_ValueError,
                      tensorflow::strings::StrCat(
                          "Error while casting from DataType ", handle_dtype,
                          " to ", desired_dtype, ". ", TF_Message(self->status))
                          .c_str());
      // Cleanup self->status before returning.
      TF_SetStatus(self->status, TF_OK, "");
      return -1;
    }
    handle_dtype = TFE_TensorHandleDataType(handle.get());
  }

  // Almost all TensorFlow kernels for GPU devices keep int32 tensors in host
  // memory. We approximate the same behavior for eager execution - keeping
  // int32 tensors in host memory.
  //
  // We do so to preclude the need for callers into such kernels from having to
  // explicitly place the int32 tensors in host memory. For example, without
  // this, one needed:
  //
  // with tf.device('/gpu:0'):
  //   ...// code here
  //   with tf.device('/cpu:0'):
  //     shape = tf.constant(...)
  //   y = tf.random_uniform(shape)
  //
  // Without the CPU device block, tfe.ops.random_uniform would fail since the
  // kernel expects the shape in host memory.
  //
  // With this support, we simplify the code:
  //
  // with tf.device('/gpu:0'):
  //   y = tf.random_uniform(...)
  //
  // The approximation is not exact there are GPU kernels which do not require
  // host memory for int32 tensors. This will lead to a discrepancy between
  // eager and graph execution.
  // TODO(ashankar): Fix this.
  if (handle_dtype != TF_INT32) {
    // Note that this is a shallow copy and will share the underlying buffer
    // if copying to the same device.
    handle = tensorflow::make_safe(CopyToDevice(handle.get(), context, device));
    if (handle == nullptr) return -1;
  }
  self->handle = handle.release();
  return 0;
}

// tp_dealloc for EagerTensor.
void EagerTensor_dealloc(EagerTensor* self) {
  TF_DeleteStatus(self->status);
  Py_DECREF(self->handle_data);
  Py_DECREF(self->keras_mask);
  Py_DECREF(self->tensor_shape);
  if (self->handle != nullptr) {
    TFE_DeleteTensorHandle(self->handle);
    self->handle = nullptr;
  }
  // We have the global interpreter lock, so use this chance to perform delayed
  // refcount decrements.
  tensorflow::ClearDecrefCache();
  auto id = self->id;
  Py_TYPE(self)->tp_free(self);
  TFE_Py_TapeSetDeleteTrace(id);
}

// Getter for `_id`.
static PyObject* EagerTensor_getid(EagerTensor* self, void* closure) {
  return PyLong_FromLongLong(self->id);
}

// Getter for `_datatype_enum`.
static PyObject* EagerTensor_datatype_enum(EagerTensor* self) {
  return PyIntFromDataType(TFE_TensorHandleDataType(self->handle));
}

// Getter for `_shape_tuple`.
static PyObject* EagerTensor_shape_tuple(EagerTensor* self) {
  auto handle = self->handle;
  int n = TFE_TensorHandleNumDims(handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
  PyObject* shape = PyTuple_New(n);
  if (PyErr_Occurred()) return nullptr;
  for (int i = 0; i < n; ++i) {
    PyObject* dim =
        PyLong_FromLongLong(TFE_TensorHandleDim(handle, i, self->status));
    if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError) ||
        dim == nullptr || PyTuple_SetItem(shape, i, dim) != 0) {
      // Cleanup self->status before returning.
      TF_SetStatus(self->status, TF_OK, "");
      Py_DECREF(shape);
      if (dim != nullptr) Py_DECREF(dim);
      PyErr_SetString(PyExc_RuntimeError, "Error while creating shape");
      return nullptr;
    }
  }
  return shape;
}

// Getter for `_rank`.
static PyObject* EagerTensor_rank(EagerTensor* self) {
  int num_dims = TFE_TensorHandleNumDims(self->handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
#if PY_MAJOR_VERSION < 3
  return PyInt_FromLong(num_dims);
#else
  return PyLong_FromLong(num_dims);
#endif
}

static PyObject* EagerTensor_tensor_handle(EagerTensor* self, void* unused) {
  Py_INCREF(self->handle_data);
  return self->handle_data;
}

static int EagerTensor_settensor_handle(EagerTensor* self, PyObject* value,
                                        void* unused) {
  Py_DECREF(self->handle_data);
  Py_INCREF(value);
  self->handle_data = value;
  return 0;
}

static PyObject* EagerTensor_keras_mask(EagerTensor* self, void* unused) {
  Py_INCREF(self->keras_mask);
  return self->keras_mask;
}

static int EagerTensor_setkeras_mask(EagerTensor* self, PyObject* value,
                                     void* unused) {
  Py_DECREF(self->keras_mask);
  Py_INCREF(value);
  self->keras_mask = value;
  return 0;
}

static PyObject* EagerTensor_tensor_shape(EagerTensor* self, void* unused) {
  Py_INCREF(self->tensor_shape);
  return self->tensor_shape;
}

static int EagerTensor_settensor_shape(EagerTensor* self, PyObject* value,
                                       void* unused) {
  Py_DECREF(self->tensor_shape);
  Py_INCREF(value);
  self->tensor_shape = value;
  return 0;
}
// Function `_copy_to_device`.
static PyObject* EagerTensor_copy_to_device(EagerTensor* self, PyObject* args,
                                            PyObject* kwds) {
  const char* kwlist[] = {"context", "device", nullptr};
  PyObject* ctx = nullptr;
  PyObject* dev = nullptr;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", const_cast<char**>(kwlist),
                                   &ctx, &dev) ||
      !ctx || !dev) {
    return nullptr;
  }
  auto handle = CopyToDevice(self->handle, ctx, dev);
  return EagerTensorFromHandle(handle);
}

// Function `_numpy`.
// Convert an EagerTensor to a Python numpy.ndarray object.
// The two may share underlying storage so changes to one may reflect in the
// other.
// Note that if `self` is not on CPU, we raise an Exception.
static PyObject* EagerTensor_numpy(EagerTensor* self) {
  auto status = tensorflow::make_safe(TF_NewStatus());
  const tensorflow::Tensor* t =
      TFE_TensorHandleUnderlyingTensorInHostMemory(self->handle, status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    PyErr_SetString(PyExc_RuntimeError, TF_Message(status.get()));
    return nullptr;
  }
  PyObject* ret = nullptr;
  auto cppstatus = tensorflow::TensorToNdarray(*t, &ret);
  if (MaybeRaiseExceptionFromStatus(cppstatus, PyExc_RuntimeError)) {
    Py_XDECREF(ret);
    return nullptr;
  } else {
    return ret;
  }
}

// Getter `device`.
static PyObject* EagerTensor_device(EagerTensor* self) {
  const char* device = TFE_TensorHandleDeviceName(self->handle, self->status);
  if (MaybeRaiseExceptionFromTFStatus(self->status, PyExc_ValueError)) {
    // Cleanup self->status before returning.
    TF_SetStatus(self->status, TF_OK, "");
    return nullptr;
  }
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromString(device);
#else
  return PyBytes_FromString(device);
#endif
}

static PyGetSetDef EagerTensor_getseters[] = {
    {const_cast<char*>("_id"), (getter)EagerTensor_getid, nullptr,
     const_cast<char*>("_id"), nullptr},
    {const_cast<char*>("device"), (getter)EagerTensor_device, nullptr,
     const_cast<char*>("device"), nullptr},
    {const_cast<char*>("_handle_data"), (getter)EagerTensor_tensor_handle,
     (setter)EagerTensor_settensor_handle, const_cast<char*>("_tensor_handle"),
     nullptr},
    {const_cast<char*>("_keras_mask"), (getter)EagerTensor_keras_mask,
     (setter)EagerTensor_setkeras_mask, const_cast<char*>("_keras_mask"),
     nullptr},
    {const_cast<char*>("_tensor_shape"), (getter)EagerTensor_tensor_shape,
     (setter)EagerTensor_settensor_shape, const_cast<char*>("_tensor_shape"),
     nullptr},
    {nullptr} /* Sentinel */
};

static PyMethodDef EagerTensor_methods[] = {
    {"_numpy", (PyCFunction)EagerTensor_numpy, METH_NOARGS,
     PyDoc_STR("_numpy")},
    {"_datatype_enum", (PyCFunction)EagerTensor_datatype_enum, METH_NOARGS,
     PyDoc_STR("_datatype_enum")},
    {"_shape_tuple", (PyCFunction)EagerTensor_shape_tuple, METH_NOARGS,
     PyDoc_STR("_shape_tuple")},
    {"_rank", (PyCFunction)EagerTensor_rank, METH_NOARGS, PyDoc_STR("_rank")},
    {"_copy_to_device", (PyCFunction)EagerTensor_copy_to_device,
     METH_VARARGS | METH_KEYWORDS, PyDoc_STR("_copy_to_device")},
    {nullptr, nullptr},
};

// Note that here we are trying to dynamically create a new class as a subclass
// of a "HEAPTYPE" class that is itself created in python code and passed in at
// runtime. This is fairly atypical and undocumented.
//
// We use the following strategy for this. Unfortunately, we have to use
// different approaches for python2.x vs python3.x
// For python2.x, we create the class as a static type and set its tp_base to
// the passed in type. Unfortunately setting tp_flags to include
// Py_TPFLAGS_HEAPTYPE does not work by itself since it needs some more
// initialization of the underlying PyHeapTypeObject and not doing that leads to
// some random crashes especially during garbage collection.
// python3.x explicitly disables a static subclass of a HEAPTYPE base class.
// However it provides a new function, PyType_FromSpecWithBases, to create
// types dynamically.

// Type object for EagerTensor. This is set by TFE_Py_InitEagerTensor.
PyTypeObject* EagerTensorType = nullptr;

#if PY_MAJOR_VERSION >= 3
static PyType_Slot EagerTensor_Type_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void*>(EagerTensor_dealloc)},
    {Py_tp_methods, reinterpret_cast<void*>(EagerTensor_methods)},
    {Py_tp_getset, reinterpret_cast<void*>(EagerTensor_getseters)},
    {Py_tp_init, reinterpret_cast<void*>(EagerTensor_init)},
    {0, nullptr},
};

PyType_Spec EagerTensor_Type_spec = {"EagerTensor", sizeof(EagerTensor), 0,
                                     Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE,
                                     EagerTensor_Type_slots};
#else
// TODO(agarwal): support active_trace.
static PyTypeObject _EagerTensorType = {
    // clang-format off
    PyVarObject_HEAD_INIT(nullptr, 0)
    // clang-format on
    "EagerTensor",                   /* tp_name */
    sizeof(EagerTensor),             /* tp_basicsize */
    0,                               /* tp_itemsize */
    (destructor)EagerTensor_dealloc, /* tp_dealloc */
    nullptr,                         /* tp_print */
    nullptr,                         /* tp_getattr */
    nullptr,                         /* tp_setattr */
    nullptr,                         /* tp_compare */
    nullptr,                         /* tp_repr */
    nullptr,                         /* tp_as_number */
    nullptr,                         /* tp_as_sequence */
    nullptr,                         /* tp_as_mapping */
    nullptr,                         /* tp_hash */
    nullptr,                         /* tp_call */
    nullptr,                         /* tp_str */
    nullptr,                         /* tp_getattro */
    nullptr,                         /* tp_setattro */
    nullptr,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,              /* tp_flags */
    nullptr,                         /* tp_doc */
    nullptr,                         /* tp_traverse */
    nullptr,                         /* tp_clear */
    nullptr,                         /* tp_richcompare */
    0,                               /* tp_weaklistoffset */
    nullptr,                         /* tp_iter */
    nullptr,                         /* tp_iternext */
    EagerTensor_methods,             /* tp_methods */
    nullptr,                         /* tp_members */
    EagerTensor_getseters,           /* tp_getset */
    nullptr,                         /* tp_base */
    nullptr,                         /* tp_dict */
    nullptr,                         /* tp_descr_get */
    nullptr,                         /* tp_descr_set */
    0,                               /* tp_dictoffset */
    (initproc)EagerTensor_init,      /* tp_init */
    nullptr,                         /* tp_alloc */
    nullptr,                         /* tp_new */
};

#endif

}  // extern "C"

bool EagerTensor_CheckExact(const PyObject* o) {
  return Py_TYPE(o) == EagerTensorType;
}

TFE_TensorHandle* EagerTensor_Handle(const PyObject* o) {
  return reinterpret_cast<const EagerTensor*>(o)->handle;
}

PyObject* EagerTensorFromHandle(TFE_TensorHandle* handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  EagerTensor* t = reinterpret_cast<EagerTensor*>(
      EagerTensorType->tp_new(EagerTensorType, Py_None, Py_None));
  if (t != nullptr) {
    t->id = get_uid();
    Py_INCREF(Py_None);
    t->handle_data = Py_None;
    Py_INCREF(Py_None);
    t->keras_mask = Py_None;
    Py_INCREF(Py_None);
    t->tensor_shape = Py_None;
    t->handle = handle;
    t->status = TF_NewStatus();
  }
  return reinterpret_cast<PyObject*>(t);
}

tensorflow::int64 EagerTensor_id(const PyObject* tensor) {
  CHECK(EagerTensor_CheckExact(tensor));
  return reinterpret_cast<const EagerTensor*>(tensor)->id;
}

PyObject* TFE_Py_InitEagerTensor(PyObject* base_class) {
  if (!PyType_Check(base_class)) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Expecting a class definition for `base_class` passed to ",
            "TFE_InitEagerTensor. Got ", Py_TYPE(base_class)->tp_name)
            .c_str());
    return nullptr;
  }
  // Note that we allocated kMaxEagerTensorParentSize bytes of unused space in
  // EagerTensor to allow for the space usage of the base class.
  PyTypeObject* base_class_type = reinterpret_cast<PyTypeObject*>(base_class);
  if (base_class_type->tp_basicsize > kMaxEagerTensorParentSize) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Unable to create subclass EagerTensor from base class ",
            Py_TYPE(base_class)->tp_name,
            ". Need its size to be <= ", kMaxEagerTensorParentSize)
            .c_str());
    return nullptr;
  }
  if (base_class_type->tp_itemsize != 0) {
    PyErr_SetString(
        PyExc_TypeError,
        tensorflow::strings::StrCat(
            "Unable to create subclass EagerTensor from base class ",
            Py_TYPE(base_class)->tp_name,
            " which supports variable length instances.")
            .c_str());
    return nullptr;
  }
  Py_INCREF(base_class);
#if PY_MAJOR_VERSION >= 3
  PyObject* bases = PyTuple_New(1);
  PyTuple_SET_ITEM(bases, 0, base_class);
  EagerTensorType = reinterpret_cast<PyTypeObject*>(
      PyType_FromSpecWithBases(&EagerTensor_Type_spec, bases));
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (EagerTensorType == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Error while creating EagerTensorType");
    return nullptr;
  }
#else
  _EagerTensorType.tp_base = reinterpret_cast<PyTypeObject*>(base_class);

  if (PyType_Ready(&_EagerTensorType) < 0) {
    if (PyErr_Occurred()) return nullptr;
    PyErr_SetString(PyExc_RuntimeError,
                    "Error while creating EagerTensor type.");
    return nullptr;
  }
  EagerTensorType = &_EagerTensorType;
  Py_INCREF(EagerTensorType);
#endif
  // We disable instance based attribute lookup. Its not clear if these
  // dictionaries are correctly initialized in the first place.
  EagerTensorType->tp_dictoffset = 0;
  return reinterpret_cast<PyObject*>(EagerTensorType);
}

PyObject* TFE_Py_TensorShapeSlice(PyObject* tensor_list, int slice_dim) {
  if (!PyList_Check(tensor_list)) {
    PyErr_SetString(PyExc_TypeError,
                    tensorflow::strings::StrCat(
                        "tensor_list argument must be a list. Got \"",
                        Py_TYPE(tensor_list)->tp_name, "\"")
                        .c_str());
    return nullptr;
  }
  if (slice_dim < 0) {
    PyErr_SetString(
        PyExc_ValueError,
        tensorflow::strings::StrCat("Slice dimension must be non-negative. "
                                    "Got ",
                                    slice_dim)
            .c_str());
    return nullptr;
  }

  Py_ssize_t num_tensors = PyList_Size(tensor_list);
  int64_t num_tensors_int = static_cast<int64_t>(num_tensors);
  auto tensor = tensorflow::make_safe(TF_AllocateTensor(
      TF_INT32, &num_tensors_int, /*num_dims=*/1, /*len=*/4 * num_tensors_int));
  int32_t* data = reinterpret_cast<int32_t*>(TF_TensorData(tensor.get()));
  auto status = tensorflow::make_safe(TF_NewStatus());
  for (Py_ssize_t i = 0; i < num_tensors; ++i) {
    PyObject* tensor_obj = PyList_GET_ITEM(tensor_list, i);
    if (!EagerTensor_CheckExact(tensor_obj)) {
      PyErr_SetString(PyExc_TypeError,
                      tensorflow::strings::StrCat(
                          "Expected a list of EagerTensors but "
                          "element ",
                          i, " has type \"", Py_TYPE(tensor_obj)->tp_name, "\"")
                          .c_str());
      return nullptr;
    }

    EagerTensor* t = reinterpret_cast<EagerTensor*>(tensor_obj);
    TFE_TensorHandle* handle = t->handle;
    int num_dims = TFE_TensorHandleNumDims(handle, status.get());
    if (MaybeRaiseExceptionFromTFStatus(status.get(), PyExc_ValueError)) {
      return nullptr;
    }
    if (slice_dim >= num_dims) {
      PyErr_SetString(
          PyExc_IndexError,
          tensorflow::strings::StrCat("Slice dimension (", slice_dim,
                                      ") must be smaller than rank of all "
                                      "tensors, but tensor at index ",
                                      i, " has rank ", num_dims)
              .c_str());
      return nullptr;
    }
    int64_t dim = TFE_TensorHandleDim(handle, slice_dim, status.get());
    if (MaybeRaiseExceptionFromTFStatus(status.get(), PyExc_ValueError)) {
      return nullptr;
    }
    data[i] = dim;
  }

  TFE_TensorHandle* handle = TFE_NewTensorHandle(tensor.get(), status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    PyErr_SetString(
        PyExc_RuntimeError,
        tensorflow::strings::StrCat("Failed to construct new tensor handle: ",
                                    TF_Message(status.get()))
            .c_str());
    return nullptr;
  }

  return EagerTensorFromHandle(handle);
}

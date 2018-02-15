/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/python/lib/core/py_func.h"

#include <array>

#include "numpy/arrayobject.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

#include <Python.h>

namespace tensorflow {
namespace {

static mutex mu(LINKER_INITIALIZED);
static PyObject* py_trampoline GUARDED_BY(mu) = nullptr;

// Returns the py_trampoline that is used to pass the control to the
// python runtime.
PyObject* GetPyTrampoline() {
  mutex_lock l(mu);
  return py_trampoline;
}

// A call to the registered python function.
struct PyCall {
  // Passed to python runtime to call the python function registered
  // with this "token".
  string token;

  // The device on which Tensors are stored; only used for EagerPyFunc.
  Device* device;

  // True if and only if the op has been placed on a GPU.
  bool gpu;

  // True if the call is associated with an EagerPyFunc.
  bool eager;

  // Inputs and outputs of this function invocation.
  std::vector<Tensor> ins;
  std::vector<Tensor> out;
};

// Givens the 'call', prepares the token and inputs as a python tuple
// that is appropriate for calling the trampoline.
Status MakeArgTuple(const PyCall* call, PyObject** tuple) {
  int64 n = call->ins.size();
  PyObject* lst = PyList_New(n);
  CHECK(lst);
  for (int64 i = 0; i < n; ++i) {
    PyObject* arg = nullptr;
    const Tensor& t = call->ins[i];
    if (call->eager) {
      if (call->gpu) {
        arg = EagerTensorFromHandle(new TFE_TensorHandle(t, call->device));
      } else {
        // TFE_TensorHandle assumes that CPU is identified by `nullptr`.
        arg = EagerTensorFromHandle(new TFE_TensorHandle(t, nullptr));
      }
      if (arg == nullptr) {
        return errors::Internal("Unable to procure EagerTensor from Tensor.");
      }
    } else {
      Status s = ConvertTensorToNdarray(t, &arg);
      if (!s.ok()) {
        Py_DECREF(lst);
        return s;
      }
    }
    PyList_SetItem(lst, i, arg);
  }
  *tuple = Py_BuildValue("(sON)", call->token.c_str(),
                         call->gpu ? Py_True : Py_False, lst);
  CHECK(*tuple);
  return Status::OK();
}

// Returns the corresponding tf dtype in 'tf' for numpy data type
// 'np'.  Returns an error if the type is not supported by this
// module.
Status NumericNpDTypeToTfDType(const int np, DataType* tf) {
  switch (np) {
    case NPY_FLOAT16:
      *tf = DT_HALF;
      break;
    case NPY_FLOAT32:
      *tf = DT_FLOAT;
      break;
    case NPY_FLOAT64:
      *tf = DT_DOUBLE;
      break;
    case NPY_INT32:
      *tf = DT_INT32;
      break;
    case NPY_UINT8:
      *tf = DT_UINT8;
      break;
    case NPY_INT8:
      *tf = DT_INT8;
      break;
    case NPY_INT16:
      *tf = DT_INT16;
      break;
    case NPY_INT64:
      *tf = DT_INT64;
      break;
    case NPY_BOOL:
      *tf = DT_BOOL;
      break;
    case NPY_COMPLEX64:
      *tf = DT_COMPLEX64;
      break;
    case NPY_COMPLEX128:
      *tf = DT_COMPLEX128;
      break;
    default:
      return errors::Unimplemented("Unsupported numpy type ", np);
  }
  return Status::OK();
}

bool IsSingleNone(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    return false;
  }
  PyArrayObject* array_obj = reinterpret_cast<PyArrayObject*>(obj);
  if (PyArray_NDIM(array_obj) != 0 || PyArray_SIZE(array_obj) != 1) {
    return false;
  }
  std::array<npy_intp, 0> indices;
  char* item_ptr =
      static_cast<char*>(PyArray_GetPtr(array_obj, indices.data()));
  PyObject* item = PyArray_GETITEM(array_obj, item_ptr);
  CHECK(item);
  return item == Py_None;
}

// Retrieves a Tensor from `eager_tensor` and stores it in `output_tensor`.
void ExtractTensorFromEagerTensor(const PyObject* eager_tensor,
                                  Tensor* output_tensor) {
  *output_tensor = EagerTensor_Handle(eager_tensor)->t;
}

// Calls the registered py function through the trampoline.
Status DoCallPyFunc(PyCall* call, bool* out_log_on_error) {
  *out_log_on_error = true;
  PyObject* trampoline = GetPyTrampoline();
  if (trampoline == nullptr) {
    return errors::InvalidArgument(
        "Missing py trampoline. Most likely, it is a link error.");
  }
  // Prepare the argument.
  PyObject* args = nullptr;
  TF_RETURN_IF_ERROR(MakeArgTuple(call, &args));
  CHECK(args);

  // Invokes the trampoline.
  PyObject* result = PyEval_CallObject(trampoline, args);
  Py_DECREF(args);
  if (result == nullptr) {
    if (PyErr_Occurred()) {
      if (PyErr_ExceptionMatches(PyExc_ValueError) ||
          PyErr_ExceptionMatches(PyExc_TypeError)) {
        return errors::InvalidArgument(PyExceptionFetch());
      } else if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
        *out_log_on_error = false;
        return errors::OutOfRange(PyExceptionFetch());
      } else if (PyErr_ExceptionMatches(PyExc_MemoryError)) {
        return errors::ResourceExhausted(PyExceptionFetch());
      } else if (PyErr_ExceptionMatches(PyExc_NotImplementedError)) {
        return errors::Unimplemented(PyExceptionFetch());
      } else {
        // TODO(ebrevdo): Check if exception is an OpError and use the
        // OpError.error_code property to map it back in the Status.
        return errors::Unknown(PyExceptionFetch());
      }
    } else {
      return errors::Internal("Failed to run py callback ", call->token,
                              ": see error log.");
    }
  }

  // Process the return values and convert them to TF Tensors.
  Status s = Status::OK();
  if (PyList_Check(result)) {
    // `result` is a Python list; if this operation is an `EagerPyFunc`, then
    // every item in the list must be an `EagerTensor`; otherwise, every element
    // must be a NumPy array.
    call->out.clear();
    for (int i = 0; i < PyList_Size(result); ++i) {
      Tensor t;
      if (call->eager) {
        const PyObject* item = PyList_GetItem(result, i);
        if (EagerTensor_CheckExact(item)) {
          ExtractTensorFromEagerTensor(item, &t);
        } else {
          s = errors::FailedPrecondition(
              "Expected EagerTensor, found PyObject of type: ",
              Py_TYPE(item)->tp_name);
        }
      } else {
        s = ConvertNdarrayToTensor(PyList_GetItem(result, i), &t);
      }

      if (!s.ok()) {
        break;
      }
      call->out.push_back(t);
    }
  } else if (EagerTensor_CheckExact(result) || result == Py_None) {
    // result is an `EagerTensor` or `None`.
    DCHECK(call->eager);
    Tensor t;
    if (result != Py_None) {
      ExtractTensorFromEagerTensor(result, &t);
      call->out.push_back(t);
    }
  } else if (PyArray_Check(result)) {
    // `result` is a NumPy array.
    DCHECK(!call->eager);
    if (!IsSingleNone(result)) {
      Tensor t;
      s = ConvertNdarrayToTensor(result, &t);
      if (s.ok()) {
        call->out.push_back(t);
      }
    }
  } else {
    s = errors::Internal("Unexpected PyObject was returned: ",
                         Py_TYPE(result)->tp_name);
  }
  Py_DECREF(result);
  return s;
}

}  // end namespace

// Outside anonymous namespace just to make the friend declaration in
// tensorflow::Tensor apply.
class NumpyTensorBuffer : public TensorBuffer {
 public:
  NumpyTensorBuffer(PyArrayObject* array, size_t len, void* data)
      : array_(array), len_(len), data_(data) {}

  ~NumpyTensorBuffer() override {
    // Note: The session::run wrapper is responsible for freeing this while
    // holding the GIL.
    DelayedNumpyDecref(data_, len_, array_);
  }

  void* data() const override { return data_; }
  size_t size() const override { return len_; }
  TensorBuffer* root_buffer() override { return this; }
  void FillAllocationDescription(AllocationDescription* proto) const override {
    tensorflow::int64 rb = size();
    proto->set_requested_bytes(rb);
    proto->set_allocator_name(tensorflow::cpu_allocator()->Name());
  }
  Tensor MakeTensor(DataType dtype, const TensorShape& shape) {
    CHECK_EQ(len_, shape.num_elements() * DataTypeSize(dtype));
    return Tensor(dtype, shape, this);
  }

  // Prevents input forwarding from overwriting this buffer.
  bool OwnsMemory() const override { return false; }

 private:
  PyArrayObject* array_;
  size_t len_;
  void* data_;
};

Status ConvertNdarrayToTensor(PyObject* obj, Tensor* ret) {
  PyArrayObject* input = reinterpret_cast<PyArrayObject*>(obj);
  DataType dtype = DT_INVALID;
  TensorShape shape;
  for (int i = 0; i < PyArray_NDIM(input); ++i) {
    shape.AddDim(PyArray_SHAPE(input)[i]);
  }
  const int np_type = PyArray_TYPE(input);
  switch (np_type) {
    case NPY_OBJECT: {
      dtype = DT_STRING;
      Tensor t(dtype, shape);
      auto tflat = t.flat<string>();
      PyObject** input_data = reinterpret_cast<PyObject**>(PyArray_DATA(input));
      for (int i = 0; i < tflat.dimension(0); ++i) {
        char* el;
        Py_ssize_t el_size;
        if (PyBytes_AsStringAndSize(input_data[i], &el, &el_size) == -1) {
#if PY_MAJOR_VERSION >= 3
          el = PyUnicode_AsUTF8AndSize(input_data[i], &el_size);
#else
          el = nullptr;
          if (PyUnicode_Check(input_data[i])) {
            PyObject* unicode = PyUnicode_AsUTF8String(input_data[i]);
            if (unicode) {
              if (PyString_AsStringAndSize(unicode, &el, &el_size) == -1) {
                Py_DECREF(unicode);
                el = nullptr;
              }
            }
          }
#endif
          if (!el) {
            return errors::Unimplemented("Unsupported object type ",
                                         input_data[i]->ob_type->tp_name);
          }
        }
        tflat(i) = string(el, el_size);
      }
      *ret = t;
      break;
    }
    case NPY_STRING: {
      dtype = DT_STRING;
      Tensor t(dtype, shape);
      auto tflat = t.flat<string>();
      char* input_data = PyArray_BYTES(input);
      Py_ssize_t el_size = PyArray_ITEMSIZE(input);
      for (int i = 0; i < tflat.dimension(0); ++i) {
        tflat(i) = string(input_data + i * el_size, el_size);
      }
      *ret = t;
      break;
    }
    default: {
      TF_RETURN_IF_ERROR(NumericNpDTypeToTfDType(PyArray_TYPE(input), &dtype));
      CHECK(DataTypeCanUseMemcpy(dtype));
      if (reinterpret_cast<intptr_t>(PyArray_DATA(input)) %
              EIGEN_MAX_ALIGN_BYTES !=
          0) {
        Tensor t(dtype, shape);
        StringPiece p = t.tensor_data();
        memcpy(const_cast<char*>(p.data()), PyArray_DATA(input), p.size());
        *ret = t;
      } else {
        // Incref the array as the calling context will decref it when we
        // return and we want to keep a handle to this memory.
        Py_INCREF(input);
        NumpyTensorBuffer* buf = new NumpyTensorBuffer(
            input, shape.num_elements() * DataTypeSize(dtype),
            PyArray_DATA(input));
        *ret = buf->MakeTensor(dtype, shape);
        buf->Unref();
      }
    }
  }
  return Status::OK();
}

// Creates a numpy array in 'ret' which either aliases the content of 't' or has
// a copy.
Status ConvertTensorToNdarray(const Tensor& t, PyObject** ret) {
  int typenum = -1;
  TF_RETURN_IF_ERROR(TF_DataType_to_PyArray_TYPE(
      static_cast<TF_DataType>(t.dtype()), &typenum));
  PyArray_Descr* descr = PyArray_DescrFromType(typenum);
  CHECK(descr);
  std::vector<npy_intp> dims;
  dims.reserve(t.dims());
  for (int i = 0; i < t.dims(); ++i) {
    dims.push_back(t.dim_size(i));
  }
  Tensor* copy = new Tensor(t);
  if (ArrayFromMemory(dims.size(), dims.data(),
                      const_cast<char*>(copy->tensor_data().data()), t.dtype(),
                      [copy]() { delete copy; }, ret)
          .ok()) {
    return Status::OK();
  }
  delete copy;

  PyObject* obj = PyArray_Empty(dims.size(), dims.data(), descr, 0);
  if (obj == nullptr) {
    return errors::Internal("Failed to allocate np array: ",
                            t.shape().DebugString());
  }
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  if (typenum == NPY_OBJECT) {
    CHECK_EQ(DT_STRING, t.dtype());
    auto tflat = t.flat<string>();
    PyObject** out = reinterpret_cast<PyObject**>(PyArray_DATA(np_array));
    for (int i = 0; i < tflat.dimension(0); ++i) {
      const string& el = tflat(i);
      out[i] = PyBytes_FromStringAndSize(el.data(), el.size());
      if (out[i] == nullptr) {
        for (int j = 0; j < i; ++j) {
          Py_DECREF(out[j]);
        }
        Py_DECREF(obj);
        return errors::Internal("Failed to allocate a copy of string ", i);
      }
    }
  } else {
    CHECK(DataTypeCanUseMemcpy(t.dtype()));
    StringPiece p = t.tensor_data();
    memcpy(PyArray_DATA(np_array), p.data(), p.size());
  }
  *ret = PyArray_Return(np_array);
  return Status::OK();
}

void InitializePyTrampoline(PyObject* trampoline) {
  mutex_lock l(mu);
  if (py_trampoline == nullptr) {
    py_trampoline = trampoline;
    Py_INCREF(py_trampoline);
  } else {
    LOG(WARNING) << "InitializeCallback should only be called once";
  }
}

class PyFuncOp : public OpKernel {
 public:
  explicit PyFuncOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("token", &token_));
    eager_ = type_string() == "EagerPyFunc";
    gpu_ = ctx->device_type().type_string() == DEVICE_GPU;
  }

  void Compute(OpKernelContext* ctx) override {
    PyCall call;
    call.token = token_;
    call.gpu = gpu_;
    call.eager = eager_;
    if (call.eager) {
      // Eager's C API uses `Device`, whereas `OpKernelContext` stores a
      // `DeviceBase`; attempt to downcast.
      call.device = dynamic_cast<Device*>(ctx->device());
      if (call.device == nullptr) {
        ctx->CtxFailureWithWarning(
            errors::Internal("Unrecognized device class"));
      }
    }

    for (int i = 0; i < ctx->num_inputs(); ++i) {
      call.ins.push_back(ctx->input(i));
    }

    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();
    bool log_on_error;
    Status s = DoCallPyFunc(&call, &log_on_error);
    // Sometimes py_funcs can be called without a session and leak memory. This
    // ensures we clear the decref cache so this doesn't happen.
    ClearDecrefCache();
    PyGILState_Release(py_threadstate);

    // Ensures that GIL is released even when !s.ok().
    if (!s.ok()) {
      if (log_on_error) {
        ctx->CtxFailureWithWarning(s);
      } else {
        ctx->CtxFailure(s);
      }
      return;
    }

    OP_REQUIRES(ctx, static_cast<int32>(call.out.size()) == ctx->num_outputs(),
                errors::InvalidArgument(token_, " returns ", call.out.size(),
                                        " values, but expects to see ",
                                        ctx->num_outputs(), " values."));
    for (size_t i = 0; i < call.out.size(); ++i) {
      const auto& t = call.out[i];
      OP_REQUIRES(
          ctx, t.dtype() == output_type(i),
          errors::InvalidArgument(i, "-th value returned by ", token_, " is ",
                                  DataTypeString(t.dtype()), ", but expects ",
                                  DataTypeString(output_type(i))));
      ctx->set_output(i, t);
    }
  }

 private:
  string token_;

  // True if and only if this op has been placed on a GPU.
  bool gpu_;

  // True if and only if this op should execute the python function eagerly,
  // i.e., if and only if the eager attribute is set.
  bool eager_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyFuncOp);
};

REGISTER_KERNEL_BUILDER(Name("PyFunc").Device(DEVICE_CPU), PyFuncOp);
REGISTER_KERNEL_BUILDER(Name("PyFuncStateless").Device(DEVICE_CPU), PyFuncOp);
REGISTER_KERNEL_BUILDER(Name("EagerPyFunc").Device(DEVICE_CPU), PyFuncOp);
REGISTER_KERNEL_BUILDER(Name("EagerPyFunc").Device(DEVICE_GPU), PyFuncOp);

}  // end namespace tensorflow

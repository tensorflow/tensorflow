/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <Python.h>
#include "numpy/arrayobject.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

// Return type of import_array() changed between Python 2 and 3
// NUMPY_IMPORT_ARRAY_RETVAL is NULL for Python 3
#if PY_MAJOR_VERSION >= 3
#define NUMPY_IMPORT_ARRAY_RETURN_TYPE int
#else
#define NUMPY_IMPORT_ARRAY_RETURN_TYPE void
#endif

namespace tensorflow {
namespace {

static mutex mu;
static bool initialized GUARDED_BY(mu) = false;
static PyObject* py_trampoline GUARDED_BY(mu) = nullptr;

// Returns the py_trampoline that is used to pass the control to the
// python runtime.
PyObject* GetPyTrampoline() {
  mutex_lock l(mu);
  return py_trampoline;
}

// Module initialization (mainly import numpy) if needed.
NUMPY_IMPORT_ARRAY_RETURN_TYPE InitIfNeeded() {
  mutex_lock l(mu);
  if (!initialized) {
    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();
    import_array();
    PyGILState_Release(py_threadstate);
    initialized = true;
  }
}

// Returns a single-thread threadpool used to execute python
// trampoline and the python function. It is single threaded because
// GIL is needed running the trampoline.
thread::ThreadPool* py_thread() {
  static thread::ThreadPool* w =
      new thread::ThreadPool(Env::Default(), "PyTrampoline", 1);
  return w;
}

// Returns the corresponding numpy dtype in 'np' for tf data type
// 'tf'.  Returns an error if the type is not supported by this
// module.
Status TfDTypeToNpDType(const DataType& tf, int* np) {
  switch (tf) {
    case DT_FLOAT:
      *np = NPY_FLOAT32;
      break;
    case DT_DOUBLE:
      *np = NPY_FLOAT64;
      break;
    case DT_INT32:
      *np = NPY_INT32;
      break;
    case DT_UINT8:
      *np = NPY_UINT8;
      break;
    case DT_INT8:
      *np = NPY_INT8;
      break;
    case DT_INT16:
      *np = NPY_INT16;
      break;
    case DT_INT64:
      *np = NPY_INT64;
      break;
    case DT_BOOL:
      *np = NPY_BOOL;
      break;
    case DT_COMPLEX64:
      *np = NPY_COMPLEX64;
      break;
    case DT_COMPLEX128:
      *np = NPY_COMPLEX128;
      break;
    case DT_STRING:
      *np = NPY_OBJECT;
      break;
    default:
      return errors::Unimplemented("Unsupported tf type ", DataTypeString(tf));
  }
  return Status::OK();
}

// A call to the registered python function.
struct PyCall {
  // Passed to python runtime to call the python function registered
  // with this "token".
  string token;

  // Inputs and outputs of this function invokation.
  std::vector<Tensor> ins;
  std::vector<Tensor> out;
};

// Givens the 'call', prepares the token and inputs as a python tuple
// that is appropriate for calling the trampoline.
Status MakeArgTuple(PyCall* call, PyObject** tuple) {
  int64 n = call->ins.size();
  PyObject* lst = PyList_New(n);
  CHECK(lst);
  for (int64 i = 0; i < n; ++i) {
    const Tensor& t = call->ins[i];
    PyObject* a = nullptr;
    Status s = ConvertTensorToNdarray(t, &a);
    if (!s.ok()) {
      Py_DECREF(lst);
      return s;
    }
    PyList_SetItem(lst, i, a);
  }
  *tuple = Py_BuildValue("(sN)", call->token.c_str(), lst);
  CHECK(*tuple);
  return Status::OK();
}

// Returns the corresponding tf dtype in 'tf' for numpy data type
// 'np'.  Returns an error if the type is not supported by this
// module.
Status NumericNpDTypeToTfDType(const int np, DataType* tf) {
  switch (np) {
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

// Given an numpy ndarray object 'obj', creates a corresponding tf
// Tensor in '*ret'.
Status ConvertNdarrayToTensor(PyObject* obj, Tensor* ret) {
  PyArrayObject* input = reinterpret_cast<PyArrayObject*>(obj);
  DataType dtype;
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
          return errors::Unimplemented("Unsupported object type ",
                                       input_data[i]->ob_type->tp_name);
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
      Tensor t(dtype, shape);
      CHECK(DataTypeCanUseMemcpy(dtype));
      StringPiece p = t.tensor_data();
      memcpy(const_cast<char*>(p.data()), input->data, p.size());
      *ret = t;
    }
  }
  return Status::OK();
}

// Calls the registered py function through the trampoline.
Status DoCallPyFunc(PyCall* call) {
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
      // TODO(zhifengc): Consider pretty-print error using LOG(STDERR).
      PyErr_Print();
    }
    return errors::Internal("Failed to run py callback ", call->token,
                            ": see error log.");
  }

  // Process the return values and converts them to tf Tensors.
  Status s;
  if (PyList_Check(result)) {
    // 'result' is a list.
    call->out.clear();
    for (int i = 0; i < PyList_Size(result); ++i) {
      Tensor t;
      s = ConvertNdarrayToTensor(PyList_GetItem(result, i), &t);
      if (!s.ok()) {
        break;
      }
      call->out.push_back(t);
    }
  } else if (PyArray_Check(result)) {
    // 'result' is a single ndarray.
    Tensor t;
    s = ConvertNdarrayToTensor(result, &t);
    if (s.ok()) {
      call->out.push_back(t);
    }
  } else {
    s = errors::Internal("Unexpected pyobject is returned: ",
                         Py_TYPE(result)->tp_name);
  }
  Py_DECREF(result);
  return s;
}

// Calls the python function in a separate thread. Arranges to call
// done() when the python function returns.
void CallPyFunc(PyCall* call, std::function<void(Status)> done) {
  InitIfNeeded();
  py_thread()->Schedule([call, done]() {
    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();
    Status s = DoCallPyFunc(call);
    PyGILState_Release(py_threadstate);
    done(s);
  });
}

}  // end namespace

// Creates a numpy array in 'ret' and copies the content of tensor 't'
// into 'ret'.
Status ConvertTensorToNdarray(const Tensor& t, PyObject** ret) {
  InitIfNeeded();
  int typenum = -1;
  TF_RETURN_IF_ERROR(TfDTypeToNpDType(t.dtype(), &typenum));
  PyArray_Descr* descr = PyArray_DescrFromType(typenum);
  CHECK(descr);
  std::vector<npy_intp> dims;
  for (int i = 0; i < t.dims(); ++i) {
    dims.push_back(t.dim_size(i));
  }
  PyObject* obj = PyArray_Empty(dims.size(), dims.data(), descr, 0);
  if (obj == nullptr) {
    return errors::Internal("Failed to allocate np array: ",
                            t.shape().DebugString());
  }
  PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(obj);
  if (typenum == NPY_OBJECT) {
    CHECK_EQ(DT_STRING, t.dtype());
    auto tflat = t.flat<string>();
    PyObject** out = reinterpret_cast<PyObject**>(np_array->data);
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
    memcpy(np_array->data, p.data(), p.size());
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

class PyFuncOp : public AsyncOpKernel {
 public:
  explicit PyFuncOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("token", &token_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    PyCall* call = new PyCall;
    call->token = token_;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      call->ins.push_back(ctx->input(i));
    }
    CallPyFunc(call, [this, ctx, call, done](Status s) {
      std::unique_ptr<PyCall> delete_me(call);
      OP_REQUIRES_OK_ASYNC(ctx, s, done);
      OP_REQUIRES_ASYNC(
          ctx, static_cast<int32>(call->out.size()) == ctx->num_outputs(),
          errors::InvalidArgument(token_, " returns ", call->out.size(),
                                  " values, but expects to see ",
                                  ctx->num_outputs(), " values."),
          done);
      for (size_t i = 0; i < call->out.size(); ++i) {
        const auto& t = call->out[i];
        OP_REQUIRES_ASYNC(
            ctx, t.dtype() == output_type(i),
            errors::InvalidArgument(i, "-th value returned by ", token_, " is ",
                                    DataTypeString(t.dtype()), ", but expects ",
                                    DataTypeString(output_type(i))),
            done);
        ctx->set_output(i, t);
      }
      done();
    });
  }

 private:
  string token_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyFuncOp);
};
REGISTER_KERNEL_BUILDER(Name("PyFunc").Device(DEVICE_CPU), PyFuncOp);

}  // end namespace tensorflow

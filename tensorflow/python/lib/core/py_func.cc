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
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include <Python.h>

namespace tensorflow {
namespace {

static mutex mu;
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

  // Inputs and outputs of this function invocation.
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

bool IsSingleNone(PyObject* obj) {
  if (!PyArray_Check(obj)) {
    return false;
  }
  PyArrayObject* array_obj = reinterpret_cast<PyArrayObject*>(obj);
  if (PyArray_NDIM(array_obj) != 0 || PyArray_SIZE(array_obj) != 1) {
    return false;
  }
  std::array<npy_intp, 0> indices;
  char* item_ptr = static_cast<char*>(PyArray_GetPtr(array_obj, indices.data()));
  PyObject* item = PyArray_GETITEM(array_obj, item_ptr);
  CHECK(item);
  return item == Py_None;
}

// py.__class__.__name__
const char* ClassName(PyObject* py) {
/* PyPy doesn't have a separate C API for old-style classes. */
#if PY_MAJOR_VERSION < 3 && !defined(PYPY_VERSION)
  if (PyClass_Check(py))
    return PyString_AS_STRING(
        CHECK_NOTNULL(reinterpret_cast<PyClassObject*>(py)->cl_name));
  if (PyInstance_Check(py))
    return PyString_AS_STRING(CHECK_NOTNULL(
        reinterpret_cast<PyInstanceObject*>(py)->in_class->cl_name));
#endif
  if (Py_TYPE(py) == &PyType_Type) {
    return reinterpret_cast<PyTypeObject*>(py)->tp_name;
  }
  return Py_TYPE(py)->tp_name;
}

string PyExcFetch() {
  CHECK(PyErr_Occurred()) << "Must only call PyExcFetch after an exception.";
  PyObject* ptype;
  PyObject* pvalue;
  PyObject* ptraceback;
  PyErr_Fetch(&ptype, &pvalue, &ptraceback);
  PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
  string err = ClassName(ptype);
  if (pvalue) {
    PyObject* str = PyObject_Str(pvalue);
    if (str) {
#if PY_MAJOR_VERSION < 3
      strings::StrAppend(&err, ": ", PyString_AS_STRING(str));
#else
      strings::StrAppend(&err, ": ", PyUnicode_AsUTF8(str));
#endif
      Py_DECREF(str);
    }
    Py_DECREF(pvalue);
  }
  Py_DECREF(ptype);
  Py_XDECREF(ptraceback);
  return err;
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
      if (PyErr_ExceptionMatches(PyExc_ValueError) ||
          PyErr_ExceptionMatches(PyExc_TypeError)) {
        return errors::InvalidArgument(PyExcFetch());
      } else if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
        return errors::OutOfRange(PyExcFetch());
      } else if (PyErr_ExceptionMatches(PyExc_MemoryError)) {
        return errors::ResourceExhausted(PyExcFetch());
      } else if (PyErr_ExceptionMatches(PyExc_NotImplementedError)) {
        return errors::Unimplemented(PyExcFetch());
      } else {
        // TODO(ebrevdo): Check if exception is an OpError and use the
        // OpError.error_code property to map it back in the Status.
        return errors::Unknown(PyExcFetch());
      }
    } else {
      return errors::Internal("Failed to run py callback ", call->token,
                              ": see error log.");
    }
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
    if (!IsSingleNone(result)) {
      Tensor t;
      s = ConvertNdarrayToTensor(result, &t);
      if (s.ok()) {
        call->out.push_back(t);
      }
    }
  } else {
    s = errors::Internal("Unexpected pyobject is returned: ",
                         Py_TYPE(result)->tp_name);
  }
  Py_DECREF(result);
  return s;
}

}  // end namespace

Status TF_TensorToTensor(const TF_Tensor* src, Tensor* dst);
TF_Tensor* TF_TensorFromTensor(const tensorflow::Tensor& src,
                               TF_Status* status);

Status ConvertNdarrayToTensor(PyObject* obj, Tensor* ret) {
  Safe_TF_TensorPtr tf_tensor = make_safe(static_cast<TF_Tensor*>(nullptr));
  Status s = PyArrayToTF_Tensor(obj, &tf_tensor);
  if (!s.ok()) {
    return s;
  }
  return TF_TensorToTensor(tf_tensor.get(), ret);
}

// Creates a numpy array in 'ret' which either aliases the content of 't' or has
// a copy.
Status ConvertTensorToNdarray(const Tensor& t, PyObject** ret) {
  TF_Status* status = TF_NewStatus();
  Safe_TF_TensorPtr tf_tensor = make_safe(TF_TensorFromTensor(t, status));
  Status tf_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  if (!tf_status.ok()) {
    return tf_status;
  }
  return TF_TensorToPyArray(std::move(tf_tensor), ret);
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
  }

  void Compute(OpKernelContext* ctx) override {
    PyCall call;
    call.token = token_;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      call.ins.push_back(ctx->input(i));
    }

    PyGILState_STATE py_threadstate;
    py_threadstate = PyGILState_Ensure();
    Status s = DoCallPyFunc(&call);
    PyGILState_Release(py_threadstate);

    // Ensures that GIL is released even when !s.ok().
    OP_REQUIRES_OK(ctx, s);

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

  TF_DISALLOW_COPY_AND_ASSIGN(PyFuncOp);
};
REGISTER_KERNEL_BUILDER(Name("PyFunc").Device(DEVICE_CPU), PyFuncOp);
REGISTER_KERNEL_BUILDER(Name("PyFuncStateless").Device(DEVICE_CPU), PyFuncOp);

}  // end namespace tensorflow

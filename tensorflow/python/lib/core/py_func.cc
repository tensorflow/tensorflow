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

#include <Python.h>

// clang-format: off
// Must be included first.
#include "tensorflow/python/lib/core/numpy.h"
// clang-format: on

#include <array>

#include "numpy/arrayobject.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/ndarray_tensor.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"
#include "tensorflow/python/lib/core/py_util.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace tensorflow {
namespace {

static mutex mu(LINKER_INITIALIZED);
static PyObject* py_trampoline TF_GUARDED_BY(mu) = nullptr;

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
  Device* device = nullptr;

  // True if the call is associated with an EagerPyFunc.
  bool eager = false;

  // True if the call is running under eager async mode.
  bool eager_async = false;

  // Inputs and outputs of this function invocation.
  std::vector<Tensor> ins;
  std::vector<Tensor> out;
};

bool IsCPUDevice(const Device* d) {
  return d == nullptr || d->tensorflow_accelerator_device_info() == nullptr;
}

// Givens the 'call', prepares the token and inputs as a python tuple
// that is appropriate for calling the trampoline.
Status MakeArgTuple(const PyCall* call, TFE_Context* ctx, PyObject** tuple) {
  int64_t n = call->ins.size();
  PyObject* lst = PyList_New(n);
  CHECK(lst);
  // TFE_TensorHandle assumes that CPU is identified by nullptr.
  //
  // Set device name to be empty if the device is CPU.
  const char* device_name = nullptr;

  if (call->device != nullptr && !IsCPUDevice(call->device))
    device_name = call->device->name().c_str();

  for (int64_t i = 0; i < n; ++i) {
    PyObject* arg = nullptr;
    if (call->eager) {
      Tensor t = call->ins[i];
      arg = EagerTensorFromHandle(tensorflow::wrap(
          tensorflow::unwrap(ctx)->CreateLocalHandleFromTFTensor(t,
                                                                 device_name)));
      if (arg == nullptr) {
        Py_DECREF(lst);
        return errors::Internal("Unable to procure EagerTensor from Tensor.");
      }
    } else {
      Status s = TensorToNdarray(call->ins[i], &arg);
      if (!s.ok()) {
        Py_DECREF(lst);
        return s;
      }
      arg = PyArray_Return(reinterpret_cast<PyArrayObject*>(arg));
    }
    PyList_SetItem(lst, i, arg);
  }
  *tuple = Py_BuildValue("(ssN)", call->token.c_str(), device_name, lst);
  CHECK(*tuple);
  return OkStatus();
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
// Validates that `output_tensor` is backed by memory in `expected_device`
// (which is assumed to be a local device, one on which the kernel was
// executed.)
//
// It may be nice to copy the tensor to the right device instead of failing if
// it isn't already there. This is left as a future exercise.  The required
// device-copying logic is implemented in Python at the moment.
tensorflow::Status ExtractTensorFromEagerTensor(const PyObject* eager_tensor,
                                                TFE_Context* ctx,
                                                const Device* expected_device,
                                                const Tensor** output_tensor) {
  tensorflow::TensorHandle* handle = down_cast<tensorflow::TensorHandle*>(
      tensorflow::unwrap(ctx)->TFTensorHandleFromInterface(
          tensorflow::unwrap(EagerTensor_Handle(eager_tensor))));

  Device* actual_device = handle->device();
  TF_RETURN_IF_ERROR(handle->Tensor(output_tensor));
  // actual_device may be nullptr, which implies local CPU.
  if (expected_device == actual_device) return OkStatus();
  const string& expected_device_name = expected_device->attributes().name();
  if (actual_device == nullptr) {
    if (!IsCPUDevice(expected_device)) {
      return errors::Internal(
          "Expected the py_func to return a Tensor backed by memory in ",
          expected_device_name,
          ", but is actually backed by local host memory. This is a bug.");
    }
    return OkStatus();
  }
  // NOTE(ebrevdo): Here we could try comparing "actual_device_name"
  // (actual_device->attributes()->name()) to expected_device_name and ensure
  // they're the same.  However, this comparison fails if we create a ClusterDef
  // on localhost, mainly because the Device created by Eager code doesn't match
  // the device created by a session.  In this case, expected_device_name may
  // contain "worker" but the Eager device name contains "localhost".  Since we
  // can't easily access the true underlying device of "worker" here, we are not
  // able to perform a proper comparison.  Furthermore, we can't check
  // IsCPUDevice(actual_device) because the kernel's device may indeed be a
  // GPU device (the python interpreter doesn't use it, however).
  return OkStatus();
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
  std::unique_ptr<EagerExecutor> new_executor = nullptr;
  EagerExecutor* old_executor = nullptr;
  if (call->eager) {
    // See FuncRegistry._ctx.
    TFE_Context* ctx = reinterpret_cast<TFE_Context*>(PyCapsule_GetPointer(
        PyObject_GetAttrString(trampoline, "_ctx"), nullptr));
    CHECK_NE(ctx, nullptr);
    TF_RETURN_IF_ERROR(MakeArgTuple(call, ctx, &args));
    new_executor.reset(new EagerExecutor(call->eager_async));
    old_executor = &(tensorflow::unwrap(ctx)->Executor());
    tensorflow::unwrap(ctx)->SetExecutorForThread(new_executor.get());
  } else {
    TF_RETURN_IF_ERROR(MakeArgTuple(call, nullptr, &args));
  }
  CHECK(args);

  // Invokes the trampoline.
  PyObject* result = PyEval_CallObject(trampoline, args);
  Py_DECREF(args);
  Status s = OkStatus();
  if (result == nullptr) {
    if (PyErr_Occurred()) {
      if (PyErr_ExceptionMatches(PyExc_ValueError) ||
          PyErr_ExceptionMatches(PyExc_TypeError)) {
        s = errors::InvalidArgument(PyExceptionFetch());
      } else if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
        *out_log_on_error = false;
        s = errors::OutOfRange(PyExceptionFetch());
      } else if (PyErr_ExceptionMatches(PyExc_MemoryError)) {
        s = errors::ResourceExhausted(PyExceptionFetch());
      } else if (PyErr_ExceptionMatches(PyExc_NotImplementedError)) {
        s = errors::Unimplemented(PyExceptionFetch());
      } else {
        // TODO(ebrevdo): Check if exception is an OpError and use the
        // OpError.error_code property to map it back in the Status.
        s = errors::Unknown(PyExceptionFetch());
      }
    } else {
      s = errors::Internal("Failed to run py callback ", call->token,
                           ": see error log.");
    }
  }

  TFE_Context* ctx = reinterpret_cast<TFE_Context*>(PyCapsule_GetPointer(
      PyObject_GetAttrString(trampoline, "_ctx"), /*name=*/nullptr));
  if (new_executor != nullptr) {
    s.Update(new_executor->WaitForAllPendingNodes());
    tensorflow::unwrap(ctx)->SetExecutorForThread(old_executor);
  }

  TF_RETURN_IF_ERROR(s);

  // Process the return values and convert them to TF Tensors.
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
          const Tensor* tensor = nullptr;
          s = ExtractTensorFromEagerTensor(item, ctx, call->device, &tensor);
          if (s.ok()) t = *tensor;
        } else {
          s = errors::FailedPrecondition(
              "Expected EagerTensor, found PyObject of type: ",
              Py_TYPE(item)->tp_name);
        }
      } else {
        s = NdarrayToTensor(PyList_GetItem(result, i), &t);
      }

      if (!s.ok()) {
        break;
      }
      call->out.push_back(t);
    }
  } else if (EagerTensor_CheckExact(result) || result == Py_None) {
    // result is an `EagerTensor` or `None`.
    DCHECK(call->eager);
    if (result != Py_None) {
      const Tensor* t = nullptr;
      s = ExtractTensorFromEagerTensor(result, ctx, call->device, &t);
      if (s.ok()) call->out.push_back(*t);
    }
  } else if (PyArray_Check(result)) {
    // `result` is a NumPy array.
    DCHECK(!call->eager);
    if (!IsSingleNone(result)) {
      Tensor t;
      s = NdarrayToTensor(result, &t);
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
    if (eager_) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("is_async", &eager_async_));
    }
  }

  bool IsExpensive() override { return true; }

  void Compute(OpKernelContext* ctx) override {
    PyCall call;
    call.token = token_;
    call.eager = eager_;
    if (call.eager) {
      // Eager's C API uses `Device`, whereas `OpKernelContext` stores a
      // `DeviceBase`; attempt to downcast.
      call.device = dynamic_cast<Device*>(ctx->device());
      if (call.device == nullptr) {
        ctx->CtxFailureWithWarning(errors::Internal(
            "Unrecognized device class: ", ctx->device()->name()));
        return;
      }
      call.eager_async = eager_async_;
    }

    for (int i = 0; i < ctx->num_inputs(); ++i) {
      call.ins.push_back(ctx->input(i));
    }

    // NOTE(mrry): There is a potential time-of-check-to-time-of-use race here.
    // because it is possible that `Py_Finalize()` could be called in another
    // thread between this check and the  call to `PyGILState_Ensure()`, which
    // will abort the process if `Py_Finalize()` has been called. A more robust
    // solution would be welcome, but it is not obvious how to make this work
    // using the current Python C API.
    OP_REQUIRES(ctx, Py_IsInitialized(),
                errors::FailedPrecondition(
                    "Python interpreter state is not initialized. "
                    "The process may be terminated."));

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

  // True if and only if this op should execute the python function eagerly,
  // i.e., if and only if the eager attribute is set.
  bool eager_;

  bool eager_async_;

  TF_DISALLOW_COPY_AND_ASSIGN(PyFuncOp);
};

REGISTER_KERNEL_BUILDER(Name("PyFunc").Device(DEVICE_CPU), PyFuncOp);
REGISTER_KERNEL_BUILDER(Name("PyFuncStateless").Device(DEVICE_CPU), PyFuncOp);
REGISTER_KERNEL_BUILDER(Name("EagerPyFunc").Device(DEVICE_CPU), PyFuncOp);

DataType gpu_types[] = {
    // No strings and int32s, no ref types and no resource/variant types.
    DT_FLOAT,      DT_DOUBLE,   DT_UINT8,  DT_INT16,   DT_INT8,
    DT_COMPLEX64,  DT_INT64,    DT_BOOL,   DT_QINT8,   DT_QUINT8,
    DT_QINT32,     DT_BFLOAT16, DT_QINT16, DT_QUINT16, DT_UINT16,
    DT_COMPLEX128, DT_HALF,     DT_UINT32, DT_UINT64,
};

REGISTER_KERNEL_BUILDER(Name("EagerPyFunc")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint("Tin", gpu_types)
                            .TypeConstraint("Tout", gpu_types),
                        PyFuncOp);

}  // end namespace tensorflow

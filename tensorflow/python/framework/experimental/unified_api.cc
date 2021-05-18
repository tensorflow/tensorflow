/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <pybind11/stl.h>

#include <memory>

#include "pybind11/pybind11.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/c/eager/abstract_operation.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"
#include "tensorflow/c/eager/c_api_unified_experimental_internal.h"
#include "tensorflow/c/eager/immediate_execution_context.h"
#include "tensorflow/c/eager/immediate_execution_tensor_handle.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/llvm_rtti/llvm_rtti.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_ptr.h"

namespace py = pybind11;

using tensorflow::AbstractContext;
using tensorflow::AbstractContextPtr;
using tensorflow::AbstractFunction;
using tensorflow::AbstractOperation;
using tensorflow::AbstractOperationPtr;
using tensorflow::AbstractTensorHandle;
using tensorflow::AbstractTensorHandlePtr;
using tensorflow::OutputList;

using tensorflow::tracing::TracingContext;
using tensorflow::tracing::TracingOperation;
using tensorflow::tracing::TracingTensorHandle;

using tensorflow::ImmediateContextPtr;
using tensorflow::ImmediateExecutionContext;
using tensorflow::ImmediateExecutionTensorHandle;

using tensorflow::dyn_cast;
using tensorflow::isa;
using tensorflow::unwrap;
using tensorflow::wrap;

using tensorflow::DataType;
using tensorflow::make_safe;
using tensorflow::MaybeRaiseRegisteredFromStatus;
using tensorflow::MaybeRaiseRegisteredFromTFStatus;
using tensorflow::Pyo;
using tensorflow::Safe_TF_StatusPtr;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::TFE_TensorHandleToNumpy;
using tensorflow::core::RefCountPtr;

using tensorflow::errors::Internal;
using tensorflow::errors::InvalidArgument;

PYBIND11_MODULE(_unified_api, m) {
  // Context creation functions.
  m.def("SetTracingImplementation", [](const char* impl) {
    Safe_TF_StatusPtr status = make_safe(TF_NewStatus());
    TF_SetTracingImplementation(impl, status.get());
    MaybeRaiseRegisteredFromStatus(status->status);
  });
  m.def("NewTracingContext", [](const char* fn_name) {
    Safe_TF_StatusPtr status = make_safe(TF_NewStatus());
    auto* ctx = unwrap(TF_CreateFunction(fn_name, status.get()));
    MaybeRaiseRegisteredFromTFStatus(status.get());
    if (!ctx) {
      MaybeRaiseRegisteredFromStatus(
          Internal("TF_CreateFunction returned nullptr"));
    }
    if (!isa<TracingContext>(ctx)) {
      // TODO(srbs): Add a helper to convert the kind enum to a user-friendly
      // string.
      MaybeRaiseRegisteredFromStatus(
          Internal("TF_CreateFunction must return a TracingContext, found ",
                   ctx->getKind()));
    }
    return dyn_cast<TracingContext>(ctx);
  });
  m.def("EagerContextToImmediateExecutionContext", [](py::handle& obj) {
    TFE_Context* ctx =
        static_cast<TFE_Context*>(PyCapsule_GetPointer(obj.ptr(), nullptr));
    if (!ctx) {
      MaybeRaiseRegisteredFromStatus(InvalidArgument("TFE_Context is nullptr"));
    }
    return unwrap(ctx);
  });

  // Unified execution context.
  py::class_<AbstractContext, AbstractContextPtr>(m, "AbstractContext")
      .def("CreateOperation",
           [](AbstractContext* self, const char* op,
              const char* raw_device_name) {
             auto operation = self->CreateOperation();
             operation->Reset(op, raw_device_name);
             return operation;
           })
      .def("RegisterFunction",
           [](AbstractContext* self, AbstractFunction* f) {
             Status s = self->RegisterFunction(f);
             MaybeRaiseRegisteredFromStatus(s);
           })
      .def("RemoveFunction", [](AbstractContext* self, const string& func) {
        Status s = self->RemoveFunction(func);
        MaybeRaiseRegisteredFromStatus(s);
      });

  py::class_<TracingContext, AbstractContext>(m, "TracingContext")
      .def("AddParameter",
           [](TracingContext* self, DataType dtype) {
             TracingTensorHandle* handle = nullptr;
             // TODO(srbs): Add shape argument to this function.
             tensorflow::PartialTensorShape shape;
             Status s = self->AddParameter(dtype, shape, &handle);
             MaybeRaiseRegisteredFromStatus(s);
             return static_cast<AbstractTensorHandle*>(handle);
           })
      .def("Finalize", [](TracingContext* self, py::handle& outputs) {
        // TODO(srbs): Using OutputList seems like an overkill here. Should we
        // simply pass in an absl::Span?
        OutputList output_list;
        if (outputs.ptr() != Py_None) {
          if (!PyList_Check(outputs.ptr())) {
            MaybeRaiseRegisteredFromStatus(
                InvalidArgument("must provide a list of Tensors as inputs"));
          }
          Py_ssize_t len = PyList_Size(outputs.ptr());
          output_list.outputs.resize(len);
          for (Py_ssize_t i = 0; i < len; ++i) {
            PyObject* elem = PyList_GetItem(outputs.ptr(), i);
            if (!elem) {
              MaybeRaiseRegisteredFromStatus(
                  InvalidArgument("Tensor at index  ", i, " is None."));
            }
            py::handle elem_h = elem;
            AbstractTensorHandle* handle = elem_h.cast<AbstractTensorHandle*>();
            if (!isa<TracingTensorHandle>(handle)) {
              MaybeRaiseRegisteredFromStatus(InvalidArgument(
                  "Tensor at index  ", i, " is not a graph tensor."));
            }
            output_list.outputs[i] = handle;
          }
        }
        AbstractFunction* f = nullptr;
        Status s = self->Finalize(&output_list, &f);
        MaybeRaiseRegisteredFromStatus(s);
        return f;
      });

  // Note: This does not take ownership of the C++ context, the lifetime of
  // which is managed by the python `Context` and is expected to outlive this
  // object.
  // TODO(srbs): Make AbstractContext refcounted so that the above comment is
  // not needed.
  py::class_<ImmediateExecutionContext, AbstractContext,
             std::unique_ptr<ImmediateExecutionContext, py::nodelete>>
      ImmediateExecutionContext(m, "ImmediateExecutionContext");

  // Unified execution operation.
  py::class_<AbstractOperation, AbstractOperationPtr>(m, "AbstractOperation")
      .def("Reset",
           [](AbstractOperation* self, const char* op,
              const char* raw_device_name) {
             Status s = self->Reset(op, raw_device_name);
             MaybeRaiseRegisteredFromStatus(s);
           })
      .def("SetOpName",
           [](AbstractOperation* self, const char* op_name) {
             // TODO(srbs): We could provide SetOpName on TracingOperation
             // but then we need to do a hasattr check or try/pass in python.
             if (isa<TracingOperation>(self)) {
               auto tracing_op = reinterpret_cast<TracingOperation*>(self);
               Status s = tracing_op->SetOpName(op_name);
               MaybeRaiseRegisteredFromStatus(s);
             }
           })
      .def("Name", &AbstractOperation::Name)
      .def("DeviceName", &AbstractOperation::DeviceName)
      .def("SetDeviceName",
           [](AbstractOperation* self, const char* name) {
             Status s = self->SetDeviceName(name);
             MaybeRaiseRegisteredFromStatus(s);
           })
      .def("AddInput",
           [](AbstractOperation* self, AbstractTensorHandle* input) {
             Status s = self->AddInput(input);
             MaybeRaiseRegisteredFromStatus(s);
           })
      .def("SetAttrType",
           [](AbstractOperation* self, const char* attr_name, DataType value) {
             Status s = self->SetAttrType(attr_name, value);
             MaybeRaiseRegisteredFromStatus(s);
           })
      .def("Execute", [](AbstractOperation* self, int num_outputs) {
        std::vector<AbstractTensorHandle*> outputs(num_outputs);
        MaybeRaiseRegisteredFromStatus(
            self->Execute(absl::MakeSpan(outputs), &num_outputs));
        return outputs;
      });

  // Unified execution tensor handle.
  py::class_<AbstractTensorHandle, AbstractTensorHandlePtr>(
      m, "AbstractTensorHandle")
      .def("DataType", &AbstractTensorHandle::DataType)
      .def("numpy", [](AbstractTensorHandle* self) {
        // TODO(srbs): Export this on ImmediateExecutionTensorHandle only.
        if (!isa<ImmediateExecutionTensorHandle>(self)) {
          // TODO(srbs): Add a helper to convert the kind enum to a
          // user-friendly string.
          MaybeRaiseRegisteredFromStatus(Internal(
              "AbstractTensorHandle.numpy() must be called with an ",
              "ImmediateExecutionTensorHandle found type: ", self->getKind()));
        }
        TF_Status s;
        TFE_TensorHandle* handle =
            wrap(dyn_cast<ImmediateExecutionTensorHandle>(self));
        auto result = TFE_TensorHandleToNumpy(handle, &s);
        MaybeRaiseRegisteredFromStatus(s.status);
        return Pyo(result);
      });

  m.def("EagerTensorToImmediateExecutionTensorHandle", [](py::object handle) {
    if (!EagerTensor_CheckExact(handle.ptr())) {
      MaybeRaiseRegisteredFromStatus(
          InvalidArgument("EagerTensorToImmediateExecutionTensorHandle called "
                          "with non-EagerTensor."));
    }
    TFE_TensorHandle* eager_tensor = EagerTensor_Handle(handle.ptr());
    auto t = static_cast<AbstractTensorHandle*>(unwrap(eager_tensor));
    t->Ref();
    return t;
  });

  py::class_<AbstractFunction, RefCountPtr<AbstractFunction>> AbstractFunction(
      m, "AbstractFunction");
}

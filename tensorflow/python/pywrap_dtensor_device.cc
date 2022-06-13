/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/dtensor/cc/dtensor_device.h"
#include "tensorflow/python/eager/pywrap_tensor.h"
#include "tensorflow/python/eager/pywrap_tfe.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/lib/core/pybind11_status.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/util.h"

namespace py = ::pybind11;
using tensorflow::dtensor::AddMesh;
using tensorflow::dtensor::AllocateDTensorDevice;
using tensorflow::dtensor::ClearTPUCoreIDs;
using tensorflow::dtensor::ExperimentalClearDefaultLayout;
using tensorflow::dtensor::ExperimentalClearDefaultMesh;
using tensorflow::dtensor::ExperimentalSetDefaultLayout;
using tensorflow::dtensor::ExperimentalSetDefaultMesh;
using tensorflow::dtensor::FetchLayout;
using tensorflow::dtensor::IsSparseDTensor;
using tensorflow::dtensor::Pack;
using tensorflow::dtensor::SetSameShapePolicy;
using tensorflow::dtensor::SetTPUCoreIDs;
using tensorflow::dtensor::SparsePack;
using tensorflow::dtensor::TPUCoreIDsToLocations;
using tensorflow::dtensor::TPUCoreLocationsToIDs;
using tensorflow::dtensor::Unpack;

void PyXDecref(PyObject* obj) { Py_XDECREF(obj); }

void CallDelete_Device(PyObject* capsule) {
  delete reinterpret_cast<TFE_CustomDevice*>(
      PyCapsule_GetPointer(capsule, "TFE_CustomDevice"));
}

void CallDelete_DeviceInfo(PyObject* capsule) {
  void (*destructor)(void*) =
      reinterpret_cast<void (*)(void*)>(PyCapsule_GetContext(capsule));
  destructor(PyCapsule_GetPointer(capsule, "TFE_CustomDevice_DeviceInfo"));
}

// Supports 2 cases:
//  i) input is an EagerTensor.
//  ii) input is an arbitrary python list/tuple.
void ConvertToTensor(TFE_Context* ctx, PyObject* input,
                     tensorflow::Safe_PyObjectPtr* output_handle,
                     TF_Status* status) {
  if (EagerTensor_CheckExact(input)) {
    // Input is already a EagerTensor so increment the reference, since the
    // caller will use it through output_handle.
    Py_INCREF(input);
    output_handle->reset(input);
    return;
  }
  TFE_TensorHandle* handle =
      tensorflow::ConvertToEagerTensor(ctx, input, tensorflow::DT_INVALID);
  if (handle == nullptr) {
    TF_SetStatus(status, TF_INTERNAL, "Failure converting to eager tensor.");
    return;
  }
  output_handle->reset(EagerTensorFromHandle(handle));
}

PYBIND11_MODULE(_pywrap_dtensor_device, m) {
  m.def("Allocate", [](const std::string& name) {
    TFE_CustomDevice* device = new TFE_CustomDevice;
    std::unique_ptr<PyObject, decltype(&PyXDecref)> device_capsule(
        PyCapsule_New(device, "TFE_CustomDevice", &CallDelete_Device),
        PyXDecref);
    void* device_info;
    AllocateDTensorDevice(name, device, &device_info);
    std::unique_ptr<PyObject, decltype(&PyXDecref)> device_info_capsule(
        PyCapsule_New(device_info, "TFE_CustomDevice_DeviceInfo",
                      &CallDelete_DeviceInfo),
        PyXDecref);
    // The PyCapsule destructor needs a pointer to the destructor for
    // DeviceInfo.
    PyCapsule_SetContext(device_info_capsule.get(),
                         reinterpret_cast<void*>(device->delete_device));
    if (PyErr_Occurred()) throw py::error_already_set();
    return pybind11::reinterpret_steal<pybind11::object>(
        PyTuple_Pack(2, device_capsule.get(), device_info_capsule.get()));
  });
  m.def("AddMesh", [](const py::capsule& device_info,
                      const std::string& serialized_mesh, bool is_async,
                      bool is_host_mesh) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    AddMesh(
        serialized_mesh,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        is_async, is_host_mesh, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def(
      "ExperimentalSetDefaultLayout",
      [](const py::capsule& device_info, const std::string& serialized_layout) {
        std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
            TF_NewStatus(), TF_DeleteStatus);
        ExperimentalSetDefaultLayout(
            serialized_layout,
            PyCapsule_GetPointer(device_info.ptr(),
                                 "TFE_CustomDevice_DeviceInfo"),
            status.get());
        if (TF_GetCode(status.get()) != TF_OK) {
          PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
          throw py::error_already_set();
        }
      });
  m.def("ExperimentalClearDefaultLayout", [](const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    ExperimentalClearDefaultLayout(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("ExperimentalSetDefaultMesh", [](const py::capsule& device_info,
                                         const std::string& serialized_mesh) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    ExperimentalSetDefaultMesh(
        serialized_mesh,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("ExperimentalClearDefaultMesh", [](const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    ExperimentalClearDefaultMesh(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("SetSameShapePolicy", [](const py::capsule& device_info, bool enabled) {
    SetSameShapePolicy(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        enabled);
  });
  m.def("SetTPUCoreIDs", [](const py::capsule& device_info,
                            const std::string& mesh_name,
                            const std::vector<int>& tpu_core_ids) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    SetTPUCoreIDs(
        mesh_name, tpu_core_ids,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"),
        status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
  });
  m.def("ClearTPUCoreIDs", [](const py::capsule& device_info) {
    ClearTPUCoreIDs(
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"));
  });
  m.def("TPUCoreIDsToLocations", [](const py::handle& context,
                                    const py::capsule& device_info,
                                    const std::vector<int>& tpu_core_ids) {
    return TPUCoreIDsToLocations(
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr)),
        tpu_core_ids,
        PyCapsule_GetPointer(device_info.ptr(), "TFE_CustomDevice_DeviceInfo"));
  });
  m.def("TPUCoreLocationsToIDs",
        [](const py::handle& context, const py::capsule& device_info,
           const std::vector<std::vector<int>>& tpu_core_locations) {
          return TPUCoreLocationsToIDs(
              static_cast<TFE_Context*>(
                  PyCapsule_GetPointer(context.ptr(), nullptr)),
              tpu_core_locations,
              PyCapsule_GetPointer(device_info.ptr(),
                                   "TFE_CustomDevice_DeviceInfo"));
        });
  m.def("Pack", [](const py::handle& context, const py::handle& input_tensors,
                   const std::string& string_layout,
                   const py::capsule& device_info, const bool is_sparse) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    TFE_Context* ctx =
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr));
    // Convert each python object to safe py eagertensors.
    std::vector<tensorflow::Safe_PyObjectPtr> py_eager_tensor_handles;
    Py_ssize_t len = PyList_Size(input_tensors.ptr());
    py_eager_tensor_handles.resize(len);

    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* elem = PyList_GetItem(input_tensors.ptr(), i);
      ConvertToTensor(ctx, elem, &py_eager_tensor_handles[i], status.get());

      if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
        return tensorflow::PyoOrThrow(nullptr);
    }
    std::vector<TFE_TensorHandle*> input_vector;
    input_vector.resize(len);
    for (int i = 0; i < len; ++i)
      input_vector[i] = EagerTensor_Handle(py_eager_tensor_handles[i].get());
    TFE_TensorHandle* packed_tensor;
    if (is_sparse) {
      auto size = input_vector.size() / 3;
      packed_tensor = SparsePack(
          ctx,
          /*num_inputs=*/input_vector.size() / 3,
          /*indices=*/
          std::vector<TFE_TensorHandle*>(input_vector.begin(),
                                         input_vector.begin() + size)
              .data(),
          /*values=*/
          std::vector<TFE_TensorHandle*>(input_vector.begin() + size,
                                         input_vector.begin() + 2 * size)
              .data(),
          /*shapes=*/
          std::vector<TFE_TensorHandle*>(input_vector.begin() + 2 * size,
                                         input_vector.end())
              .data(),
          string_layout, device_info, status.get());
    } else {
      packed_tensor = Pack(ctx, input_vector.size(), input_vector.data(),
                           string_layout, device_info, status.get());
    }
    if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
      return tensorflow::PyoOrThrow(nullptr);
    // Convert c++ packed tensor handle into a python eager tensor object.
    tensorflow::Safe_PyObjectPtr flat_result(PyList_New(1));
    PyList_SET_ITEM(flat_result.get(), 0, EagerTensorFromHandle(packed_tensor));
    auto* result = PyList_GET_ITEM(flat_result.get(), 0);
    Py_INCREF(result);
    return tensorflow::PyoOrThrow(result);
  });
  m.def("Unpack", [](const py::handle& context,
                     const py::handle& dtensor_handle,
                     const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);

    TFE_TensorHandle* input_handle = EagerTensor_Handle(dtensor_handle.ptr());
    std::vector<TFE_TensorHandle*> unpacked_handles = Unpack(
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr)),
        input_handle, device_info, status.get());

    if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
      return tensorflow::PyoOrThrow(nullptr);
    // Convert all TFE_TensorHandles to py EagerTensor and
    // return a python list of them.
    int num_outputs = unpacked_handles.size();
    PyObject* result(PyList_New(num_outputs));
    for (int i = 0; i < num_outputs; ++i) {
      PyList_SET_ITEM(result, i, EagerTensorFromHandle(unpacked_handles[i]));
    }
    return tensorflow::PyoOrThrow(result);
  });
  m.def(
      "FetchLayout",
      [](const py::handle& context, const py::handle& dtensor_handle,
         const py::capsule& device_info) -> py::object {
        std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
            TF_NewStatus(), TF_DeleteStatus);

        std::string layout_string =
            FetchLayout(static_cast<TFE_Context*>(
                            PyCapsule_GetPointer(context.ptr(), nullptr)),
                        EagerTensor_Handle(dtensor_handle.ptr()), device_info,
                        status.get());
        if (tensorflow::MaybeRaiseExceptionFromTFStatus(status.get(), nullptr))
          return tensorflow::PyoOrThrow(nullptr);
        return tensorflow::PyoOrThrow(
            PyUnicode_FromString(layout_string.c_str()));
      });
  m.def("IsSparseDTensor", [](const py::handle& context,
                              const py::handle& dtensor_handle,
                              const py::capsule& device_info) {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);

    TFE_TensorHandle* input_handle = EagerTensor_Handle(dtensor_handle.ptr());
    bool is_sparse = IsSparseDTensor(
        static_cast<TFE_Context*>(PyCapsule_GetPointer(context.ptr(), nullptr)),
        input_handle, device_info, status.get());

    if (TF_GetCode(status.get()) != TF_OK) {
      PyErr_SetString(PyExc_ValueError, TF_Message(status.get()));
      throw py::error_already_set();
    }
    return is_sparse;
  });
}

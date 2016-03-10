/* Copyright 2016 Google Inc. All Rights Reserved.

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

%include "tensorflow/python/platform/base.i"

%{
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/public/session_options.h"
%}

%typemap(in, numinputs=0) const tensorflow::SessionOptions& options (
    tensorflow::SessionOptions temp) {
  $1 = &temp;
}

%typemap(in, numinputs=0) std::vector<tensorflow::Device*>* devices (
    std::vector<tensorflow::Device*> temp) {
  $1 = &temp;
}

%typemap(argout) std::vector<tensorflow::Device*>* devices {
  std::vector< std::unique_ptr<tensorflow::Device> > safe_devices;
  for (auto* device : *$1) safe_devices.emplace_back(device);

  auto temp_string_list = tensorflow::make_safe(PyList_New(0));
  if (!temp_string_list) {
    SWIG_fail;
  }

  for (const auto& device : safe_devices) {
    const tensorflow::DeviceAttributes& attr = device->attributes();
    string attr_serialized;
    if (!attr.SerializeToString(&attr_serialized)) {
      PyErr_SetString(PyExc_RuntimeError,
                      "Unable to serialize DeviceAttributes");
      SWIG_fail;
    }

    tensorflow::Safe_PyObjectPtr safe_attr_string = tensorflow::make_safe(
    %#if PY_MAJOR_VERSION < 3
      PyString_FromStringAndSize(
    %#else
      PyUnicode_FromStringAndSize(
    %#endif
    reinterpret_cast<const char*>(
        attr_serialized.data()), attr_serialized.size()));

    if (PyList_Append(temp_string_list.get(), safe_attr_string.get()) == -1) {
      SWIG_fail;
    }
  }

  $result = temp_string_list.release();
}


%ignoreall

%unignore tensorflow;
%unignore tensorflow::DeviceFactory;
%unignore tensorflow::DeviceFactory::AddDevices;

%include "tensorflow/core/common_runtime/device_factory.h"

%unignoreall

%newobject tensorflow::SessionOptions;

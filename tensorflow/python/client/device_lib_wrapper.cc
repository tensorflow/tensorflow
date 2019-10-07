/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "include/pybind11/pybind11.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/python/lib/core/pybind11_proto.h"
#include "tensorflow/python/lib/core/pybind11_status.h"

namespace py = ::pybind11;

PYBIND11_MODULE(_pywrap_device_lib, m) {
  m.def("list_devices", [](py::object serialized_config) {
    tensorflow::ConfigProto config;
    if (!serialized_config.is_none()) {
      config.ParseFromString(
          static_cast<std::string>(serialized_config.cast<py::bytes>()));
    }

    tensorflow::SessionOptions options;
    options.config = config;
    std::vector<std::unique_ptr<tensorflow::Device>> devices;
    tensorflow::MaybeRaiseFromStatus(tensorflow::DeviceFactory::AddDevices(
        options, /*name_prefix=*/"", &devices));

    py::list results;
    std::string serialized_attr;
    for (const auto& device : devices) {
      if (!device->attributes().SerializeToString(&serialized_attr)) {
        tensorflow::MaybeRaiseFromStatus(tensorflow::errors::Internal(
            "Could not serialize DeviceAttributes to bytes"));
      }

      // The default type caster for std::string assumes its contents
      // is UTF8-encoded.
      results.append(py::bytes(serialized_attr));
    }
    return results;
  });
}

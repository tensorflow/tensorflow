/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

%typemap(in) const tensorflow::ConfigProto& (tensorflow::ConfigProto temp) {
  char* c_string;
  Py_ssize_t py_size;
  if (PyBytes_AsStringAndSize($input, &c_string, &py_size) == -1) {
    // Python has raised an error (likely TypeError or UnicodeEncodeError).
    SWIG_fail;
  }

  if (!temp.ParseFromString(string(c_string, py_size))) {
    PyErr_SetString(
        PyExc_TypeError,
        "The ConfigProto could not be parsed as a valid protocol buffer");
    SWIG_fail;
  }
  $1 = &temp;
}

%{
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace swig {

static std::vector<string> ListDevicesWithSessionConfig(
    const tensorflow::ConfigProto& config, TF_Status* out_status) {
  std::vector<string> output;
  SessionOptions options;
  options.config = config;
  std::vector<Device*> devices;
  Status status = DeviceFactory::AddDevices(
      options, "" /* name_prefix */, &devices);
  if (!status.ok()) {
    Set_TF_Status_from_Status(out_status, status);
  }

  std::vector<std::unique_ptr<Device>> device_holder(devices.begin(),
                                                     devices.end());

  for (const Device* device : devices) {
    const DeviceAttributes& attr = device->attributes();
    string attr_serialized;
    if (!attr.SerializeToString(&attr_serialized)) {
      Set_TF_Status_from_Status(
          out_status,
          errors::Internal("Could not serialize device string"));
      output.clear();
      return output;
    }
    output.push_back(attr_serialized);
  }

  return output;
}

std::vector<string> ListDevices(TF_Status* out_status) {
  tensorflow::ConfigProto session_config;
  return ListDevicesWithSessionConfig(session_config, out_status);
}

}  // namespace swig
}  // namespace tensorflow

%}

%ignoreall

%unignore tensorflow;
%unignore tensorflow::swig;
%unignore tensorflow::swig::ListDevicesWithSessionConfig;
%unignore tensorflow::swig::ListDevices;

// Wrap this function
namespace tensorflow {
namespace swig {
std::vector<string> ListDevices(TF_Status* out_status);
static std::vector<string> ListDevicesWithSessionConfig(
    const tensorflow::ConfigProto& config, TF_Status* out_status);
}  // namespace swig
}  // namespace tensorflow

%insert("python") %{
def list_devices(session_config=None):
  from tensorflow.python.framework import errors

  with errors.raise_exception_on_not_ok_status() as status:
    if session_config:
      return ListDevicesWithSessionConfig(session_config.SerializeToString(),
                                          status)
    else:
      return ListDevices(status)
%}

%unignoreall

%newobject tensorflow::SessionOptions;

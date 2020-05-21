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

// A simple logging device to test custom device registration.
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/test.h"

namespace {

struct LoggingDevice {
  tensorflow::string device_name;
  tensorflow::string underlying_device;
  // Set to true whenever a TensorHandle is copied onto the device
  bool* arrived_flag;
  // Set to true whenever an operation is executed
  bool* executed_flag;
};

struct LoggedTensor {
  TFE_TensorHandle* tensor;
  LoggedTensor() = delete;
  explicit LoggedTensor(TFE_TensorHandle* tensor) : tensor(tensor) {}
  ~LoggedTensor() { TFE_DeleteTensorHandle(tensor); }
};

void LoggedTensorDeallocator(void* data, size_t len, void* arg) {
  delete reinterpret_cast<LoggedTensor*>(data);
}

TFE_TensorHandle* MakeLoggedTensorHandle(
    TFE_Context* context, const tensorflow::string& logging_device_name,
    std::unique_ptr<LoggedTensor> t, TF_Status* status) {
  std::vector<int64_t> shape(TFE_TensorHandleNumDims(t->tensor, status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(t->tensor, i, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  auto dtype = TFE_TensorHandleDataType(t->tensor);
  return TFE_NewTensorHandleFromDeviceMemory(
      context, logging_device_name.c_str(), dtype, shape.data(), shape.size(),
      t.release(), 1, &LoggedTensorDeallocator, nullptr, status);
}

TFE_TensorHandle* CopyToLoggingDevice(TFE_Context* context,
                                      TFE_TensorHandle* tensor,
                                      TF_Status* status, void* device_info) {
  LoggingDevice* dev = reinterpret_cast<LoggingDevice*>(device_info);
  TFE_TensorHandle* t = TFE_TensorHandleCopyToDevice(
      tensor, context, dev->underlying_device.c_str(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  auto dst = std::make_unique<LoggedTensor>(t);
  *(dev->arrived_flag) = true;
  return MakeLoggedTensorHandle(context, dev->device_name, std::move(dst),
                                status);
}

TFE_TensorHandle* CopyTensorFromLoggingDevice(TFE_Context* context,
                                              TFE_TensorHandle* tensor,
                                              const char* target_device_name,
                                              TF_Status* status,
                                              void* device_info) {
  TF_SetStatus(status, TF_INTERNAL,
               "Trying to copy a tensor out of a logging device.");
  return nullptr;
}

void LoggingDeviceExecute(TFE_Context* context, int num_inputs,
                          TFE_TensorHandle** inputs, const char* operation_name,
                          const TFE_OpAttrs* attributes, int* num_outputs,
                          TFE_TensorHandle** outputs, TF_Status* s,
                          void* device_info) {
  LoggingDevice* dev = reinterpret_cast<LoggingDevice*>(device_info);
  TFE_Op* op(TFE_NewOp(context, operation_name, s));
  if (TF_GetCode(s) != TF_OK) return;
  TFE_OpAddAttrs(op, attributes);
  TFE_OpSetDevice(op, dev->underlying_device.c_str(), s);
  for (int j = 0; j < num_inputs; ++j) {
    TFE_TensorHandle* input = inputs[j];
    const char* input_device = TFE_TensorHandleDeviceName(input, s);
    if (TF_GetCode(s) != TF_OK) return;
    if (dev->device_name == input_device) {
      LoggedTensor* t = reinterpret_cast<LoggedTensor*>(
          TFE_TensorHandleDevicePointer(input, s));
      if (TF_GetCode(s) != TF_OK) return;
      TFE_OpAddInput(op, t->tensor, s);
    } else {
      TFE_OpAddInput(op, input, s);
    }
    if (TF_GetCode(s) != TF_OK) return;
  }
  std::vector<TFE_TensorHandle*> op_outputs(*num_outputs);
  TFE_Execute(op, op_outputs.data(), num_outputs, s);
  TFE_DeleteOp(op);
  if (TF_GetCode(s) != TF_OK) return;
  std::vector<TFE_TensorHandle*> unwrapped_outputs;
  for (auto* handle : op_outputs) {
    unwrapped_outputs.push_back(handle);
  }
  for (int i = 0; i < *num_outputs; ++i) {
    auto logged_tensor = std::make_unique<LoggedTensor>(unwrapped_outputs[i]);
    outputs[i] = MakeLoggedTensorHandle(context, dev->device_name,
                                        std::move(logged_tensor), s);
  }
  *(dev->executed_flag) = true;
}

void DeleteLoggingDevice(void* device_info) {
  delete reinterpret_cast<LoggingDevice*>(device_info);
}

}  // namespace

void RegisterLoggingDevice(TFE_Context* context, const char* name,
                           bool* arrived_flag, bool* executed_flag,
                           TF_Status* status) {
  TFE_CustomDevice custom_device;
  custom_device.copy_tensor_to_device = &CopyToLoggingDevice;
  custom_device.copy_tensor_from_device = &CopyTensorFromLoggingDevice;
  custom_device.delete_device = &DeleteLoggingDevice;
  custom_device.execute = &LoggingDeviceExecute;
  LoggingDevice* device = new LoggingDevice;
  device->arrived_flag = arrived_flag;
  device->executed_flag = executed_flag;
  device->device_name = name;
  device->underlying_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_RegisterCustomDevice(context, custom_device, name, device, status);
}

TFE_TensorHandle* UnpackTensorHandle(TFE_TensorHandle* logged_tensor_handle,
                                     TF_Status* status) {
  return reinterpret_cast<LoggedTensor*>(
             TFE_TensorHandleDevicePointer(logged_tensor_handle, status))
      ->tensor;
}

void AllocateLoggingDevice(const char* name, bool* arrived_flag,
                           bool* executed_flag, TFE_CustomDevice** device,
                           void** device_info) {
  TFE_CustomDevice* custom_device = new TFE_CustomDevice;
  custom_device->copy_tensor_to_device = &CopyToLoggingDevice;
  custom_device->copy_tensor_from_device = &CopyTensorFromLoggingDevice;
  custom_device->delete_device = &DeleteLoggingDevice;
  custom_device->execute = &LoggingDeviceExecute;
  *device = custom_device;
  LoggingDevice* logging_device = new LoggingDevice;
  logging_device->arrived_flag = arrived_flag;
  logging_device->executed_flag = executed_flag;
  logging_device->device_name = name;
  logging_device->underlying_device =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  *device_info = reinterpret_cast<void*>(logging_device);
}

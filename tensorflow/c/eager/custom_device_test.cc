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
#include "tensorflow/core/platform/test.h"

namespace {

struct LoggingDevice {
  TFE_Context* ctx;
  tensorflow::string device_name;
  tensorflow::string underlying_device;
  // Set to true whenever a TensorHandle is copied onto the device
  bool* arrived_flag;
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
    TFE_Context* ctx, const tensorflow::string& logging_device_name,
    std::unique_ptr<LoggedTensor> t, TF_Status* status) {
  std::vector<int64_t> shape(TFE_TensorHandleNumDims(t->tensor, status));
  if (TF_GetCode(status) != TF_OK) return nullptr;
  for (int i = 0; i < shape.size(); ++i) {
    shape[i] = TFE_TensorHandleDim(t->tensor, i, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  auto dtype = TFE_TensorHandleDataType(t->tensor);
  return TFE_NewTensorHandleFromDeviceMemory(
      ctx, logging_device_name.c_str(), dtype, shape.data(), shape.size(),
      t.release(), 1, &LoggedTensorDeallocator, nullptr, status);
}

TFE_TensorHandle* CopyToLoggingDevice(TFE_TensorHandle* tensor,
                                      TF_Status* status, void* device_info) {
  LoggingDevice* dev = reinterpret_cast<LoggingDevice*>(device_info);
  TFE_TensorHandle* t = TFE_TensorHandleCopyToDevice(
      tensor, dev->ctx, dev->underlying_device.c_str(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  auto dst = std::make_unique<LoggedTensor>(t);
  *(dev->arrived_flag) = true;
  return MakeLoggedTensorHandle(dev->ctx, dev->device_name, std::move(dst),
                                status);
}

TFE_TensorHandle* CopyTensorFromLoggingDevice(TFE_TensorHandle* tensor,
                                              const char* target_device_name,
                                              TF_Status* status,
                                              void* device_info) {
  TF_SetStatus(status, TF_INTERNAL,
               "Trying to copy a tensor out of a logging device.");
  return nullptr;
}

void LoggingDeviceExecute(int num_inputs, TFE_TensorHandle** inputs,
                          const char* operation_name, int* num_outputs,
                          TFE_TensorHandle** outputs, TF_Status* s,
                          void* device_info) {
  LoggingDevice* dev = reinterpret_cast<LoggingDevice*>(device_info);
  TFE_Op* op(TFE_NewOp(dev->ctx, operation_name, s));
  if (TF_GetCode(s) != TF_OK) return;
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
    outputs[i] = MakeLoggedTensorHandle(dev->ctx, dev->device_name,
                                        std::move(logged_tensor), s);
  }
}

void DeleteLoggingDevice(void* device_info) {
  delete reinterpret_cast<LoggingDevice*>(device_info);
}

void RegisterLoggingDevice(TFE_Context* context, const char* name,
                           bool* arrived_flag) {
  TFE_CustomDevice custom_device;
  custom_device.copy_tensor_to_device = &CopyToLoggingDevice;
  custom_device.copy_tensor_from_device = &CopyTensorFromLoggingDevice;
  custom_device.delete_device = &DeleteLoggingDevice;
  custom_device.execute = &LoggingDeviceExecute;
  LoggingDevice* device = new LoggingDevice;
  device->ctx = context;
  device->arrived_flag = arrived_flag;
  device->device_name = name;
  device->underlying_device = "/job:localhost/replica:0/task:0/device:CPU:0";
  TFE_RegisterCustomDevice(context, custom_device, name, device);
}

TEST(CUSTOM_DEVICE, RegisterSimpleDevice) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* context = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool arrived = false;
  const char* name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context, name, &arrived);
  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  ASSERT_FALSE(arrived);
  TFE_TensorHandle* hdevice =
      TFE_TensorHandleCopyToDevice(hcpu, context, name, status.get());
  ASSERT_TRUE(arrived);
  TFE_DeleteTensorHandle(hcpu);
  TFE_DeleteTensorHandle(hdevice);
  TFE_DeleteContext(context);
}

}  // namespace

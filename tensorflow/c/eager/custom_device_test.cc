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

TEST(CUSTOM_DEVICE, RegisterSimpleDevice) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* context = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool arrived = false;
  bool executed = false;
  const char* name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context, name, &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  ASSERT_FALSE(arrived);
  TFE_TensorHandle* hdevice =
      TFE_TensorHandleCopyToDevice(hcpu, context, name, status.get());
  ASSERT_TRUE(arrived);
  ASSERT_FALSE(executed);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> matmul(
      MatMulOp(context, hcpu, hdevice), TFE_DeleteOp);
  TFE_OpSetDevice(matmul.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_TensorHandle* retval;
  int num_retvals = 1;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);

  TFE_DeleteTensorHandle(retval);
  TFE_DeleteTensorHandle(hcpu);
  TFE_DeleteTensorHandle(hdevice);
  TFE_DeleteContext(context);
}

TEST(CUSTOM_DEVICE, ResetOperation) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts, status.get()), TFE_DeleteContext);
  TFE_DeleteContextOptions(opts);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool arrived = false;
  bool executed = false;
  const char* custom_device_name =
      "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context.get(), custom_device_name, &arrived, &executed,
                        status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> reused_op(
      TFE_NewOp(context.get(), "Identity", status.get()), TFE_DeleteOp);
  TFE_OpReset(reused_op.get(), "Identity", custom_device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_EQ(tensorflow::string(TFE_OpGetDevice(reused_op.get(), status.get())),
            tensorflow::string(custom_device_name));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpReset(reused_op.get(), "Identity",
              "/job:localhost/replica:0/task:0/device:CPU:0", status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_EQ(tensorflow::string(TFE_OpGetDevice(reused_op.get(), status.get())),
            tensorflow::string("/job:localhost/replica:0/task:0/device:CPU:0"));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
}

TEST(CUSTOM_DEVICE, MakeVariable) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool arrived = false;
  bool executed = false;
  const char* name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context.get(), name, &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a variable handle placed on the custom device.
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context.get(), "VarHandleOp", status.get()), TFE_DeleteOp);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op.get(), "shape", {}, 0, status.get());
  TFE_OpSetAttrString(op.get(), "container", "", 0);
  TFE_OpSetAttrString(op.get(), "shared_name", "", 0);
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  executed = false;
  TFE_Execute(op.get(), &var_handle, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);
  auto handle_cleaner = tensorflow::gtl::MakeCleanup(
      [var_handle]() { TFE_DeleteTensorHandle(var_handle); });

  // Assign to the variable, copying to the custom device.
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> one(
      TestScalarTensorHandle(111.f), TFE_DeleteTensorHandle);
  op.reset(TFE_NewOp(context.get(), "AssignVariableOp", status.get()));
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpAddInput(op.get(), one.get(), status.get());
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  executed = false;
  num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);

  // Read the variable's value.
  op.reset(TFE_NewOp(context.get(), "ReadVariableOp", status.get()));
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpSetDevice(op.get(), name, status.get());
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  executed = false;
  num_retvals = 1;
  TFE_TensorHandle* var_value = nullptr;
  TFE_Execute(op.get(), &var_value, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);
  auto value_cleaner = tensorflow::gtl::MakeCleanup(
      [var_value]() { TFE_DeleteTensorHandle(var_value); });
  ASSERT_EQ(tensorflow::string(name),
            tensorflow::string(
                TFE_TensorHandleBackingDeviceName(var_value, status.get())));
  TFE_TensorHandle* var_value_unpacked =
      reinterpret_cast<LoggedTensor*>(
          TFE_TensorHandleDevicePointer(var_value, status.get()))
          ->tensor;
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> resolved_value(
      TFE_TensorHandleResolve(var_value_unpacked, status.get()),
      TF_DeleteTensor);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_EQ(111., *static_cast<float*>(TF_TensorData(resolved_value.get())));

  // Free the backing buffer for the variable.
  op.reset(TFE_NewOp(context.get(), "DestroyResourceOp", status.get()));
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
}

TEST(CUSTOM_DEVICE, AccessVariableOnWrongDevice) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool arrived = false;
  bool executed = false;
  const char* name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context.get(), name, &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a variable handle placed on the custom device.
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context.get(), "VarHandleOp", status.get()), TFE_DeleteOp);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op.get(), "shape", {}, 0, status.get());
  TFE_OpSetAttrString(op.get(), "container", "", 0);
  TFE_OpSetAttrString(op.get(), "shared_name", "", 0);
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  executed = false;
  TFE_Execute(op.get(), &var_handle, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);
  auto handle_cleaner = tensorflow::gtl::MakeCleanup(
      [var_handle]() { TFE_DeleteTensorHandle(var_handle); });

  // Assign to the variable, copying to the custom device.
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> one(
      TestScalarTensorHandle(111.f), TFE_DeleteTensorHandle);
  op.reset(TFE_NewOp(context.get(), "AssignVariableOp", status.get()));
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpAddInput(op.get(), one.get(), status.get());
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  executed = false;
  num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);

  // Read the variable's value.
  op.reset(TFE_NewOp(context.get(), "ReadVariableOp", status.get()));
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  executed = false;
  num_retvals = 1;
  TFE_TensorHandle* var_value = nullptr;
  TFE_Execute(op.get(), &var_value, &num_retvals, status.get());
  EXPECT_FALSE(TF_GetCode(status.get()) == TF_OK)
      << "Execution should fail because the variable is being used on the "
         "wrong device.";
  // Free the backing buffer for the variable.
  op.reset(TFE_NewOp(context.get(), "DestroyResourceOp", status.get()));
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
}

TEST(CUSTOM_DEVICE, InvalidRegistrationError) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool arrived = false;
  bool executed = false;
  RegisterLoggingDevice(context.get(), "/device:CUSTOM:0", &arrived, &executed,
                        status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_INVALID_ARGUMENT)
      << TF_Message(status.get());

  const char* name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context.get(), name, &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  RegisterLoggingDevice(context.get(), name, &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_ALREADY_EXISTS)
      << TF_Message(status.get());

  RegisterLoggingDevice(context.get(),
                        "/job:localhost/replica:0/task:0/device:CPU:0",
                        &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_ALREADY_EXISTS)
      << TF_Message(status.get());
}

}  // namespace

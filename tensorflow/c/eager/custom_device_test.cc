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

#include "absl/strings/match.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/custom_device_testutil.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/test.h"

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
  RegisterLoggingDevice(context, name, /*strict_scope_placement=*/true,
                        &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_TensorHandle* hcpu = TestMatrixTensorHandle(context);
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
  RegisterLoggingDevice(context.get(), custom_device_name,
                        /*strict_scope_placement=*/true, &arrived, &executed,
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
  RegisterLoggingDevice(context.get(), name, /*strict_scope_placement=*/true,
                        &arrived, &executed, status.get());
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
      TestScalarTensorHandle(context.get(), 111.f), TFE_DeleteTensorHandle);
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
      UnpackTensorHandle(var_value, status.get());
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

TEST(CUSTOM_DEVICE, AccessVariableOnCustomDevice) {
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
  RegisterLoggingDevice(context.get(), name, /*strict_scope_placement=*/false,
                        &arrived, &executed, status.get());
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
      TestScalarTensorHandle(context.get(), 111.f), TFE_DeleteTensorHandle);
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
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpAddInput(op.get(), var_handle, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetAttrType(op.get(), "dtype", TF_FLOAT);
  executed = false;
  num_retvals = 1;
  TFE_TensorHandle* var_value = nullptr;
  TFE_Execute(op.get(), &var_value, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);
  ASSERT_EQ(
      tensorflow::string(name),
      tensorflow::string(TFE_TensorHandleDeviceName(var_value, status.get())));
  TFE_DeleteTensorHandle(var_value);

  // Free the backing buffer for the variable.
  op.reset(TFE_NewOp(context.get(), "DestroyResourceOp", status.get()));
  TFE_OpAddInput(op.get(), var_handle, status.get());
  TFE_OpSetDevice(op.get(), name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
}

TEST(CUSTOM_DEVICE, InputBasedPlacement) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  const char* custom0 = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  const char* custom1 = "/job:localhost/replica:0/task:0/device:CUSTOM:1";
  bool arrived = false;
  bool executed = false;
  RegisterLoggingDevice(context.get(), custom0,
                        /*strict_scope_placement=*/false, &arrived, &executed,
                        status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  RegisterLoggingDevice(context.get(), custom1,
                        /*strict_scope_placement=*/true, &arrived, &executed,
                        status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> hcpu(
      TestMatrixTensorHandle(context.get()), TFE_DeleteTensorHandle);
  ASSERT_FALSE(arrived);
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> hcustom0(
      TFE_TensorHandleCopyToDevice(hcpu.get(), context.get(), custom0,
                                   status.get()),
      TFE_DeleteTensorHandle);
  ASSERT_TRUE(arrived);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  arrived = false;
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> hcustom1(
      TFE_TensorHandleCopyToDevice(hcpu.get(), context.get(), custom1,
                                   status.get()),
      TFE_DeleteTensorHandle);
  ASSERT_TRUE(arrived);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Base case: two CPU inputs executes fine.
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> matmul(
      MatMulOp(context.get(), hcpu.get(), hcpu.get()), TFE_DeleteOp);
  TFE_TensorHandle* retval;
  int num_retvals = 1;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_DeleteTensorHandle(retval);

  // Custom device: inputs in same custom device works.
  matmul.reset(MatMulOp(context.get(), hcustom0.get(), hcustom0.get()));
  num_retvals = 1;
  executed = false;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_TRUE(executed);
  TFE_DeleteTensorHandle(retval);

  // Custom device: inputs in different custom devices fails.
  matmul.reset(MatMulOp(context.get(), hcustom0.get(), hcustom1.get()));
  num_retvals = 1;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  ASSERT_NE(TF_OK, TF_GetCode(status.get()));
  ASSERT_TRUE(absl::StrContains(TF_Message(status.get()), custom0));
  ASSERT_TRUE(absl::StrContains(TF_Message(status.get()), custom1));

  // Custom device: mix of custom/physical places the op on the custom device.
  matmul.reset(MatMulOp(context.get(), hcustom0.get(), hcpu.get()));
  num_retvals = 1;
  executed = false;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  EXPECT_TRUE(executed);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_DeleteTensorHandle(retval);

  // Explicit placement still forces the op onto the requested device
  matmul.reset(MatMulOp(context.get(), hcustom0.get(), hcpu.get()));
  TFE_OpSetDevice(matmul.get(), "/job:localhost/replica:0/task:0/device:CPU:0",
                  status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  num_retvals = 1;
  executed = false;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  EXPECT_FALSE(executed);
  ASSERT_FALSE(TF_GetCode(status.get()) == TF_OK);

  // Custom devices can refuse to do type-based dispatch (as hcustom1 is
  // configured to do)
  matmul.reset(MatMulOp(context.get(), hcustom1.get(), hcpu.get()));
  num_retvals = 1;
  executed = false;
  TFE_Execute(matmul.get(), &retval, &num_retvals, status.get());
  EXPECT_FALSE(executed);
  ASSERT_FALSE(TF_GetCode(status.get()) == TF_OK);
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
  RegisterLoggingDevice(context.get(), "/device:CUSTOM:0",
                        /*strict_scope_placement=*/true, &arrived, &executed,
                        status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_INVALID_ARGUMENT)
      << TF_Message(status.get());

  const char* name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  RegisterLoggingDevice(context.get(), name, /*strict_scope_placement=*/true,
                        &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  RegisterLoggingDevice(context.get(), name, /*strict_scope_placement=*/true,
                        &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_ALREADY_EXISTS)
      << TF_Message(status.get());

  RegisterLoggingDevice(
      context.get(), "/job:localhost/replica:0/task:0/device:CPU:0",
      /*strict_scope_placement=*/true, &arrived, &executed, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_ALREADY_EXISTS)
      << TF_Message(status.get());
}

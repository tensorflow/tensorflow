/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api_test_util.h"

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::string;

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, float value) {
  float data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, int value) {
  int data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_INT32, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestScalarTensorHandle(TFE_Context* ctx, bool value) {
  bool data[] = {value};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_BOOL, nullptr, 0, status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* DoubleTestMatrixTensorHandle(TFE_Context* ctx) {
  int64_t dims[] = {2, 2};
  double data[] = {1.0, 2.0, 3.0, 4.0};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_DOUBLE, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle(TFE_Context* ctx) {
  int64_t dims[] = {2, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle100x100(TFE_Context* ctx) {
  constexpr int64_t dims[] = {100, 100};
  constexpr int num_elements = dims[0] * dims[1];
  float data[num_elements];
  for (int i = 0; i < num_elements; ++i) {
    data[i] = 1.0f;
  }
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* DoubleTestMatrixTensorHandle3X2(TFE_Context* ctx) {
  int64_t dims[] = {3, 2};
  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestMatrixTensorHandle3X2(TFE_Context* ctx) {
  int64_t dims[] = {3, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_FLOAT, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_TensorHandle* TestVariable(TFE_Context* ctx, float value,
                               const tensorflow::string& device_name) {
  TF_Status* status = TF_NewStatus();
  // Create the variable handle.
  TFE_Op* op = TFE_NewOp(ctx, "VarHandleOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op, "shape", {}, 0, status);
  TFE_OpSetAttrString(op, "container", "", 0);
  TFE_OpSetAttrString(op, "shared_name", "", 0);
  if (!device_name.empty()) {
    TFE_OpSetDevice(op, device_name.c_str(), status);
  }
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op, &var_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(1, num_retvals);

  // Assign 'value' to it.
  op = TFE_NewOp(ctx, "AssignVariableOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var_handle, status);

  // Convert 'value' to a TF_Tensor then a TFE_TensorHandle.
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> t(
      TF_AllocateTensor(TF_FLOAT, nullptr, 0, sizeof(value)), TF_DeleteTensor);
  memcpy(TF_TensorData(t.get()), &value, TF_TensorByteSize(t.get()));

  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
      value_handle(TFE_NewTensorHandle(t.get(), status),
                   TFE_DeleteTensorHandle);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_OpAddInput(op, value_handle.get(), status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  num_retvals = 0;
  TFE_Execute(op, nullptr, &num_retvals, status);
  TFE_DeleteOp(op);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  CHECK_EQ(0, num_retvals);

  TF_DeleteStatus(status);

  return var_handle;
}

TFE_Op* AddOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "AddV2", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, b, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* MatMulOp(TFE_Context* ctx, TFE_TensorHandle* a, TFE_TensorHandle* b) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "MatMul", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, b, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* IdentityOp(TFE_Context* ctx, TFE_TensorHandle* a) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Identity", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_Op* ShapeOp(TFE_Context* ctx, TFE_TensorHandle* a) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Shape", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, a, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

TFE_TensorHandle* TestAxisTensorHandle(TFE_Context* ctx) {
  int64_t dims[] = {1};
  int data[] = {1};
  TF_Status* status = TF_NewStatus();
  TF_Tensor* t = TFE_AllocateHostTensor(ctx, TF_INT32, &dims[0],
                                        sizeof(dims) / sizeof(int64_t), status);
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TFE_TensorHandle* th = TFE_NewTensorHandleFromTensor(ctx, t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
}

TFE_Op* MinOp(TFE_Context* ctx, TFE_TensorHandle* input,
              TFE_TensorHandle* axis) {
  TF_Status* status = TF_NewStatus();

  TFE_Op* op = TFE_NewOp(ctx, "Min", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, input, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, axis, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrBool(op, "keep_dims", 1);
  TFE_OpSetAttrType(op, "Tidx", TF_INT32);
  TF_DeleteStatus(status);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(input));

  return op;
}

bool GetDeviceName(TFE_Context* ctx, string* device_name,
                   const char* device_type) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status.get());
  CHECK_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const int num_devices = TF_DeviceListCount(devices);
  for (int i = 0; i < num_devices; ++i) {
    const string dev_type(TF_DeviceListType(devices, i, status.get()));
    CHECK_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    const string dev_name(TF_DeviceListName(devices, i, status.get()));
    CHECK_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    if (dev_type == device_type) {
      *device_name = dev_name;
      LOG(INFO) << "Found " << device_type << " device " << *device_name;
      TF_DeleteDeviceList(devices);
      return true;
    }
  }
  TF_DeleteDeviceList(devices);
  return false;
}

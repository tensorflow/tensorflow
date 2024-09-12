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

#include "tensorflow/c/eager/parallel_device/parallel_device_testlib.h"

#include <array>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/test.h"

// NOTE(allenl): These tests currently go through TFE_Execute and so are
// integration testing rather than purely testing the parallel device. They
// correspond fairly well to the implementation, but testing the C++ directly is
// another option.

namespace tensorflow {
namespace parallel_device {

Variable* Variable::Create(TFE_Context* context, TF_DataType type,
                           const int64_t* dims, const int num_dims,
                           const char* device, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "VarHandleOp", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op.get(), "dtype", type);
  TFE_OpSetAttrShape(op.get(), "shape", dims, num_dims, status);
  TFE_OpSetAttrString(op.get(), "container", "", 0);
  // Use the special GUID for no buffer sharing
  //
  // TODO(allenl): Should we provide a better API for this? AFAIK this is the
  // only reasonable way to make variables with no aliasing using the eager C
  // API.
  std::string no_sharing = "cd2c89b7-88b7-44c8-ad83-06c2a9158347";
  TFE_OpSetAttrString(op.get(), "shared_name", no_sharing.c_str(),
                      no_sharing.length());
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op.get(), &var_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return new Variable(var_handle, type);
}

void Variable::Destroy(TFE_Context* context, TF_Status* status) {
  // Free the backing buffer for the variable.
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "DestroyResourceOp", status), &TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpAddInput(op.get(), handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* device = TFE_TensorHandleDeviceName(handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return;
  int num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return;
  // Delete the variable handle itself.
  TFE_DeleteTensorHandle(handle_);
}

TensorHandlePtr Variable::Read(TFE_Context* context, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "ReadVariableOp", status), &TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpAddInput(op.get(), handle_, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  const char* device = TFE_TensorHandleDeviceName(handle_, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op.get(), "dtype", type_);
  int num_retvals = 1;
  TFE_TensorHandle* var_value = nullptr;
  TFE_Execute(op.get(), &var_value, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(var_value);
}

void Variable::GeneralAssignment(const char* op_name, TFE_Context* context,
                                 TFE_TensorHandle* value, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, op_name, status), &TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetAttrType(op.get(), "dtype", type_);
  TFE_OpAddInput(op.get(), handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpAddInput(op.get(), value, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* device = TFE_TensorHandleDeviceName(handle_, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetDevice(op.get(), device, status);

  int num_retvals = 0;
  TFE_Execute(op.get(), nullptr, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return;
}

void Variable::AssignAdd(TFE_Context* context, TFE_TensorHandle* value,
                         TF_Status* status) {
  GeneralAssignment("AssignAddVariableOp", context, value, status);
}

void Variable::Assign(TFE_Context* context, TFE_TensorHandle* value,
                      TF_Status* status) {
  GeneralAssignment("AssignVariableOp", context, value, status);
}

// Passed to `TF_NewTensor` to indicate how an array of floats should be
// deleted.
static void FloatDeallocator(void* data, size_t, void* arg) {
  delete[] static_cast<float*>(data);
}

// Creates a TFE_TensorHandle with value `v`.
TensorHandlePtr FloatTensorHandle(float v, TF_Status* status) {
  const int num_bytes = sizeof(float);
  float* values = new float[1];
  values[0] = v;
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
      TF_NewTensor(TF_FLOAT, nullptr, 0, values, num_bytes, &FloatDeallocator,
                   nullptr),
      TF_DeleteTensor);
  return TensorHandlePtr(TFE_NewTensorHandle(tensor.get(), status));
}

// Creates a rank-one TFE_TensorHandle with value `v`.
TensorHandlePtr VectorFloatTensorHandle(const std::vector<float>& v,
                                        TF_Status* status) {
  const int num_bytes = v.size() * sizeof(float);
  float* values = new float[v.size()];
  memcpy(values, v.data(), num_bytes);
  int64_t dims = v.size();
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tensor(
      TF_NewTensor(TF_FLOAT, &dims, 1 /* num_dims */, values, num_bytes,
                   &FloatDeallocator, nullptr),
      TF_DeleteTensor);
  return TensorHandlePtr(TFE_NewTensorHandle(tensor.get(), status));
}

// Helper to un-pack `num_replicas` TFE_TensorHandles from one parallel handle.
template <std::size_t num_replicas>
void ExtractPerDeviceValues(
    TFE_Context* context, TFE_TensorHandle* input,
    std::array<TensorHandlePtr, num_replicas>* components, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "TPUReplicatedOutput", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetAttrInt(op.get(), "num_replicas", num_replicas);
  TFE_OpAddInput(op.get(), input, status);
  if (TF_GetCode(status) != TF_OK) return;
  const char* device = TFE_TensorHandleDeviceName(input, status);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return;

  TFE_TensorHandle* result_handles[num_replicas];
  int num_retvals = num_replicas;
  TFE_Execute(op.get(), result_handles, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return;
  for (int i = 0; i < num_replicas; ++i) {
    (*components)[i].reset(result_handles[i]);
  }
}

TensorHandlePtr Multiply(TFE_Context* context, TFE_TensorHandle* first,
                         TFE_TensorHandle* second, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "Mul", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpAddInput(op.get(), first, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpAddInput(op.get(), second, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  const char* first_device = TFE_TensorHandleDeviceName(first, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetDevice(op.get(), first_device, status);

  TFE_TensorHandle* result_handle;
  int num_retvals = 1;
  TFE_Execute(op.get(), &result_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(result_handle);
}

// Create and modify a variable placed on a parallel device which composes
// `first_device` and `second_device`.
void BasicTestsForTwoDevices(TFE_Context* context, const char* first_device,
                             const char* second_device) {
  // Register the custom device
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{first_device, second_device};
  RegisterParallelDevice(context, device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a variable handle (uninitialized to start) placed on the parallel
  // device.
  std::function<void(Variable*)> variable_deleter = [&](Variable* to_delete) {
    to_delete->Destroy(context, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    delete to_delete;
  };
  std::unique_ptr<Variable, decltype(variable_deleter)> variable(
      Variable::Create(context, TF_FLOAT, /* Scalar */ {}, 0, device_name,
                       status.get()),
      variable_deleter);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Assign an initial value to the variable, mirroring it to each component
  // device.
  {
    TensorHandlePtr initial_value_cpu = FloatTensorHandle(20., status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TFE_TensorHandle*, 2> components{initial_value_cpu.get(),
                                                initial_value_cpu.get()};
    TensorHandlePtr initial_value =
        CreatePerDeviceValues(context, components, device_name, status.get());
    variable->Assign(context, initial_value.get(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  }

  // Read from the variable and verify that we have a parallel tensor.
  {
    TensorHandlePtr read = variable->Read(context, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TensorHandlePtr, 2> components;
    ExtractPerDeviceValues(context, read.get(), &components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    ExpectScalarEq<float>(components[0].get(), 20.);
    ExpectScalarEq<float>(components[1].get(), 20.);

    std::string first_device =
        TFE_TensorHandleBackingDeviceName(components[0].get(), status.get());
    ASSERT_EQ(underlying_devices[0], first_device);
    std::string second_device =
        TFE_TensorHandleBackingDeviceName(components[1].get(), status.get());
    ASSERT_EQ(underlying_devices[1], second_device);
  }

  // Add a parallel tensor with different values on each device to the variable.
  {
    TensorHandlePtr value_one(FloatTensorHandle(3., status.get()));
    TensorHandlePtr value_two(FloatTensorHandle(-2., status.get()));
    std::array<TFE_TensorHandle*, 2> components{value_one.get(),
                                                value_two.get()};
    TensorHandlePtr combined_value =
        CreatePerDeviceValues(context, components, device_name, status.get());
    variable->AssignAdd(context, combined_value.get(), status.get());
  }

  // Read the variable and verify that each component has the right modified
  // value.
  {
    TensorHandlePtr read = variable->Read(context, status.get());
    std::array<TensorHandlePtr, 2> components;
    ExtractPerDeviceValues(context, read.get(), &components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    ExpectScalarEq<float>(components[0].get(), 23.);
    ExpectScalarEq<float>(components[1].get(), 18.);

    std::string first_device =
        TFE_TensorHandleBackingDeviceName(components[0].get(), status.get());
    ASSERT_EQ(underlying_devices[0], first_device);
    std::string second_device =
        TFE_TensorHandleBackingDeviceName(components[1].get(), status.get());
    ASSERT_EQ(underlying_devices[1], second_device);
  }
}

}  // namespace parallel_device
}  // namespace tensorflow

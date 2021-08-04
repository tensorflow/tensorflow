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

#ifndef TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_TESTLIB_H_
#define TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_TESTLIB_H_

#include <array>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/parallel_device/parallel_device.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace parallel_device {

// A helper for performing common operations on variables. A much more
// restricted stand-in for tf.Variable in Python.
class Variable {
 public:
  // Construct a Variable from a resource-dtype TFE_TensorHandle and an
  // indication of the dtype of the variable's value.
  //
  // Note that creating this resource-dtype handle can fail, so `Create` is a
  // separate static method which returns a status.
  Variable(TFE_TensorHandle* handle, TF_DataType type)
      : handle_(handle), type_(type) {}

  // Helper for constructing a resource handle and wrapping it in a `Variable`
  // object.
  static Variable* Create(TFE_Context* context, TF_DataType type,
                          const int64_t* dims, const int num_dims,
                          const char* device, TF_Status* status);
  // Dereferences the backing buffer for the variable. Note that since this can
  // fail (it runs operations), it must be called explicitly and the resulting
  // `status` checked.
  void Destroy(TFE_Context* context, TF_Status* status);

  // Reads from the variable.
  TensorHandlePtr Read(TFE_Context* context, TF_Status* status);
  // Assigns a new value to the variable.
  void Assign(TFE_Context* context, TFE_TensorHandle* value, TF_Status* status);
  // Adds `value` to the existing value of the variable.
  void AssignAdd(TFE_Context* context, TFE_TensorHandle* value,
                 TF_Status* status);

 private:
  // Helper for running any single-argument assignment ops (Assign, AssignAdd,
  // AssignSub, ...).
  void GeneralAssignment(const char* op_name, TFE_Context* context,
                         TFE_TensorHandle* value, TF_Status* status);

  // The a handle for the resource-dtype tensor pointing to the variable's
  // buffer.
  TFE_TensorHandle* handle_;
  // The dtype of the variable's buffer (input dtype for assignments, output
  // dtype of read operations).
  TF_DataType type_;
};

// Creates a TFE_TensorHandle with value `v`.
TensorHandlePtr FloatTensorHandle(float v, TF_Status* status);

// Creates a rank-one TFE_TensorHandle with value `v`.
TensorHandlePtr VectorFloatTensorHandle(const std::vector<float>& v,
                                        TF_Status* status);

// Helper to un-pack `num_replicas` TFE_TensorHandles from one parallel handle.
template <std::size_t num_replicas>
void ExtractPerDeviceValues(
    TFE_Context* context, TFE_TensorHandle* input,
    std::array<TensorHandlePtr, num_replicas>* components, TF_Status* status);

// Helper to pack `num_replicas` TFE_TensorHandles into one parallel handle.
template <std::size_t num_replicas>
TensorHandlePtr CreatePerDeviceValues(
    TFE_Context* context,
    const std::array<TFE_TensorHandle*, num_replicas>& components,
    const char* device, TF_Status* status);

TensorHandlePtr Multiply(TFE_Context* context, TFE_TensorHandle* first,
                         TFE_TensorHandle* second, TF_Status* status);

// Assert that `handle` is equal to `expected_value`.
template <typename value_type>
void ExpectScalarEq(TFE_TensorHandle* handle, value_type expected_value);

template <std::size_t num_devices>
void RegisterParallelDevice(
    TFE_Context* context, const char* device_name,
    const std::array<const char*, num_devices>& underlying_devices,
    TF_Status* status);

// Create and modify a variable placed on a parallel device which composes
// `first_device` and `second_device`.
void BasicTestsForTwoDevices(TFE_Context* context, const char* first_device,
                             const char* second_device);

// Implementations of templated functions ******************************

template <std::size_t num_replicas>
TensorHandlePtr CreatePerDeviceValues(
    TFE_Context* context,
    const std::array<TFE_TensorHandle*, num_replicas>& components,
    const char* device, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "TPUReplicatedInput", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrInt(op.get(), "N", num_replicas);
  for (int i = 0; i < num_replicas; ++i) {
    TFE_OpAddInput(op.get(), components[i], status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
  }
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_TensorHandle* result_handle;
  int num_retvals = 1;
  TFE_Execute(op.get(), &result_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(result_handle);
}

template <typename value_type>
void ExpectScalarEq(TFE_TensorHandle* handle, value_type expected_value) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> actual_value(
      TFE_TensorHandleResolve(handle, status.get()), TF_DeleteTensor);
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
  ASSERT_EQ(TF_TensorType(actual_value.get()),
            static_cast<TF_DataType>(DataTypeToEnum<value_type>().value));
  EXPECT_EQ(expected_value,
            *static_cast<value_type*>(TF_TensorData(actual_value.get())));
}

template <std::size_t num_devices>
void RegisterParallelDevice(
    TFE_Context* context, const char* device_name,
    const std::array<const char*, num_devices>& underlying_devices,
    TF_Status* status) {
  TFE_CustomDevice device;
  void* device_info;
  tensorflow::parallel_device::AllocateParallelDevice(
      device_name, underlying_devices.data(), underlying_devices.size(),
      &device, &device_info);
  TFE_RegisterCustomDevice(context, device, device_name, device_info, status);
}

}  // namespace parallel_device
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_PARALLEL_DEVICE_PARALLEL_DEVICE_TESTLIB_H_

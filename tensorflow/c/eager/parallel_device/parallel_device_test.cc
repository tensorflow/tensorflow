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

#include "tensorflow/c/eager/parallel_device/parallel_device.h"

#include <array>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/parallel_device/parallel_device_testlib.h"
#include "tensorflow/core/platform/test.h"

// NOTE(allenl): These tests currently go through TFE_Execute and so are
// integration testing rather than purely testing the parallel device. They
// correspond fairly well to the implementation, but testing the C++ directly is
// another option.

TEST(PARALLEL_DEVICE, TestBasicCPU) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*xla*/ false,
          /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
          2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  BasicTestsForTwoDevices(context.get(),
                          "/job:localhost/replica:0/task:0/device:CPU:0",
                          "/job:localhost/replica:0/task:0/device:CPU:1");
}

TEST(PARALLEL_DEVICE, TestBasicCPUAliased) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  BasicTestsForTwoDevices(context.get(),
                          "/job:localhost/replica:0/task:0/device:CPU:0",
                          "/job:localhost/replica:0/task:0/device:CPU:0");
}

TEST(PARALLEL_DEVICE, TestBasicTPUAliased) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Skip the test if no TPU is available.
  std::unique_ptr<TF_DeviceList, decltype(&TF_DeleteDeviceList)> devices(
      TFE_ContextListDevices(context.get(), status.get()), TF_DeleteDeviceList);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  bool has_tpu = false;
  for (int device_index = 0; device_index < TF_DeviceListCount(devices.get());
       ++device_index) {
    std::string device_type =
        TF_DeviceListType(devices.get(), device_index, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    if (device_type == "TPU") {
      has_tpu = true;
      break;
    }
  }
  if (has_tpu) {
    BasicTestsForTwoDevices(context.get(),
                            "/job:localhost/replica:0/task:0/device:TPU:0",
                            "/job:localhost/replica:0/task:0/device:TPU:0");
  }
}

TEST(PARALLEL_DEVICE, TestExplicitCopies) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*xla*/ false,
          /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
          2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  const char* first_device_name =
      "/job:localhost/replica:0/task:0/device:CPU:0";
  const char* second_device_name =
      "/job:localhost/replica:0/task:0/device:CPU:1";
  std::array<const char*, 2> underlying_devices{first_device_name,
                                                second_device_name};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TensorHandlePtr cpu_value(FloatTensorHandle(3., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Copying on to a parallel device is OK.
  TensorHandlePtr device_value(TFE_TensorHandleCopyToDevice(
      cpu_value.get(), context.get(), device_name, status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  const char* backing_device =
      TFE_TensorHandleBackingDeviceName(device_value.get(), status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ASSERT_EQ(std::string(device_name), backing_device);

  // Un-pack the parallel tensor to verify that the copy was successful.
  {
    std::array<TensorHandlePtr, 2> components;
    ExtractPerDeviceValues(context.get(), device_value.get(), &components,
                           status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    // The value of the original tensor is replicated on each device.
    ExpectScalarEq<float>(components[0].get(), 3.);
    ExpectScalarEq<float>(components[1].get(), 3.);

    // Verify that the mirrors are placed on the component devices.
    std::string first_device =
        TFE_TensorHandleBackingDeviceName(components[0].get(), status.get());
    ASSERT_EQ(underlying_devices[0], first_device);
    std::string second_device =
        TFE_TensorHandleBackingDeviceName(components[1].get(), status.get());
    ASSERT_EQ(underlying_devices[1], second_device);
  }

  // Copies off of parallel devices must be explicit.
  TensorHandlePtr copy_back(TFE_TensorHandleCopyToDevice(
      device_value.get(), context.get(), first_device_name, status.get()));
  ASSERT_EQ(TF_GetCode(status.get()), TF_INTERNAL);
}

TEST(PARALLEL_DEVICE, TestDifferentShapes) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*xla*/ false,
          /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
          2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create two vectors with different lengths
  std::vector<float> size_two_value{1., 2.};
  std::vector<float> size_three_value{1., 2., 3.};
  TensorHandlePtr size_two(
      VectorFloatTensorHandle(size_two_value, status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TensorHandlePtr size_three(
      VectorFloatTensorHandle(size_three_value, status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Try to combine these values into a single parallel tensor.
  std::array<TFE_TensorHandle*, 2> components{size_two.get(), size_three.get()};
  TensorHandlePtr combined_value = CreatePerDeviceValues(
      context.get(), components, device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_UNIMPLEMENTED)
      << TF_Message(status.get());
}

TEST(PARALLEL_DEVICE, TestNestedParallelDevices) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*xla*/ false,
          /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
          3),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a parallel device with two CPUs
  const char* first_device_name =
      "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> first_underlying_devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  RegisterParallelDevice(context.get(), first_device_name,
                         first_underlying_devices, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a second parallel device with the first parallel device and one
  // additional CPU.
  const char* second_device_name =
      "/job:localhost/replica:0/task:0/device:CUSTOM:1";
  std::array<const char*, 2> second_underlying_devices{
      "/job:localhost/replica:0/task:0/device:CUSTOM:0",
      "/job:localhost/replica:0/task:0/device:CPU:2"};
  RegisterParallelDevice(context.get(), second_device_name,
                         second_underlying_devices, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a tensor on the first parallel device
  TensorHandlePtr value_one(FloatTensorHandle(1., status.get()));
  TensorHandlePtr value_two(FloatTensorHandle(2., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::array<TFE_TensorHandle*, 2> components{value_one.get(), value_two.get()};
  TensorHandlePtr first_combined_value = CreatePerDeviceValues(
      context.get(), components, first_device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Nest the first parallel tensor into a second
  TensorHandlePtr value_three(FloatTensorHandle(3., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  components[0] = first_combined_value.get();
  components[1] = value_three.get();
  TensorHandlePtr second_combined_value = CreatePerDeviceValues(
      context.get(), components, second_device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TensorHandlePtr negative_one(FloatTensorHandle(3., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TensorHandlePtr multiply_result(Multiply(context.get(),
                                           second_combined_value.get(),
                                           negative_one.get(), status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Un-pack the parallel tensor to verify that the operation was
  // successful. The resulting structure should be:
  //   second_device{first_device{1. * 3., 2. * 3.}, 3. * 3.}.
  std::array<TensorHandlePtr, 2> second_components;
  ExtractPerDeviceValues(context.get(), multiply_result.get(),
                         &second_components, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  ExpectScalarEq<float>(second_components[1].get(), 9.);

  // Verify that the mirrors are placed on the component devices.
  std::string first_device = TFE_TensorHandleBackingDeviceName(
      second_components[0].get(), status.get());
  ASSERT_EQ(second_underlying_devices[0], first_device);
  std::string second_device = TFE_TensorHandleBackingDeviceName(
      second_components[1].get(), status.get());
  ASSERT_EQ(second_underlying_devices[1], second_device);

  // Un-pack the first parallel device's tensor too
  std::array<TensorHandlePtr, 2> first_components;
  ExtractPerDeviceValues(context.get(), second_components[0].get(),
                         &first_components, status.get());
  ExpectScalarEq<float>(first_components[0].get(), 3.);
  ExpectScalarEq<float>(first_components[1].get(), 6.);

  first_device = TFE_TensorHandleBackingDeviceName(first_components[0].get(),
                                                   status.get());
  ASSERT_EQ(first_underlying_devices[0], first_device);
  second_device = TFE_TensorHandleBackingDeviceName(first_components[1].get(),
                                                    status.get());
  ASSERT_EQ(first_underlying_devices[1], second_device);
}

TEST(PARALLEL_DEVICE, TestInvalidPacking) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 1> underlying_devices{
      "/job:localhost/replica:0/task:0/device:CPU:0"};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TensorHandlePtr value_one(FloatTensorHandle(1., status.get()));
  TensorHandlePtr value_two(FloatTensorHandle(2., status.get()));
  {
    // Try to pack two TensorHandles onto a parallel device with a single
    // component.
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    std::array<TFE_TensorHandle*, 2> components{value_one.get(),
                                                value_two.get()};
    TensorHandlePtr combined_value = CreatePerDeviceValues(
        context.get(), components, device_name, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_INVALID_ARGUMENT)
        << TF_Message(status.get());
  }

  {
    // Try to extract the wrong number of components from a parallel tensor
    std::array<TFE_TensorHandle*, 1> correct_components{value_one.get()};
    TensorHandlePtr combined_value = CreatePerDeviceValues(
        context.get(), correct_components, device_name, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    std::array<TensorHandlePtr, 2> incorrect_components;
    ExtractPerDeviceValues(context.get(), combined_value.get(),
                           &incorrect_components, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_INVALID_ARGUMENT)
        << TF_Message(status.get());
  }

  {
    // Try to pass a ParallelTensor to TPUReplicatedInput
    std::array<TFE_TensorHandle*, 1> correct_components{value_one.get()};
    TensorHandlePtr combined_value = CreatePerDeviceValues(
        context.get(), correct_components, device_name, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    std::array<TFE_TensorHandle*, 1> incorrect_components{combined_value.get()};
    TensorHandlePtr recombined_value = CreatePerDeviceValues(
        context.get(), incorrect_components, device_name, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_INVALID_ARGUMENT)
        << TF_Message(status.get());
  }

  {
    // Try to pass a non-parallel tensor to TPUReplicatedOutput
    std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
        TFE_NewOp(context.get(), "TPUReplicatedOutput", status.get()),
        TFE_DeleteOp);
    if (TF_GetCode(status.get()) != TF_OK) return;
    TFE_OpSetAttrInt(op.get(), "num_replicas", 1);
    TFE_OpAddInput(op.get(), value_one.get(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) return;
    TFE_OpSetDevice(op.get(), device_name, status.get());
    if (TF_GetCode(status.get()) != TF_OK) return;

    TFE_TensorHandle* result_handles;
    int num_retvals = 1;
    TFE_Execute(op.get(), &result_handles, &num_retvals, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_INVALID_ARGUMENT)
        << TF_Message(status.get());
  }
}

TensorHandlePtr CollectiveSum(TFE_Context* context, TFE_TensorHandle* input,
                              int group_size, TF_Status* status) {
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context, "CollectiveReduce", status), TFE_DeleteOp);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  const char* device = TFE_TensorHandleDeviceName(input, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetDevice(op.get(), device, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op.get(), "T", TFE_TensorHandleDataType(input));
  TFE_OpSetAttrInt(op.get(), "group_size", group_size);
  TFE_OpSetAttrInt(op.get(), "group_key", 0);
  TFE_OpSetAttrInt(op.get(), "instance_key", 0);
  const std::string merge_op("Add");
  TFE_OpSetAttrString(op.get(), "merge_op", merge_op.c_str(),
                      merge_op.length());
  const std::string final_op("Id");
  TFE_OpSetAttrString(op.get(), "final_op", final_op.c_str(),
                      final_op.length());
  TFE_OpSetAttrIntList(op.get(), "subdiv_offsets", nullptr, 0);

  TFE_OpAddInput(op.get(), input, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  TFE_TensorHandle* result_handle;
  int num_retvals = 1;
  TFE_Execute(op.get(), &result_handle, &num_retvals, status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  return TensorHandlePtr(result_handle);
}

void TestCollective(bool async) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*xla*/ false,
          /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
          2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::unique_ptr<TFE_Executor, decltype(&TFE_DeleteExecutor)> executor(
      TFE_NewExecutor(async), TFE_DeleteExecutor);
  TFE_ContextSetExecutorForThread(context.get(), executor.get());

  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Create a tensor on the parallel device
  TensorHandlePtr value_one(FloatTensorHandle(1., status.get()));
  TensorHandlePtr value_two(FloatTensorHandle(2., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::array<TFE_TensorHandle*, 2> components{value_one.get(), value_two.get()};
  TensorHandlePtr parallel_value = CreatePerDeviceValues(
      context.get(), components, device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  // Run a collective sum, so each component should now be the same.
  TensorHandlePtr reduced(
      CollectiveSum(context.get(), parallel_value.get(), 2, status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  std::array<TensorHandlePtr, 2> result_components;
  ExtractPerDeviceValues(context.get(), reduced.get(), &result_components,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ExpectScalarEq<float>(result_components[0].get(), 3.);
  ExpectScalarEq<float>(result_components[1].get(), 3.);
  // Destroying the context's default executor first isn't safe.
  context.reset();
}

TEST(PARALLEL_DEVICE, TestCollectiveSync) { TestCollective(/*async=*/false); }

// Note that ops on the parallel device currently don't execute
// asynchronously. The test is just that we don't get deadlocks.
TEST(PARALLEL_DEVICE, TestCollectiveAsync) { TestCollective(/*async=*/true); }

void RegisterCollectiveMulFunction(TFE_Context* context,
                                   const char* function_name, int group_size,
                                   TF_Status* status) {
  std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)> body(TF_NewGraph(),
                                                            TF_DeleteGraph);
  TF_OperationDescription* placeholder_desc =
      TF_NewOperation(body.get(), "Placeholder", "Placeholder");
  TF_SetAttrType(placeholder_desc, "dtype", TF_FLOAT);
  TF_Operation* placeholder_op = TF_FinishOperation(placeholder_desc, status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_Output x{placeholder_op, 0};

  TF_OperationDescription* reduce_desc =
      TF_NewOperation(body.get(), "CollectiveReduce", "CollectiveReduce");
  TF_SetAttrType(reduce_desc, "T", TF_FLOAT);
  TF_SetAttrInt(reduce_desc, "group_size", group_size);
  TF_SetAttrInt(reduce_desc, "group_key", 0);
  TF_SetAttrInt(reduce_desc, "instance_key", 0);

  const std::string merge_op("Mul");
  TF_SetAttrString(reduce_desc, "merge_op", merge_op.c_str(),
                   merge_op.length());
  const std::string final_op("Id");
  TF_SetAttrString(reduce_desc, "final_op", final_op.c_str(),
                   final_op.length());
  TF_SetAttrIntList(reduce_desc, "subdiv_offsets", nullptr, 0);
  TF_AddInput(reduce_desc, x);
  TF_Operation* reduce_op = TF_FinishOperation(reduce_desc, status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_Operation* operations[]{placeholder_op, reduce_op};
  TF_Output y{reduce_op, 0};
  const char* output_name = "y";
  std::unique_ptr<TF_Function, decltype(&TF_DeleteFunction)> function(
      TF_GraphToFunction(
          /* fn_body */ body.get(), /* fn_name */ function_name,
          /* append_hash_to_fn_name */ 0, /* num_opers */ 2,
          /* opers */ operations, /* ninputs */ 1, /* inputs */ &x,
          /* noutputs */ 1, /* outputs */ &y, /* output_names */ &output_name,
          /* opts */ nullptr, /* description */ "", /* status */ status),
      TF_DeleteFunction);
  if (TF_GetCode(status) != TF_OK) return;
  TFE_ContextAddFunction(context, function.get(), status);
}

TEST(PARALLEL_DEVICE, TestFunction) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*xla*/ false,
          /* gpu_memory_allow_growth */ true, /* num_cpu_devices */
          2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  const char* device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:0";
  std::array<const char*, 2> underlying_devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  RegisterParallelDevice(context.get(), device_name, underlying_devices,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  const char* function_name = "test_reduce_mul";
  RegisterCollectiveMulFunction(context.get(), function_name, 2, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TensorHandlePtr value_one(FloatTensorHandle(7., status.get()));
  TensorHandlePtr value_two(FloatTensorHandle(9., status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::array<TFE_TensorHandle*, 2> components{value_one.get(), value_two.get()};
  TensorHandlePtr parallel_value = CreatePerDeviceValues(
      context.get(), components, device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> op(
      TFE_NewOp(context.get(), function_name, status.get()), TFE_DeleteOp);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetDevice(op.get(), device_name, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpAddInput(op.get(), parallel_value.get(), status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  TFE_TensorHandle* raw_result_handle;
  int num_retvals = 1;
  TFE_Execute(op.get(), &raw_result_handle, &num_retvals, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TensorHandlePtr reduced(raw_result_handle);

  std::array<TensorHandlePtr, 2> result_components;
  ExtractPerDeviceValues(context.get(), reduced.get(), &result_components,
                         status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  ExpectScalarEq<float>(result_components[0].get(), 7. * 9.);
  ExpectScalarEq<float>(result_components[1].get(), 7. * 9.);

  std::string first_device = TFE_TensorHandleBackingDeviceName(
      result_components[0].get(), status.get());
  ASSERT_EQ(underlying_devices[0], first_device);
  std::string second_device = TFE_TensorHandleBackingDeviceName(
      result_components[1].get(), status.get());
  ASSERT_EQ(underlying_devices[1], second_device);
}

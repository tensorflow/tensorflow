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

#include "tensorflow/c/eager/parallel_device/parallel_device_lib.h"

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace parallel_device {

TEST(PARALLEL_DEVICE_LIB, TestOpWithError) {
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

  std::vector<std::string> devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  ParallelDevice parallel_device(std::move(devices));
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> handle_op(
      TFE_NewOp(context.get(), "VarHandleOp", status.get()), TFE_DeleteOp);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetAttrType(handle_op.get(), "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(handle_op.get(), "shape", /*dims=*/nullptr, /*num_dims=*/0,
                     status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  auto outputs =
      parallel_device.Execute(context.get(), std::vector<ParallelTensor*>(),
                              "VarHandleOp", TFE_OpGetAttrs(handle_op.get()),
                              /*expected_max_outputs=*/1, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  const std::vector<std::unique_ptr<ParallelTensor>>& handles = *outputs;
  std::vector<ParallelTensor*> handle_inputs;
  handle_inputs.reserve(handles.size());
  for (auto& handle : handles) {
    handle_inputs.push_back(handle.get());
  }
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> read_op(
      TFE_NewOp(context.get(), "ReadVariableOp", status.get()), TFE_DeleteOp);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetAttrType(read_op.get(), "dtype", TF_FLOAT);
  parallel_device.Execute(context.get(), handle_inputs, "ReadVariableOp",
                          TFE_OpGetAttrs(read_op.get()),
                          /*expected_max_outputs=*/1, status.get());
  ASSERT_FALSE(TF_GetCode(status.get()) == TF_OK);
  TF_SetStatus(status.get(), TF_OK, "");

  // Check that ops still run successfully on the device.
  parallel_device.Execute(context.get(), std::vector<ParallelTensor*>(),
                          "VarHandleOp", TFE_OpGetAttrs(handle_op.get()),
                          /*expected_max_outputs=*/1, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
}

TEST(PARALLEL_DEVICE_LIB, TestExplicitOutputShape) {
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

  std::vector<std::string> devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  ParallelDevice parallel_device(std::move(devices));
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> handle_op(
      TFE_NewOp(context.get(), "VarHandleOp", status.get()), TFE_DeleteOp);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TFE_OpSetAttrType(handle_op.get(), "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(handle_op.get(), "shape", /*dims=*/nullptr, /*num_dims=*/0,
                     status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  auto outputs = parallel_device.Execute(
      context.get(), std::vector<ParallelTensor*>(), "VarHandleOp",
      TFE_OpGetAttrs(handle_op.get()), {PartialTensorShape({})}, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  const std::vector<std::unique_ptr<ParallelTensor>>& handles = *outputs;
  EXPECT_EQ(0, handles[0]->shape().size());
}

}  // namespace parallel_device
}  // namespace tensorflow

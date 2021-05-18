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
#include "tensorflow/c/eager/parallel_device/parallel_device_testlib.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace parallel_device {

using ::testing::HasSubstr;

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
  CancellationManager cancellation_manager;
  parallel_device.StartExecute(context.get(), std::vector<ParallelTensor*>(),
                               "VarHandleOp", TFE_OpGetAttrs(handle_op.get()),
                               /*expected_max_outputs=*/1,
                               cancellation_manager);
  auto outputs = parallel_device.Join(
      /*expected_output_shapes=*/{PartialTensorShape({})}, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  const std::vector<std::unique_ptr<ParallelTensor>>& handles = *outputs;
  const std::vector<int64_t>* shape;
  Status s = handles[0]->Shape(&shape);
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(0, shape->size());
}

TEST(PARALLEL_DEVICE_LIB, TestCancelOnError) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*enable_xla_compilation=*/false,
          /*gpu_memory_allow_growth=*/true, /*num_cpu_devices=*/2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  std::vector<std::string> devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  ParallelDevice parallel_device(devices);
  const FunctionDef assert_and_collective = FunctionDefHelper::Define(
      // Name
      "AssertAndCollective",
      // Args
      {"x: float", "condition: bool"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"assert"},
           "Assert",
           {"condition", "x"},
           {{"T", std::vector<DataType>{DT_FLOAT}}}},
          {{"y"},
           "CollectiveReduce",
           {"x"},
           {{"T", DT_FLOAT},
            {"group_size", static_cast<int>(devices.size())},
            {"group_key", 0},
            {"instance_key", 0},
            {"merge_op", "Add"},
            {"final_op", "Id"},
            {"subdiv_offsets", std::vector<int>()}},
           /*dep=*/{"assert"}},
      });
  TF_ASSERT_OK(ContextFromInterface(unwrap(context.get()))
                   ->AddFunctionDef(assert_and_collective));

  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> call_op(
      TFE_NewOp(context.get(), "AssertAndCollective", status.get()),
      TFE_DeleteOp);
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
  std::unique_ptr<ParallelTensor> reduced_values =
      parallel_device.ScalarsFromSequence<float>({1.0, 2.0}, context.get(),
                                                 status.get());
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
  std::unique_ptr<ParallelTensor> run_collective =
      parallel_device.ScalarsFromSequence<bool>({true, true}, context.get(),
                                                status.get());
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
  auto outputs = parallel_device.Execute(
      context.get(), {reduced_values.get(), run_collective.get()},
      "AssertAndCollective", TFE_OpGetAttrs(call_op.get()),
      /*expected_max_outputs=*/1, status.get());
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
  ASSERT_EQ(outputs->size(), 1);
  ParallelTensor* parallel_result = (*outputs)[0].get();
  ExpectScalarEq<float>(parallel_result->tensor(0), 3.);
  ExpectScalarEq<float>(parallel_result->tensor(1), 3.);

  run_collective = parallel_device.ScalarsFromSequence<bool>(
      {true, false}, context.get(), status.get());
  parallel_device.Execute(context.get(),
                          {reduced_values.get(), run_collective.get()},
                          "AssertAndCollective", TFE_OpGetAttrs(call_op.get()),
                          /*expected_max_outputs=*/1, status.get());
  EXPECT_NE(TF_GetCode(status.get()), TF_CANCELLED);
  EXPECT_EQ(TF_GetCode(status.get()), TF_INVALID_ARGUMENT);
  EXPECT_THAT(TF_Message(status.get()), HasSubstr("assertion failed"));

  // Note that future collectives with the same context do not work at the
  // moment; once canceled, the collective executor requires the program to be
  // restarted / context to be reset.
}

TEST(PARALLEL_DEVICE_LIB, TestDifferentShapes) {
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
  TensorHandlePtr two_vector = VectorFloatTensorHandle({3., 4.}, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  TensorHandlePtr three_vector =
      VectorFloatTensorHandle({5., 6., 7.}, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

  std::vector<TensorHandlePtr> vector_handles;
  vector_handles.reserve(2);
  vector_handles.push_back(std::move(two_vector));
  vector_handles.push_back(std::move(three_vector));
  std::unique_ptr<ParallelTensor> unknown_length_vector =
      ParallelTensor::FromTensorHandles(
          parallel_device, std::move(vector_handles), status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  const std::vector<int64_t>* shape;
  Status s = unknown_length_vector->Shape(&shape);
  EXPECT_FALSE(s.ok());

  TensorHandlePtr scalar = FloatTensorHandle(2., status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  two_vector = VectorFloatTensorHandle({3., 4.}, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
  std::vector<TensorHandlePtr> mixed_handles;
  mixed_handles.reserve(2);
  mixed_handles.push_back(std::move(scalar));
  mixed_handles.push_back(std::move(two_vector));
  std::unique_ptr<ParallelTensor> unknown_dims_vector =
      ParallelTensor::FromTensorHandles(parallel_device,
                                        std::move(mixed_handles), status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  // Can't take the shape of a parallel tensor with varying numbers of axes, but
  // running operations on them is OK.
  s = unknown_length_vector->Shape(&shape);
  EXPECT_FALSE(s.ok());
  std::unique_ptr<TFE_Op, decltype(&TFE_DeleteOp)> size_op(
      TFE_NewOp(context.get(), "Size", status.get()), TFE_DeleteOp);
  auto result = parallel_device.Execute(
      context.get(), {unknown_dims_vector.get()}, "Size",
      TFE_OpGetAttrs(size_op.get()), 1, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  s = (*result)[0]->Shape(&shape);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  EXPECT_EQ(0, shape->size());
}

TEST(PARALLEL_DEVICE_LIB, TestScalarsFromSequence) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_CreateConfig(
          /*enable_xla_compilation=*/false,
          /*gpu_memory_allow_growth=*/true, /*num_cpu_devices=*/2),
      TF_DeleteBuffer);
  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status.get()), TFE_DeleteContext);
  ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());

  std::vector<std::string> devices{
      "/job:localhost/replica:0/task:0/device:CPU:0",
      "/job:localhost/replica:0/task:0/device:CPU:1"};
  ParallelDevice parallel_device(std::move(devices));
  {
    std::unique_ptr<ParallelTensor> float_tensors =
        parallel_device.ScalarsFromSequence<float>({10.0, 11.0}, context.get(),
                                                   status.get());
    ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    ExpectScalarEq<float>(float_tensors->tensor(0), 10.0);
    ExpectScalarEq<float>(float_tensors->tensor(1), 11.0);
  }

  {
    std::unique_ptr<ParallelTensor> int_tensors =
        parallel_device.ScalarsFromSequence<int>({5, 6}, context.get(),
                                                 status.get());
    ASSERT_EQ(TF_GetCode(status.get()), TF_OK) << TF_Message(status.get());
    ExpectScalarEq<int>(int_tensors->tensor(0), 5);
    ExpectScalarEq<int>(int_tensors->tensor(1), 6);
  }
}

}  // namespace parallel_device
}  // namespace tensorflow

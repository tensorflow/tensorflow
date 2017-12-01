/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api.h"

#include <string.h>
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

using tensorflow::string;

namespace {

TFE_TensorHandle* TestMatrixTensorHandle() {
  int64_t dims[] = {2, 2};
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  TF_Tensor* t = TF_AllocateTensor(
      TF_FLOAT, &dims[0], sizeof(dims) / sizeof(int64_t), sizeof(data));
  memcpy(TF_TensorData(t), &data[0], TF_TensorByteSize(t));
  TF_Status* status = TF_NewStatus();
  TFE_TensorHandle* th = TFE_NewTensorHandle(t, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteTensor(t);
  TF_DeleteStatus(status);
  return th;
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
  TFE_OpSetAttrBool(op, "transpose_a", 0);
  TFE_OpSetAttrBool(op, "transpose_b", 0);
  TFE_OpSetAttrType(op, "T", TFE_TensorHandleDataType(a));

  return op;
}

void BM_InitOp(int iters) {
  tensorflow::testing::StopTiming();
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TFE_Op* matmul = MatMulOp(ctx, m, m);
    TFE_DeleteOp(matmul);
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_InitOp);

void BM_Execute(int iters) {
  tensorflow::testing::StopTiming();
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);
  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TFE_Execute(matmul, &retvals[0], &num_retvals, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_Execute);

TEST(CAPI, Context) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  TFE_DeleteContextOptions(opts);

  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteContext(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  const int num_devices = TF_DeviceListCount(devices);
  EXPECT_GE(num_devices, 1) << "At least one CPU device should exist";
  for (int i = 0; i < num_devices; ++i) {
    EXPECT_NE("", TF_DeviceListName(devices, i, status)) << i;
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  TF_DeleteDeviceList(devices);
  TF_DeleteStatus(status);
}

TEST(CAPI, TensorHandle) {
  TFE_TensorHandle* h = TestMatrixTensorHandle();
  EXPECT_EQ(TF_FLOAT, TFE_TensorHandleDataType(h));

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TF_Tensor* t = TFE_TensorHandleResolve(h, status.get());
  ASSERT_EQ(16, TF_TensorByteSize(t));
  float data[4] = {0};
  memcpy(&data[0], TF_TensorData(t), TF_TensorByteSize(t));
  EXPECT_EQ(1.0, data[0]);
  EXPECT_EQ(2.0, data[1]);
  EXPECT_EQ(3.0, data[2]);
  EXPECT_EQ(4.0, data[3]);
  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(h);
}

TEST(CAPI, TensorHandleCopyBetweenDevices) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  TF_Tensor* t = TFE_TensorHandleResolve(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  const int num_devices = TF_DeviceListCount(devices);

  const char* kCPUDevice = "CPU:0";
  for (int i = 0; i < num_devices; ++i) {
    const string name(TF_DeviceListName(devices, i, status.get()));
    if (TF_GetCode(status.get()) != TF_OK) {
      ADD_FAILURE() << i << " -- " << TF_Message(status.get());
      continue;
    }
    auto tag = tensorflow::strings::StrCat("Device #", i, " (", name, ")");
    // Copy to device
    TFE_TensorHandle* hdevice =
        TFE_TensorHandleCopyToDevice(hcpu, ctx, name.c_str(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      ADD_FAILURE() << tag << " -- " << TF_Message(status.get());
      continue;
    }
    // Copy from device to the same device.
    TFE_TensorHandle* hdevice2 =
        TFE_TensorHandleCopyToDevice(hdevice, ctx, name.c_str(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      ADD_FAILURE() << tag << " -- " << TF_Message(status.get());
      continue;
    }
    TFE_DeleteTensorHandle(hdevice);
    // Copy back to CPU
    TFE_TensorHandle* hcopy =
        TFE_TensorHandleCopyToDevice(hdevice2, ctx, kCPUDevice, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      ADD_FAILURE() << tag << " -- " << TF_Message(status.get());
      continue;
    }
    TFE_DeleteTensorHandle(hdevice2);

    // Ensure that the contents are the same!
    TF_Tensor* tcopy = TFE_TensorHandleResolve(hcopy, status.get());
    TFE_DeleteTensorHandle(hcopy);
    if (TF_GetCode(status.get()) != TF_OK) {
      ADD_FAILURE() << tag;
      continue;
    }
    EXPECT_EQ(TF_TensorByteSize(t), TF_TensorByteSize(tcopy)) << tag;
    EXPECT_EQ(
        0, memcmp(TF_TensorData(t), TF_TensorData(tcopy), TF_TensorByteSize(t)))
        << tag;
    TF_DeleteTensor(tcopy);
  }

  TF_DeleteDeviceList(devices);
  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(hcpu);
  TFE_DeleteContext(ctx, status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
}

TEST(CAPI, TensorHandleSilentCopy) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  TF_Tensor* t = TFE_TensorHandleResolve(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  const int num_devices = TF_DeviceListCount(devices);

  // Disable the test if no GPU is present.
  if (num_devices > 1) {
    const int device_to_use = 1;
    const string name(TF_DeviceListName(devices, device_to_use, status.get()));
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    TFE_TensorHandle* hgpu =
        TFE_TensorHandleCopyToDevice(hcpu, ctx, name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    TFE_Op* matmul = MatMulOp(ctx, hcpu, hgpu);
    TFE_OpSetDevice(matmul, name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(matmul, &retvals[0], &num_retvals, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_DeleteOp(matmul);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteTensorHandle(hgpu);
  }

  TF_DeleteDeviceList(devices);
  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(hcpu);
  TFE_DeleteContext(ctx, status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
}

TEST(CAPI, Execute) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);
  TFE_TensorHandle* retvals[2] = {nullptr};
  int num_retvals = 2;  // Should be reduced to 1 by the TFE_Execute call.
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  TFE_DeleteTensorHandle(retvals[0]);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(7, product[0]);
  EXPECT_EQ(10, product[1]);
  EXPECT_EQ(15, product[2]);
  EXPECT_EQ(22, product[3]);
  TF_DeleteStatus(status);
}

TEST(CAPI, Function) {
  // First create a simple identity function.
  TF_Graph* function_graph = TF_NewGraph();
  TF_OperationDescription* arg_descr =
      TF_NewOperation(function_graph, "Placeholder", "arg");
  TF_SetAttrType(arg_descr, "dtype", TF_INT32);
  TF_Status* status = TF_NewStatus();
  TF_Operation* arg = TF_FinishOperation(arg_descr, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_OperationDescription* id_descr =
      TF_NewOperation(function_graph, "Identity", "id");
  TF_SetAttrType(id_descr, "T", TF_INT32);
  TF_AddInput(id_descr, {arg, 0});
  TF_Operation* id = TF_FinishOperation(id_descr, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_Output input{arg, 0};
  TF_Output output{id, 0};
  TF_Function* fn =
      TF_GraphToFunction(function_graph, "ident", 0, 1, &id, 1, &input, 1,
                         &output, nullptr, nullptr, "test", status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteGraph(function_graph);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TFE_DeleteContextOptions(opts);
  TFE_ContextAddFunction(ctx, fn, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteFunction(fn);

  TF_Tensor* t =
      TF_AllocateTensor(TF_INT32, nullptr, 0, 1 * sizeof(tensorflow::int32));
  *reinterpret_cast<tensorflow::int32*>(TF_TensorData(t)) = 42;
  TFE_TensorHandle* h = TFE_NewTensorHandle(t, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteTensor(t);

  TFE_Op* op = TFE_NewOp(ctx, "ident", status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TFE_OpAddInput(op, h, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> result;
  result.push_back(nullptr);
  int num_retvals = 1;
  TFE_Execute(op, result.data(), &num_retvals, status);
  TFE_DeleteOp(op);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  ASSERT_EQ(num_retvals, 1);

  TF_Tensor* r = TFE_TensorHandleResolve(result[0], status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  EXPECT_EQ(*reinterpret_cast<tensorflow::int32*>(TF_TensorData(r)), 42);
  TFE_DeleteTensorHandle(h);
  TF_DeleteTensor(r);
  TFE_DeleteTensorHandle(result[0]);
  TFE_DeleteContext(ctx, status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TF_DeleteStatus(status);
}

string MatMulFunction() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'MatMulFunction'"
      "      input_arg {"
      "        name: 'a'"
      "        type: DT_FLOAT"
      "      }"
      "      output_arg {"
      "        name: 'm'"
      "        type: DT_FLOAT"
      "      }"
      "    }"
      "    node_def {"
      "      name: 'matmul'"
      "      op: 'MatMul'"
      "      input: 'a'"
      "      input: 'a'"
      "      attr {"
      "        key: 'T'"
      "        value {"
      "          type: DT_FLOAT"
      "        }"
      "      }"
      "    }"
      "    ret {"
      "      key: 'm'"
      "      value: 'matmul:product'"
      "    }",
      &def));
  return def.SerializeAsString();
}

TEST(CAPI, FunctionDefAndExecute) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  string function_def = MatMulFunction();
  TFE_ContextAddFunctionDef(ctx, function_def.data(), function_def.size(),
                            status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_TensorHandle* retval[1] = {nullptr};
  int num_retvals = 1;
  TFE_Op* op = TFE_NewOp(ctx, "MatMulFunction", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, m, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Execute(op, &retval[0], &num_retvals, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);
  TFE_DeleteOp(op);
  TFE_DeleteTensorHandle(m);
  TF_Tensor* t = TFE_TensorHandleResolve(retval[0], status);
  TFE_DeleteTensorHandle(retval[0]);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float product[4] = {0};
  EXPECT_EQ(sizeof(product), TF_TensorByteSize(t));
  memcpy(&product[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(7, product[0]);
  EXPECT_EQ(10, product[1]);
  EXPECT_EQ(15, product[2]);
  EXPECT_EQ(22, product[3]);
  TFE_DeleteContext(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}

void BM_ExecuteFunction(int iters) {
  tensorflow::testing::StopTiming();
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  string function_def = MatMulFunction();
  TFE_ContextAddFunctionDef(ctx, function_def.data(), function_def.size(),
                            status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = TFE_NewOp(ctx, "MatMulFunction", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(matmul, m, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* retval[1] = {nullptr};
  int num_retvals = 1;
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TFE_Execute(matmul, &retval[0], &num_retvals, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteTensorHandle(m);
  TFE_DeleteTensorHandle(retval[0]);
  TFE_DeleteContext(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_ExecuteFunction);

TFE_TensorHandle* CreateVariable(TFE_Context* ctx, float value,
                                 TF_Status* status) {
  // Create the variable handle.
  TFE_Op* op = TFE_NewOp(ctx, "VarHandleOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op, "shape", {}, 0, status);
  TFE_OpSetAttrString(op, "container", "");
  TFE_OpSetAttrString(op, "shared_name", "");
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_TensorHandle* var_handle = nullptr;
  int num_retvals = 1;
  TFE_Execute(op, &var_handle, &num_retvals, status);
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

  return var_handle;
}

TEST(CAPI, Variables) {
  // Variables use resource handles, so this is really a test for resource
  // tensor handling.
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* var_handle = CreateVariable(ctx, 12.0, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_Op* op = TFE_NewOp(ctx, "ReadVariableOp", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var_handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  int num_retvals = 1;
  TFE_TensorHandle* value_handle = nullptr;
  TFE_Execute(op, &value_handle, &num_retvals, status);
  TFE_DeleteOp(op);

  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);
  EXPECT_EQ(TF_FLOAT, TFE_TensorHandleDataType(value_handle));
  EXPECT_EQ(0, TFE_TensorHandleNumDims(value_handle));
  float value = 0.0f;
  TF_Tensor* t = TFE_TensorHandleResolve(value_handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(sizeof(float), TF_TensorByteSize(t));
  memcpy(&value, TF_TensorData(t), sizeof(float));
  TF_DeleteTensor(t);
  EXPECT_EQ(12.0, value);

  TFE_DeleteTensorHandle(var_handle);
  TFE_DeleteTensorHandle(value_handle);
  TFE_DeleteContext(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}

void BM_ReadVariable(int iters) {
  tensorflow::testing::StopTiming();
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* var_handle = CreateVariable(ctx, 5.0, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_Op* op = TFE_NewOp(ctx, "ReadVariableOp", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpAddInput(op, var_handle, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  int num_retvals = 1;
  TFE_TensorHandle* h = nullptr;
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TFE_Execute(op, &h, &num_retvals, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    CHECK_EQ(1, num_retvals);
    CHECK(h);
    CHECK_EQ(TF_FLOAT, TFE_TensorHandleDataType(h));
    CHECK_EQ(0, TFE_TensorHandleNumDims(h));
    h = nullptr;
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteOp(op);

  TFE_DeleteTensorHandle(var_handle);
  TFE_DeleteContext(ctx, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_ReadVariable);

}  // namespace

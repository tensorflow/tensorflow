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

#include "absl/strings/match.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

using tensorflow::string;

namespace {

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
  TFE_DeleteContext(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_InitOp);

void BM_Execute(int iters, int async) {
  tensorflow::testing::StopTiming();
  tensorflow::testing::SetLabel(async ? "ExecuteAsync" : "Execute");
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
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
  if (async) {
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_Execute)->Arg(0)->Arg(1);

void BM_Execute_Identity(int iters, int async) {
  tensorflow::testing::StopTiming();
  tensorflow::testing::SetLabel(async ? "ExecuteIdentityAsync"
                                      : "ExecuteIdentity");
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* identity = IdentityOp(ctx, m);
  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  tensorflow::testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    TFE_Execute(identity, &retvals[0], &num_retvals, status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  }
  if (async) {
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteOp(identity);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_Execute_Identity)->Arg(0)->Arg(1);

TEST(CAPI, Context) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  TFE_DeleteContextOptions(opts);

  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteContext(ctx);
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

void TensorHandleCopyBetweenDevices(bool async) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
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
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TensorHandleCopyBetweenDevices) {
  TensorHandleCopyBetweenDevices(false);
}

TEST(CAPI, TensorHandleCopyBetweenDevicesAsync) {
  TensorHandleCopyBetweenDevices(true);
}

void TensorHandleCopyBetweenDevicesError(bool async) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  const char* kErrorDevice = "NoSuchDevice:0";
  TFE_TensorHandle* hdevice =
      TFE_TensorHandleCopyToDevice(hcpu, ctx, kErrorDevice, status.get());
  EXPECT_NE(TF_OK, TF_GetCode(status.get()));
  const char* msg = "NoSuchDevice:0 unknown device";
  EXPECT_TRUE(strstr(TF_Message(status.get()), msg) != nullptr)
      << TF_Message(status.get());
  TF_SetStatus(status.get(), TF_OK, "");
  const char* kCPUDevice = "CPU:0";
  TFE_TensorHandle* hcopy =
      TFE_TensorHandleCopyToDevice(hcpu, ctx, kCPUDevice, status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status.get());
  EXPECT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteExecutor(executor);
  TFE_DeleteTensorHandle(hcopy);
  TFE_DeleteTensorHandle(hcpu);
  if (hdevice != nullptr) TFE_DeleteTensorHandle(hdevice);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TensorHandleCopyBetweenDevicesError) {
  TensorHandleCopyBetweenDevicesError(false);
}

TEST(CAPI, TensorHandleCopyBetweenDevicesErrorAsync) {
  TensorHandleCopyBetweenDevicesError(true);
}

void TensorHandleCopyBetweenTwoGPUDevices(bool async) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  TF_Tensor* t = TFE_TensorHandleResolve(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_DeviceList* devices = TFE_ContextListDevices(ctx, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  const int num_devices = TF_DeviceListCount(devices);
  bool has_gpu0 = false;
  bool has_gpu1 = false;
  for (int i = 0; i < num_devices; ++i) {
    const char* dev = TF_DeviceListName(devices, i, status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    string device_name(dev);
    if (device_name.find("GPU:0") != string::npos) {
      has_gpu0 = true;
    }
    if (device_name.find("GPU:1") != string::npos) {
      has_gpu1 = true;
    }
  }

  const char* kCPUDevice = "CPU:0";
  if (!has_gpu0 || !has_gpu1) {
    TF_DeleteDeviceList(devices);
    TF_DeleteTensor(t);
    TFE_DeleteTensorHandle(hcpu);
    TFE_DeleteContext(ctx);
    return;
  }
  const string gpu_1_name(TF_DeviceListName(devices, 1, status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  const string gpu_2_name(TF_DeviceListName(devices, 2, status.get()));
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  TFE_TensorHandle* hdevice =
      TFE_TensorHandleCopyToDevice(hcpu, ctx, gpu_1_name.c_str(), status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);

  TFE_TensorHandle* hdevice2 = TFE_TensorHandleCopyToDevice(
      hdevice, ctx, gpu_2_name.c_str(), status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  TFE_DeleteTensorHandle(hdevice);
  // Copy back to CPU
  TFE_TensorHandle* hcopy =
      TFE_TensorHandleCopyToDevice(hdevice2, ctx, kCPUDevice, status.get());
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  TFE_DeleteTensorHandle(hdevice2);

  // Ensure that the contents are the same!
  TF_Tensor* tcopy = TFE_TensorHandleResolve(hcopy, status.get());
  TFE_DeleteTensorHandle(hcopy);
  ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK);
  EXPECT_EQ(TF_TensorByteSize(t), TF_TensorByteSize(tcopy));
  EXPECT_EQ(
      0, memcmp(TF_TensorData(t), TF_TensorData(tcopy), TF_TensorByteSize(t)));
  TF_DeleteTensor(tcopy);

  TF_DeleteDeviceList(devices);
  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(hcpu);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TensorHandleCopyBetweenTwoGPUDevices) {
  TensorHandleCopyBetweenTwoGPUDevices(false);
}

TEST(CAPI, TensorHandleCopyBetweenTwoGPUDevicesAsync) {
  TensorHandleCopyBetweenTwoGPUDevices(true);
}

void TensorHandleSilentCopy(bool async) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetDevicePlacementPolicy(opts, TFE_DEVICE_PLACEMENT_SILENT);
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  TF_Tensor* t = TFE_TensorHandleResolve(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Disable the test if no GPU is present.
  string gpu_device_name;
  if (GetDeviceName(ctx, &gpu_device_name, "GPU")) {
    TFE_TensorHandle* hgpu = TFE_TensorHandleCopyToDevice(
        hcpu, ctx, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    TFE_Op* matmul = MatMulOp(ctx, hcpu, hgpu);
    TFE_OpSetDevice(matmul, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(matmul, &retvals[0], &num_retvals, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_DeleteOp(matmul);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteTensorHandle(hgpu);
  }

  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(hcpu);
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TensorHandleSilentCopy) { TensorHandleSilentCopy(false); }
TEST(CAPI, TensorHandleSilentCopyAsync) { TensorHandleSilentCopy(true); }

void TensorHandleSilentCopyLocal(bool async) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                             TFE_DEVICE_PLACEMENT_EXPLICIT);
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_ContextSetThreadLocalDevicePlacementPolicy(ctx,
                                                 TFE_DEVICE_PLACEMENT_SILENT);
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  TF_Tensor* t = TFE_TensorHandleResolve(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  // Disable the test if no GPU is present.
  string gpu_device_name;
  if (GetDeviceName(ctx, &gpu_device_name, "GPU")) {
    TFE_TensorHandle* hgpu = TFE_TensorHandleCopyToDevice(
        hcpu, ctx, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    TFE_Op* matmul = MatMulOp(ctx, hcpu, hgpu);
    TFE_OpSetDevice(matmul, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(matmul, &retvals[0], &num_retvals, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_DeleteOp(matmul);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteTensorHandle(hgpu);
  }

  TF_DeleteTensor(t);
  TFE_DeleteTensorHandle(hcpu);
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);
}
TEST(CAPI, TensorHandleSilentCopyLocal) { TensorHandleSilentCopyLocal(false); }
TEST(CAPI, TensorHandleSilentCopyLocalAsync) {
  TensorHandleSilentCopyLocal(true);
}

void SetAndGetOpDevices(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);

  // Disable the test if no GPU is present.
  string gpu_device_name;
  if (GetDeviceName(ctx, &gpu_device_name, "GPU")) {
    TFE_OpSetDevice(matmul, "GPU:0", status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    const char* device_name = TFE_OpGetDevice(matmul, status);
    ASSERT_TRUE(strstr(device_name, "GPU:0") != nullptr);

    TFE_OpSetDevice(matmul, "CPU:0", status);
    ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
    device_name = TFE_OpGetDevice(matmul, status);
    ASSERT_TRUE(strstr(device_name, "CPU:0") != nullptr);
  }

  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TFE_DeleteContext(ctx);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}

TEST(CAPI, TensorHandleNullptr) {
  TFE_TensorHandle* h = nullptr;
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  TF_Tensor* t = TFE_TensorHandleResolve(h, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(t, nullptr);
  ASSERT_EQ("The passed in handle is a nullptr",
            string(TF_Message(status.get())));

  TF_SetStatus(status.get(), TF_OK, "");

  const char* device_name = TFE_TensorHandleDeviceName(h, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(device_name, nullptr);
  ASSERT_EQ("The passed in handle is a nullptr",
            string(TF_Message(status.get())));

  TF_SetStatus(status.get(), TF_OK, "");

  device_name = TFE_TensorHandleBackingDeviceName(h, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(device_name, nullptr);
  ASSERT_EQ("The passed in handle is a nullptr",
            string(TF_Message(status.get())));

  TF_SetStatus(status.get(), TF_OK, "");

  int num_dims = TFE_TensorHandleNumDims(h, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(num_dims, -1);
  ASSERT_EQ("The passed in handle is a nullptr",
            string(TF_Message(status.get())));

  TF_SetStatus(status.get(), TF_OK, "");

  int dim = TFE_TensorHandleDim(h, 0, status.get());
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status.get()));
  ASSERT_EQ(dim, -1);
  ASSERT_EQ("The passed in handle is a nullptr",
            string(TF_Message(status.get())));
}

TEST(CAPI, TensorHandleDevices) {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status.get());
  TFE_DeleteContextOptions(opts);
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TFE_TensorHandle* hcpu = TestMatrixTensorHandle();
  const char* device_name = TFE_TensorHandleDeviceName(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_TRUE(absl::StrContains(device_name, "CPU:0")) << device_name;
  const char* backing_device_name =
      TFE_TensorHandleBackingDeviceName(hcpu, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  ASSERT_TRUE(absl::StrContains(backing_device_name, "CPU:0"))
      << backing_device_name;

  // Disable the test if no GPU is present.
  string gpu_device_name;
  if (GetDeviceName(ctx, &gpu_device_name, "GPU")) {
    TFE_TensorHandle* hgpu = TFE_TensorHandleCopyToDevice(
        hcpu, ctx, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    TFE_Op* shape_op = ShapeOp(ctx, hgpu);
    TFE_OpSetDevice(shape_op, gpu_device_name.c_str(), status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(shape_op, &retvals[0], &num_retvals, status.get());
    ASSERT_TRUE(TF_GetCode(status.get()) == TF_OK) << TF_Message(status.get());

    // .device of shape is GPU since the op is executed on GPU
    device_name = TFE_TensorHandleDeviceName(retvals[0], status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    ASSERT_TRUE(absl::StrContains(device_name, "GPU:0")) << device_name;

    // .backing_device of shape is CPU since the tensor is backed by CPU
    backing_device_name =
        TFE_TensorHandleBackingDeviceName(retvals[0], status.get());
    ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
    ASSERT_TRUE(absl::StrContains(backing_device_name, "CPU:0"))
        << backing_device_name;

    TFE_DeleteOp(shape_op);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteTensorHandle(hgpu);
  }

  TFE_DeleteTensorHandle(hcpu);
  TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
  TFE_ExecutorWaitForAllPendingNodes(executor, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());
  TFE_DeleteExecutor(executor);
  TFE_DeleteContext(ctx);
}

void Execute_MatMul_CPU(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);
  TFE_TensorHandle* retvals[2] = {nullptr, nullptr};
  int num_retvals = 2;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(1, num_retvals);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteContext(ctx);
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
TEST(CAPI, Execute_MatMul_CPU) { Execute_MatMul_CPU(false); }
TEST(CAPI, Execute_MatMul_CPUAsync) { Execute_MatMul_CPU(true); }

void Execute_MatMul_CPU_Runtime_Error(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m1 = TestMatrixTensorHandle();
  TFE_TensorHandle* m2 = DoubleTestMatrixTensorHandle3X2();
  TFE_Op* matmul = MatMulOp(ctx, m1, m2);
  TFE_OpSetDevice(matmul, "/job:localhost/replica:0/task:0/device:CPU:0",
                  status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_Op* matmul2 = MatMulOp(ctx, m1, m1);
  TFE_OpSetDevice(matmul2, "/job:localhost/replica:0/task:0/device:CPU:0",
                  status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  TFE_DeleteOp(matmul);
  if (!async) {
    EXPECT_NE(TF_OK, TF_GetCode(status));
  } else {
    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    EXPECT_NE(TF_OK, TF_GetCode(status));
    EXPECT_EQ(nullptr, t);
    const char* msg = "Matrix size-incompatible: In[0]: [2,2], In[1]: [3,2]";
    EXPECT_TRUE(strstr(TF_Message(status), msg) != nullptr)
        << TF_Message(status);
    // Since error is not cleared, the following copy with correct device will
    // still fail.
    TF_SetStatus(status, TF_OK, "");
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    EXPECT_NE(TF_OK, TF_GetCode(status));
    TF_SetStatus(status, TF_OK, "");
    retvals[0] = nullptr;
    TFE_Execute(matmul2, &retvals[0], &num_retvals, status);
    EXPECT_NE(TF_OK, TF_GetCode(status));
    TFE_ExecutorClearError(executor);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);
  }
  // Following works in async mode since TFE_ContextAsyncClearError was called.
  TF_SetStatus(status, TF_OK, "");
  if (retvals[0] != nullptr) {
    TFE_DeleteTensorHandle(retvals[0]);
  }
  retvals[0] = nullptr;
  TFE_Execute(matmul2, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  EXPECT_EQ(TF_OK, TF_GetCode(status));
  TF_DeleteTensor(t);
  TFE_DeleteOp(matmul2);
  TFE_DeleteTensorHandle(m1);
  TFE_DeleteTensorHandle(m2);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}
TEST(CAPI, Execute_MatMul_CPU_Runtime_Error) {
  Execute_MatMul_CPU_Runtime_Error(false);
}
TEST(CAPI, Execute_MatMul_CPU_Runtime_ErrorAsync) {
  Execute_MatMul_CPU_Runtime_Error(true);
}

void Execute_MatMul_CPU_Type_Error(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m1 = TestMatrixTensorHandle();
  TFE_TensorHandle* m2 = DoubleTestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m1, m2);
  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_NE(TF_OK, TF_GetCode(status));
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m1);
  TFE_DeleteTensorHandle(m2);
  if (retvals[0] != nullptr) {
    TFE_DeleteTensorHandle(retvals[0]);
  }
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}

TEST(CAPI, Execute_MatMul_CPU_Type_Error) {
  Execute_MatMul_CPU_Type_Error(false);
}
TEST(CAPI, Execute_MatMul_CPU_Type_ErrorAsync) {
  Execute_MatMul_CPU_Type_Error(true);
}
TEST(CAPI, Execute_Min_CPU) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input = TestMatrixTensorHandle();
  TFE_TensorHandle* axis = TestAxisTensorHandle();
  TFE_Op* minOp = MinOp(ctx, input, axis);
  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(minOp, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(minOp);
  TFE_DeleteTensorHandle(input);
  TFE_DeleteTensorHandle(axis);
  ASSERT_EQ(1, num_retvals);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteTensorHandle(retvals[0]);
  float output[2] = {0};
  EXPECT_EQ(sizeof(output), TF_TensorByteSize(t));
  memcpy(&output[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(1, output[0]);
  EXPECT_EQ(3, output[1]);
  TFE_DeleteContext(ctx);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}

#ifdef TENSORFLOW_EAGER_USE_XLA
void Execute_MatMul_XLA_CPU(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);

  TFE_OpSetXLACompilation(matmul, true);

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  // Running a primitive TF operator via XLA is not yet supported.
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  EXPECT_EQ(1, num_retvals);

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
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}
TEST(CAPI, Execute_MatMul_XLA_CPU) { Execute_MatMul_XLA_CPU(false); }
TEST(CAPI, Execute_MatMul_XLA_CPUAsync) { Execute_MatMul_XLA_CPU(true); }

void Execute_Min_XLA_CPU(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input = TestMatrixTensorHandle();
  TFE_TensorHandle* axis = TestAxisTensorHandle();
  TFE_Op* minOp = MinOp(ctx, input, axis);

  TFE_OpSetXLACompilation(minOp, true);

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(minOp, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(minOp);
  TFE_DeleteTensorHandle(input);
  TFE_DeleteTensorHandle(axis);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  TFE_DeleteTensorHandle(retvals[0]);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float output[2] = {0};
  EXPECT_EQ(sizeof(output), TF_TensorByteSize(t));
  memcpy(&output[0], TF_TensorData(t), TF_TensorByteSize(t));
  TF_DeleteTensor(t);
  EXPECT_EQ(1, output[0]);
  EXPECT_EQ(3, output[1]);
  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}
TEST(CAPI, Execute_Min_XLA_CPU) { Execute_Min_XLA_CPU(false); }
TEST(CAPI, Execute_Min_XLA_CPUAsync) { Execute_Min_XLA_CPU(true); }
#endif  // TENSORFLOW_EAGER_USE_XLA

void ExecuteWithTracing(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  TFE_ContextEnableRunMetadata(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* m = TestMatrixTensorHandle();
  TFE_Op* matmul = MatMulOp(ctx, m, m);
  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(matmul, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteOp(matmul);
  TFE_DeleteTensorHandle(m);
  TF_Buffer* b = TF_NewBuffer();
  TFE_ContextExportRunMetadata(ctx, b, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  tensorflow::RunMetadata rm;
  EXPECT_TRUE(
      rm.ParseFromString({reinterpret_cast<const char*>(b->data), b->length}));
  TF_DeleteBuffer(b);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);

  TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteContext(ctx);
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
TEST(CAPI, ExecuteWithTracing) { ExecuteWithTracing(false); }
TEST(CAPI, ExecuteWithTracingAsync) { ExecuteWithTracing(true); }

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

void FunctionDefAndExecute(bool async) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  string function_def = MatMulFunction();
  TFE_ContextAddFunctionDef(ctx, function_def.data(), function_def.size(),
                            status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  for (bool clear_cache : {true, false, true}) {
    if (clear_cache) {
      TFE_ContextClearCaches(ctx);
    }
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
  }
  TFE_ContextRemoveFunction(ctx, "MatMulFunction", status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TFE_DeleteContext(ctx);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
TEST(CAPI, FunctionDefAndExecute) { FunctionDefAndExecute(false); }
TEST(CAPI, FunctionDefAndExecuteAsync) { FunctionDefAndExecute(true); }

void BM_ExecuteFunction(int iters, int async) {
  tensorflow::testing::StopTiming();
  tensorflow::testing::SetLabel(async ? "ExecuteFunctionAsync"
                                      : "ExecuteFunction");
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetAsync(opts, static_cast<unsigned char>(async));
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
  if (async) {
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteTensorHandle(m);
  TFE_DeleteTensorHandle(retval[0]);
  TFE_ContextRemoveFunction(ctx, "MatMulFunction", status);
  ASSERT_TRUE(TF_GetCode(status) == TF_OK) << TF_Message(status);
  TFE_DeleteContext(ctx);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_ExecuteFunction)->Arg(0)->Arg(1);

TFE_TensorHandle* CreateVariable(TFE_Context* ctx, float value,
                                 TF_Status* status) {
  // Create the variable handle.
  TFE_Op* op = TFE_NewOp(ctx, "VarHandleOp", status);
  if (TF_GetCode(status) != TF_OK) return nullptr;
  TFE_OpSetAttrType(op, "dtype", TF_FLOAT);
  TFE_OpSetAttrShape(op, "shape", {}, 0, status);
  TFE_OpSetAttrString(op, "container", "", 0);
  TFE_OpSetAttrString(op, "shared_name", "", 0);
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
  EXPECT_EQ(0, TFE_TensorHandleNumDims(value_handle, status));
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  float value = 0.0f;
  TF_Tensor* t = TFE_TensorHandleResolve(value_handle, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(sizeof(float), TF_TensorByteSize(t));
  memcpy(&value, TF_TensorData(t), sizeof(float));
  TF_DeleteTensor(t);
  EXPECT_EQ(12.0, value);

  TFE_DeleteTensorHandle(var_handle);
  TFE_DeleteTensorHandle(value_handle);
  TFE_DeleteContext(ctx);
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
    CHECK_EQ(0, TFE_TensorHandleNumDims(h, status));
    CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    h = nullptr;
  }
  tensorflow::testing::StopTiming();
  TFE_DeleteOp(op);

  TFE_DeleteTensorHandle(var_handle);
  TFE_DeleteContext(ctx);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TF_DeleteStatus(status);
}
BENCHMARK(BM_ReadVariable);

TEST(CAPI, StringAttributes) {
  // Test that TFE_OpSetAttrString doesn't hold on to the value after it
  // returns.
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  std::vector<int64_t> dims(4, 1);
  TFE_Op* op = TFE_NewOp(ctx, "AvgPool", status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_Tensor* tensor =
      TF_AllocateTensor(TF_FLOAT, dims.data(), dims.size(), sizeof(float));
  float tensor_data[] = {1};
  memcpy(TF_TensorData(tensor), tensor_data, TF_TensorByteSize(tensor));
  TFE_TensorHandle* tensor_handle = TFE_NewTensorHandle(tensor, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(op, tensor_handle, status);
  TF_DeleteTensor(tensor);
  TFE_DeleteTensorHandle(tensor_handle);

  std::vector<int64_t> values(4, 1);
  TFE_OpSetAttrIntList(op, "ksize", values.data(), values.size());
  TFE_OpSetAttrIntList(op, "strides", values.data(), values.size());

  const int BUFFER_SIZE = 10;
  char buffer[BUFFER_SIZE];
  std::strncpy(buffer, "VALID", BUFFER_SIZE);
  TFE_OpSetAttrString(op, "padding", buffer, std::strlen(buffer));
  // Overwriting value in "buffer", should be fine since TFE_Op
  // shouldn't be holding on to it.
  std::strncpy(buffer, "NHWC", BUFFER_SIZE);
  TFE_OpSetAttrString(op, "data_format", buffer, std::strlen(buffer));

  TFE_OpSetAttrType(op, "T", TF_FLOAT);

  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* retvals[1];
  int num_retvals = 1;
  TFE_Execute(op, &retvals[0], &num_retvals, status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  ASSERT_EQ(1, num_retvals);

  tensor = TFE_TensorHandleResolve(retvals[0], status);
  ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(4, TF_TensorByteSize(tensor));
  TF_DeleteTensor(tensor);
  TFE_DeleteTensorHandle(retvals[0]);

  TFE_DeleteOp(op);

  TFE_DeleteContext(ctx);
  TF_DeleteStatus(status);
}

TEST(CAPI, TestTFE_TensorHandleCopySharingUnderlyingTensorHandle) {
  TFE_TensorHandle* h = TestMatrixTensorHandle();
  EXPECT_EQ(TF_FLOAT, TFE_TensorHandleDataType(h));

  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);

  TFE_TensorHandle* h_shares_tensor =
      TFE_TensorHandleCopySharingTensor(h, status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  TF_Tensor* t = TFE_TensorHandleResolve(h_shares_tensor, status.get());
  ASSERT_EQ(16, TF_TensorByteSize(t));
  float data[4] = {0};
  memcpy(&data[0], TF_TensorData(t), TF_TensorByteSize(t));
  EXPECT_EQ(1.0, data[0]);
  EXPECT_EQ(2.0, data[1]);
  EXPECT_EQ(3.0, data[2]);
  EXPECT_EQ(4.0, data[3]);
  TF_DeleteTensor(t);

  TFE_DeleteTensorHandle(h);
  TFE_DeleteTensorHandle(h_shares_tensor);
}

TEST(CAPI, TestTFE_OpInferSingleInputAttrs) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input = TestMatrixTensorHandle();
  TFE_TensorHandle* axis = TestAxisTensorHandle();
  TFE_Op* minOp = TFE_NewOp(ctx, "Min", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(minOp, input, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(minOp, axis, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  tensorflow::AttrValueMap attr_values;
  minOp->operation.Attrs().FillAttrValueMap(&attr_values);
  tensorflow::AttrValueMap::const_iterator attr_found = attr_values.find("T");
  EXPECT_NE(attr_found, attr_values.cend());
  EXPECT_EQ(attr_found->second.type(), tensorflow::DataType::DT_FLOAT);
  attr_found = attr_values.find("Tidx");
  EXPECT_NE(attr_found, attr_values.cend());
  EXPECT_EQ(attr_found->second.type(), tensorflow::DataType::DT_INT32);

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(minOp, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_DeleteStatus(status);
  TFE_DeleteOp(minOp);
  TFE_DeleteTensorHandle(input);
  TFE_DeleteTensorHandle(axis);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TestTFE_OpInferSingleTypeInputListAttrs) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input1 = TestMatrixTensorHandle();
  TFE_TensorHandle* input2 = TestMatrixTensorHandle();
  TFE_TensorHandle* dim = TestScalarTensorHandle(0);
  TFE_Op* concatOp = TFE_NewOp(ctx, "Concat", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* inputs[] = {input1, input2};
  TFE_OpAddInput(concatOp, dim, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInputList(concatOp, inputs, 2, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  tensorflow::AttrValueMap attr_values;
  concatOp->operation.Attrs().FillAttrValueMap(&attr_values);
  tensorflow::AttrValueMap::const_iterator attr_found = attr_values.find("T");
  EXPECT_NE(attr_found, attr_values.cend());
  EXPECT_EQ(attr_found->second.type(), tensorflow::DataType::DT_FLOAT);
  attr_found = attr_values.find("N");
  EXPECT_NE(attr_found, attr_values.cend());
  EXPECT_EQ(attr_found->second.i(), 2);

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(concatOp, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_DeleteStatus(status);
  TFE_DeleteOp(concatOp);
  TFE_DeleteTensorHandle(input1);
  TFE_DeleteTensorHandle(input2);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteTensorHandle(dim);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TestTFE_OpInferMixedTypeInputListAttrs) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* condition = TestScalarTensorHandle(true);
  TFE_TensorHandle* t1 = TestMatrixTensorHandle();
  TFE_TensorHandle* t2 = TestAxisTensorHandle();
  TFE_Op* assertOp = TFE_NewOp(ctx, "Assert", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_OpAddInput(assertOp, condition, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* data[] = {condition, t1, t2};
  TFE_OpAddInputList(assertOp, data, 3, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  tensorflow::AttrValueMap attr_values;
  assertOp->operation.Attrs().FillAttrValueMap(&attr_values);
  tensorflow::AttrValueMap::const_iterator attr_found = attr_values.find("T");
  EXPECT_NE(attr_found, attr_values.cend());
  EXPECT_EQ(attr_found->second.list().type(0), tensorflow::DataType::DT_BOOL);
  EXPECT_EQ(attr_found->second.list().type(1), tensorflow::DataType::DT_FLOAT);
  EXPECT_EQ(attr_found->second.list().type(2), tensorflow::DataType::DT_INT32);

  TFE_TensorHandle* retvals[1] = {nullptr};
  int num_retvals = 1;
  TFE_Execute(assertOp, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_DeleteStatus(status);
  TFE_DeleteOp(assertOp);
  TFE_DeleteTensorHandle(condition);
  TFE_DeleteTensorHandle(t1);
  TFE_DeleteTensorHandle(t2);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TestTFE_OpAttrsInferenceDisabledWhenNotCallingOpAddInputList) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input1 = TestMatrixTensorHandle();
  TFE_TensorHandle* input2 = TestMatrixTensorHandle();
  TFE_TensorHandle* dim = TestScalarTensorHandle(0);
  TFE_Op* concatOp = TFE_NewOp(ctx, "Concat", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* inputs[] = {input1, input2};
  TFE_OpAddInput(concatOp, dim, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  CHECK(concatOp->inference_ctx);
  TFE_OpAddInput(concatOp, inputs[0], status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_FALSE(concatOp->inference_ctx) << "Inference context is still present";
  TFE_OpAddInput(concatOp, inputs[1], status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  tensorflow::AttrValueMap attr_values;
  concatOp->operation.Attrs().FillAttrValueMap(&attr_values);
  EXPECT_EQ(attr_values.find("T"), attr_values.end());
  EXPECT_EQ(attr_values.find("N"), attr_values.end());

  TF_DeleteStatus(status);
  TFE_DeleteOp(concatOp);
  TFE_DeleteTensorHandle(input1);
  TFE_DeleteTensorHandle(input2);
  TFE_DeleteTensorHandle(dim);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TestTFE_OpGetInputAndOutputLengths) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input1 = TestMatrixTensorHandle();
  TFE_TensorHandle* input2 = TestMatrixTensorHandle();
  TFE_Op* identityOp = TFE_NewOp(ctx, "IdentityN", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Try to retrieve lengths before building the attributes (should fail)
  EXPECT_EQ(-1, TFE_OpGetInputLength(identityOp, "input", status));
  CHECK_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(-1, TFE_OpGetOutputLength(identityOp, "output", status));
  CHECK_NE(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* inputs[] = {input1, input2};
  TFE_OpAddInputList(identityOp, inputs, 2, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Try to retrieve lengths before executing the op (should work)
  EXPECT_EQ(2, TFE_OpGetInputLength(identityOp, "input", status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(2, TFE_OpGetOutputLength(identityOp, "output", status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TFE_TensorHandle* retvals[2] = {nullptr};
  int num_retvals = 2;
  TFE_Execute(identityOp, &retvals[0], &num_retvals, status);
  EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  // Try to retrieve lengths after executing the op (should work)
  EXPECT_EQ(2, TFE_OpGetInputLength(identityOp, "input", status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(2, TFE_OpGetOutputLength(identityOp, "output", status));
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  TF_DeleteStatus(status);
  TFE_DeleteOp(identityOp);
  TFE_DeleteTensorHandle(input1);
  TFE_DeleteTensorHandle(input2);
  TFE_DeleteTensorHandle(retvals[0]);
  TFE_DeleteTensorHandle(retvals[1]);
  TFE_DeleteContext(ctx);
}

TEST(CAPI, TestTFE_OpGetInputAndOutputLengthsFailForUnknownArguments) {
  TF_Status* status = TF_NewStatus();
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  TFE_Context* ctx = TFE_NewContext(opts, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_DeleteContextOptions(opts);

  TFE_TensorHandle* input1 = TestMatrixTensorHandle();
  TFE_TensorHandle* input2 = TestMatrixTensorHandle();
  TFE_Op* identityOp = TFE_NewOp(ctx, "IdentityN", status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
  TFE_TensorHandle* inputs[] = {input1, input2};
  TFE_OpAddInputList(identityOp, inputs, 2, status);
  CHECK_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

  EXPECT_EQ(-1, TFE_OpGetInputLength(identityOp, "cheese", status));
  CHECK_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status)) << TF_Message(status);
  EXPECT_EQ(-1, TFE_OpGetOutputLength(identityOp, "cheese", status));
  CHECK_EQ(TF_INVALID_ARGUMENT, TF_GetCode(status)) << TF_Message(status);

  TF_DeleteStatus(status);
  TFE_DeleteOp(identityOp);
  TFE_DeleteTensorHandle(input1);
  TFE_DeleteTensorHandle(input2);
  TFE_DeleteContext(ctx);
}

}  // namespace

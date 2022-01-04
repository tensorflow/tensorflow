/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/coordination_config.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace {

void StartWorkers(int cluster_size,
                  std::function<void(TFE_Context* ctx, TF_Status* status,
                                     int worker_id, int cluster_size)>
                      fn) {
  tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size, /*num_virtual_gpus=*/2);
  // Enable coordination service for propagating remote device attributess
  auto* config = server_def.mutable_default_session_config()
                     ->mutable_experimental()
                     ->mutable_coordination_config();
  config->set_service_type("standalone");
  config->set_service_leader("/job:worker/replica:0/task:0");

  // The blocking counter makes sure that worker/0 thread (leader that starts
  // the coordination service) does not exit early while other workers are still
  // interacting with the coordination service.
  tensorflow::BlockingCounter counter(cluster_size);
  auto worker_thread_fn = [&](int worker_id) {
    tensorflow::ServerDef server_def_copy = server_def;
    // By default, server_def has task index set to 0.
    server_def_copy.set_task_index(worker_id);
    std::string serialized = server_def_copy.SerializeAsString();

    TF_Status* status = TF_NewStatus();
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    TFE_ContextOptionsSetAsync(opts,
                               static_cast<unsigned char>(/*enable=*/true));
    TFE_ContextOptionsSetDevicePlacementPolicy(opts,
                                               TFE_DEVICE_PLACEMENT_SILENT);

    tensorflow::SessionOptions options;
    options.config = server_def_copy.default_session_config();
    opts->session_options.options = options;
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    fn(ctx, status, worker_id, cluster_size);
    counter.DecrementCount();
    counter.Wait();

    // Since we created an async EagerContext, wait for all pending operations
    // to finish before deleting the context.
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };

  std::vector<std::thread> worker_threads;
  for (int i = 0; i < cluster_size; ++i) {
    worker_threads.emplace_back([i, worker_thread_fn] { worker_thread_fn(i); });
  }
  for (auto i = 0; i < cluster_size; ++i) {
    worker_threads[i].join();
  }
}

TEST(CAPI, MultiClientCollectiveOps) {
  auto fn = [](TFE_Context* ctx, TF_Status* status, int worker_id,
               int cluster_size) {
    TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
    TFE_Op* allreduce = AllReduceOp(ctx, in, cluster_size);
    TFE_TensorHandle* retvals[1];
    int num_retvals = 1;
    TFE_Execute(allreduce, &retvals[0], &num_retvals, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

    TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    float result[4] = {0};
    EXPECT_EQ(sizeof(result), TF_TensorByteSize(t));
    memcpy(&result[0], TF_TensorData(t), TF_TensorByteSize(t));
    TF_DeleteTensor(t);
    EXPECT_EQ(2.0, result[0]);
    EXPECT_EQ(4.0, result[1]);
    EXPECT_EQ(6.0, result[2]);
    EXPECT_EQ(8.0, result[3]);

    TFE_DeleteTensorHandle(in);
    TFE_DeleteTensorHandle(retvals[0]);
    TFE_DeleteOp(allreduce);
  };
  StartWorkers(2, fn);
}

TEST(CAPI, MultiClientRemoteDevices) {
  auto fn = [](TFE_Context* ctx, TF_Status* status, int worker_id,
               int cluster_size) {
    std::vector<tensorflow::DeviceAttributes> device_attrs;
    tensorflow::EagerContext* context =
        tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
    context->ListDevices(&device_attrs);
    std::vector<std::string> device_names;
    for (const auto& device_attr : device_attrs) {
      device_names.push_back(device_attr.name());
    }

    bool has_gpu_devices = false;
    std::string unused_gpu_device_name;
    if (GetDeviceName(ctx, &unused_gpu_device_name, "GPU")) {
      has_gpu_devices = true;
    }

    std::vector<std::string> expected_device_names;
    for (int i = 0; i < cluster_size; ++i) {
      expected_device_names.push_back(tensorflow::strings::StrCat(
          "/job:worker/replica:0/task:", i, "/device:CPU:0"));
      if (has_gpu_devices) {
        expected_device_names.push_back(tensorflow::strings::StrCat(
            "/job:worker/replica:0/task:", i, "/device:GPU:0"));
        expected_device_names.push_back(tensorflow::strings::StrCat(
            "/job:worker/replica:0/task:", i, "/device:GPU:1"));
      }
    }

    EXPECT_THAT(device_names,
                testing::UnorderedElementsAreArray(expected_device_names));
  };
  StartWorkers(3, fn);
}

TEST(CAPI, MultiClientSendRecv) {
  auto fn = [](TFE_Context* ctx, TF_Status* status, int worker_id,
               int cluster_size) {
    // Test with GPUs if present (based on test configuration) and CPUs
    // otherwise.
    auto send_device = "/job:worker/replica:0/task:0/device:CPU:0";
    auto recv_device = "/job:worker/replica:0/task:1/device:CPU:0";
    std::string unused_gpu_device_name;
    if (GetDeviceName(ctx, &unused_gpu_device_name, "GPU")) {
      send_device = "/job:worker/replica:0/task:0/device:GPU:0";
      recv_device = "/job:worker/replica:0/task:1/device:GPU:0";
    }

    std::vector<tensorflow::DeviceAttributes> device_attrs;
    tensorflow::EagerContext* context =
        tensorflow::ContextFromInterface(tensorflow::unwrap(ctx));
    context->ListDevices(&device_attrs);

    tensorflow::uint64 send_device_incarnation = 0;
    for (const auto& device_attr : device_attrs) {
      if (device_attr.name() == send_device) {
        send_device_incarnation = device_attr.incarnation();
        break;
      }
    }

    if (worker_id == 0) {
      TFE_TensorHandle* in = TestMatrixTensorHandle(ctx);
      const std::string& op_name =
          tensorflow::str_util::StrContains(send_device, "GPU") ? "Send"
                                                                : "_HostSend";
      TFE_Op* sendop = SendOp(ctx, in, op_name, send_device, recv_device,
                              send_device_incarnation);
      TFE_TensorHandle* retvals[1];
      int num_retvals = 1;
      TFE_Execute(sendop, &retvals[0], &num_retvals, status);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      TFE_DeleteOp(sendop);
      TFE_DeleteTensorHandle(in);
    } else {
      const std::string& op_name =
          tensorflow::str_util::StrContains(send_device, "GPU") ? "Recv"
                                                                : "_HostRecv";
      TFE_Op* recvop = RecvOp(ctx, op_name, send_device, recv_device,
                              send_device_incarnation);
      TFE_TensorHandle* retvals[1];
      int num_retvals = 1;
      TFE_Execute(recvop, &retvals[0], &num_retvals, status);
      TF_Tensor* t = TFE_TensorHandleResolve(retvals[0], status);
      TF_DeleteTensor(t);
      EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
      TFE_DeleteTensorHandle(retvals[0]);
      TFE_DeleteOp(recvop);
    }
  };
  StartWorkers(2, fn);
}

}  // namespace

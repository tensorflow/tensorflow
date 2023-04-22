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
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace {

TEST(CAPI, MultiClientCollectiveOps) {
  const int cluster_size = 2;
  const tensorflow::ServerDef server_def =
      GetMultiClientServerDef("worker", cluster_size);
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
    TFE_Context* ctx = TFE_NewContext(opts, status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteContextOptions(opts);

    TFE_EnableCollectiveOps(ctx, serialized.data(), serialized.size(), status);
    EXPECT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);

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

    // Since we created an async EagerContext, wait for all pending operations
    // to finish before deleting the context.
    TFE_Executor* executor = TFE_ContextGetExecutorForThread(ctx);
    TFE_ExecutorWaitForAllPendingNodes(executor, status);
    ASSERT_EQ(TF_OK, TF_GetCode(status)) << TF_Message(status);
    TFE_DeleteExecutor(executor);

    TFE_DeleteContext(ctx);
    TF_DeleteStatus(status);
  };
  std::thread thread_worker1([&] { worker_thread_fn(0); });
  std::thread thread_worker2([&] { worker_thread_fn(1); });
  thread_worker1.join();
  thread_worker2.join();
}

}  // namespace

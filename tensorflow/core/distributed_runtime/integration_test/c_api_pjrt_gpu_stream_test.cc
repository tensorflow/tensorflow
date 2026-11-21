/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <string>

#include "tensorflow/c/c_api_experimental.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/eager/tfe_context_internal.h"
#include "tensorflow/c/tf_status.h"
#include "xla/tsl/protobuf/coordination_config.pb.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {
namespace {

struct StatusDeleter {
  void operator()(TF_Status* status) const { TF_DeleteStatus(status); }
};

struct ContextDeleter {
  void operator()(TFE_Context* ctx) const { TFE_DeleteContext(ctx); }
};

using UniqueStatus = std::unique_ptr<TF_Status, StatusDeleter>;
using UniqueContext = std::unique_ptr<TFE_Context, ContextDeleter>;

const stream_executor::Stream* FindFirstGpuStream(const DeviceMgr* device_mgr) {
  if (device_mgr == nullptr) {
    return nullptr;
  }
  for (Device* device : device_mgr->ListDevices()) {
    if (device->device_type() == DEVICE_GPU) {
      const DeviceBase::AcceleratorDeviceInfo* info =
          device->tensorflow_accelerator_device_info();
      if (info != nullptr) {
        return info->stream;
      }
    }
  }
  return nullptr;
}

const stream_executor::Stream* FindFirstGpuStream(TFE_Context* ctx) {
  EagerContext* context = ContextFromInterface(unwrap(ctx));
  return FindFirstGpuStream(context->local_device_mgr());
}

UniqueContext CreateDefaultContext(TF_Status* status) {
  TFE_ContextOptions* opts = TFE_NewContextOptions();
  UniqueContext ctx(TFE_NewContext(opts, status));
  TFE_DeleteContextOptions(opts);
  return ctx;
}

TEST(PjRtGpuStreamTest, EnableCollectiveOpsWithoutCoordinationReusesGpuStream) {
  UniqueStatus status(TF_NewStatus());
  UniqueContext ctx = CreateDefaultContext(status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const stream_executor::Stream* initial_stream = FindFirstGpuStream(ctx.get());
  if (initial_stream == nullptr) {
    GTEST_SKIP() << "No GPU device available.";
  }

  // Multi-worker collective ops without coordination service never replace the
  // process-global PJRT GPU client, so the gRPC server's GPU device must reuse
  // the existing PJRT GPU compute stream rather than allocating a new one.
  ServerDef server_def = GetMultiClientServerDef("worker", /*num_tasks=*/2,
                                                 /*num_virtual_gpus=*/0);
  std::string serialized = server_def.SerializeAsString();
  TFE_EnableCollectiveOps(ctx.get(), serialized.data(), serialized.size(),
                          status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const stream_executor::Stream* post_stream = FindFirstGpuStream(ctx.get());
  ASSERT_NE(post_stream, nullptr);
  EXPECT_EQ(post_stream, initial_stream);
}

TEST(PjRtGpuStreamTest,
     EnableCollectiveOpsSingleNodeWithCoordinationReusesGpuStream) {
  UniqueStatus status(TF_NewStatus());
  UniqueContext ctx = CreateDefaultContext(status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const stream_executor::Stream* initial_stream = FindFirstGpuStream(ctx.get());
  if (initial_stream == nullptr) {
    GTEST_SKIP() << "No GPU device available.";
  }

  // Single-node collective ops skip creating a distributed PJRT GPU client
  // (num_nodes <= 1), so the gRPC server must reuse the existing GPU stream.
  ServerDef server_def = GetMultiClientServerDef("worker", /*num_tasks=*/1,
                                                 /*num_virtual_gpus=*/0);
  auto* coord_config = server_def.mutable_default_session_config()
                           ->mutable_experimental()
                           ->mutable_coordination_config();
  coord_config->set_service_type("standalone");
  coord_config->set_service_leader("/job:worker/replica:0/task:0");

  std::string serialized = server_def.SerializeAsString();
  TFE_EnableCollectiveOps(ctx.get(), serialized.data(), serialized.size(),
                          status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const stream_executor::Stream* post_stream = FindFirstGpuStream(ctx.get());
  ASSERT_NE(post_stream, nullptr);
  EXPECT_EQ(post_stream, initial_stream);
}

TEST(PjRtGpuStreamTest, NewGrpcServerReusesGpuStreamByDefault) {
  UniqueStatus status(TF_NewStatus());
  UniqueContext ctx = CreateDefaultContext(status.get());
  ASSERT_EQ(TF_OK, TF_GetCode(status.get())) << TF_Message(status.get());

  const stream_executor::Stream* initial_stream = FindFirstGpuStream(ctx.get());
  if (initial_stream == nullptr) {
    GTEST_SKIP() << "No GPU device available.";
  }

  ServerDef server_def = GetServerDef("worker", /*num_tasks=*/1);
  std::unique_ptr<ServerInterface> server;
  ASSERT_TRUE(NewServer(server_def, &server).ok());

  const stream_executor::Stream* server_stream =
      FindFirstGpuStream(server->worker_env()->device_mgr);
  ASSERT_NE(server_stream, nullptr);
  EXPECT_EQ(server_stream, initial_stream);
}

}  // namespace
}  // namespace tensorflow

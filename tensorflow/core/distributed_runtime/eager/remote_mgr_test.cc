/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"

#include <memory>

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/remote_tensor_handle.pb.h"

namespace tensorflow {
namespace eager {
namespace {

class TestRemoteMgr : public RemoteMgr {
 public:
  TestRemoteMgr(bool is_master, EagerContext* ctx)
      : RemoteMgr(is_master, ctx) {}

  uint64 OpId() {
    tf_shared_lock l(next_id_mutex_);
    return next_op_id_;
  }
};

class RemoteMgrTest : public ::testing::Test {
 public:
  RemoteMgrTest() {
    std::vector<std::unique_ptr<Device>> devices;
    devices.push_back(
        DeviceFactory::NewDevice("CPU", {}, "/job:localhost/replica:0/task:0"));
    local_device_ = devices.back().get();
    devices.push_back(
        DeviceFactory::NewDevice("CPU", {}, "/job:worker/replica:0/task:0"));
    remote_device_ = devices.back().get();
    auto device_mgr = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    context_id_ = random::New64();
    tensorflow::Rendezvous* rendezvous =
        new tensorflow::IntraProcessRendezvous(device_mgr.get());
    ctx_ = new tensorflow::EagerContext(
        SessionOptions(),
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        tensorflow::ContextMirroringPolicy::MIRRORING_NONE, /*async=*/false,
        /*lazy_copy_function_remote_inputs=*/false, device_mgr.release(), true,
        rendezvous, GetDefaultCustomKernelCreator(), nullptr);
  }

  ~RemoteMgrTest() override { ctx_->Unref(); }

  Device* local_device_;
  Device* remote_device_;
  uint64 context_id_;
  EagerContext* ctx_;
};

TEST_F(RemoteMgrTest, SerializeLocalTensorHandleWithRemoteMirror) {
  RemoteMgr remote_mgr(false, ctx_);
  Tensor t(DT_FLOAT, TensorShape({0}));

  TensorHandle* handle;
  TF_ASSERT_OK(
      TensorHandle::CreateLocalHandle(t, local_device_, ctx_, &handle));
  const uint64 op_id = 2;
  const int output_num = 3;
  auto tensor_handle_data = absl::make_unique<RemoteTensorHandleData>(
      op_id, output_num, t.shape(), /*remote_task=*/"", context_id_, ctx_);
  TF_ASSERT_OK(
      handle->AddRemoteMirror(std::move(tensor_handle_data), remote_device_));
  RemoteTensorHandle remote_handle;
  TF_ASSERT_OK(remote_mgr.SerializeRemoteTensorHandle(
      handle, &remote_handle, remote_device_, remote_device_->name()));
  EXPECT_EQ(op_id, remote_handle.op_id());
  EXPECT_EQ(output_num, remote_handle.output_num());
  EXPECT_EQ(remote_device_->name(), remote_handle.device());
  handle->Unref();
}

TEST_F(RemoteMgrTest, SerializeRemoteTensorHandle) {
  RemoteMgr remote_mgr(false, ctx_);
  Tensor t(DT_FLOAT, TensorShape({0}));

  const uint64 op_id = 3;
  const int output_num = 1;
  TensorHandle* handle;
  TF_ASSERT_OK(TensorHandle::CreateRemoteHandle(
      op_id, output_num, t.shape(), /*remote_task=*/"", context_id_, DT_FLOAT,
      remote_device_,
      /*resource_device=*/nullptr, ctx_, &handle));
  RemoteTensorHandle remote_handle;
  TF_ASSERT_OK(remote_mgr.SerializeRemoteTensorHandle(
      handle, &remote_handle, remote_device_, remote_device_->name()));
  EXPECT_EQ(op_id, remote_handle.op_id());
  EXPECT_EQ(output_num, remote_handle.output_num());
  EXPECT_EQ(remote_device_->name(), remote_handle.device());
  handle->Unref();
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow

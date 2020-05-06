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
  EagerContext* ctx_;
};

TEST_F(RemoteMgrTest, SerializeLocalTensorHandleWithRemoteMirror) {
  RemoteMgr remote_mgr(false, ctx_);
  const TensorShape shape({0});
  Tensor t(DT_FLOAT, shape);

  TensorHandle* handle = TensorHandle::CreateLocalHandle(
      std::move(t), local_device_, local_device_, ctx_);
  const uint64 op_id = 2;
  const int output_num = 3;
  TF_ASSERT_OK(handle->AddUnshapedRemoteMirror(remote_device_, op_id,
                                               output_num, "", ctx_));
  TF_ASSERT_OK(
      handle->SetRemoteShape(shape, remote_device_, ctx_->GetContextViewId()));
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

  const uint64 op_id = 3;
  const int output_num = 1;
  TensorHandle* handle = TensorHandle::CreateLazyRemoteHandle(
      op_id, output_num, DT_FLOAT, remote_device_, ctx_);
  RemoteTensorHandle remote_handle;
  TF_ASSERT_OK(remote_mgr.SerializeRemoteTensorHandle(
      handle, &remote_handle, remote_device_, remote_device_->name()));
  EXPECT_EQ(op_id, remote_handle.op_id());
  EXPECT_EQ(output_num, remote_handle.output_num());
  EXPECT_EQ(remote_device_->name(), remote_handle.device());
  handle->Unref();
}

TEST_F(RemoteMgrTest, InvalidateRemoteMirrorWithClusterUpdate) {
  RemoteMgr remote_mgr(false, ctx_);
  Tensor t(DT_FLOAT, TensorShape({0}));

  TensorHandle* handle = TensorHandle::CreateLocalHandle(
      std::move(t), local_device_, local_device_, ctx_);
  const uint64 op_id = 2;
  const int output_num = 3;
  TF_ASSERT_OK(handle->AddUnshapedRemoteMirror(remote_device_, op_id,
                                               output_num, "", ctx_));
  EXPECT_TRUE(
      handle->HasRemoteMirror(remote_device_, ctx_->GetContextViewId()));

  // When updating cluster, remote mirror should be invalidated.
  ctx_->IncrementContextViewId();
  EXPECT_FALSE(
      handle->HasRemoteMirror(remote_device_, ctx_->GetContextViewId()));
  EXPECT_FALSE(handle
                   ->SetRemoteShape(TensorShape({0}), remote_device_,
                                    ctx_->GetContextViewId())
                   .ok());
  handle->Unref();
}

TEST_F(RemoteMgrTest, SetRemoteShapeWithClusterUpdate) {
  RemoteMgr remote_mgr(false, ctx_);

  const uint64 op_id = 3;
  const int output_num = 1;
  TensorHandle* handle = TensorHandle::CreateUnshapedRemoteHandle(
      op_id, output_num,
      /*remote_task=*/"", DT_FLOAT, remote_device_, ctx_);
  TF_ASSERT_OK(handle->SetRemoteShape(TensorShape({0}), remote_device_,
                                      ctx_->GetContextViewId()));
  handle->Unref();

  // Setting remote shape on primary (non-mirror) remote handle works after
  // cluster being updated
  handle = TensorHandle::CreateUnshapedRemoteHandle(
      op_id, output_num,
      /*remote_task=*/"", DT_FLOAT, remote_device_, ctx_);
  ctx_->IncrementContextViewId();
  TF_ASSERT_OK(handle->SetRemoteShape(TensorShape({0}), remote_device_,
                                      ctx_->GetContextViewId()));
  handle->Unref();
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow

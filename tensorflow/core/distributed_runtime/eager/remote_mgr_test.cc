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
#include <utility>
#include <vector>

#include "absl/time/clock.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/error_payloads.h"
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
    devices.push_back(
        DeviceFactory::NewDevice("CPU", {}, "/job:worker/replica:0/task:1"));
    another_remote_device_ = devices.back().get();
    auto device_mgr = std::make_unique<StaticDeviceMgr>(std::move(devices));
    auto rendezvous = tsl::core::RefCountPtr<tensorflow::Rendezvous>(
        new tensorflow::IntraProcessRendezvous(device_mgr.get()));
    ctx_ = new tensorflow::EagerContext(
        SessionOptions(),
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /*async=*/false, device_mgr.release(), true, std::move(rendezvous),
        nullptr, nullptr, /*run_eager_op_as_function=*/true);
  }

  ~RemoteMgrTest() override { ctx_->Unref(); }

  Device* local_device_;
  Device* remote_device_;
  Device* another_remote_device_;
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
      handle, /*wait_until_ready=*/true, &remote_handle, remote_device_,
      remote_device_->name()));
  EXPECT_EQ(op_id, remote_handle.op_id());
  EXPECT_EQ(output_num, remote_handle.output_num());
  EXPECT_EQ(remote_device_->name(), remote_handle.device());
  handle->Unref();
}

TEST_F(RemoteMgrTest, SerializeByWaitingDeadlockAvoided) {
  RemoteMgr remote_mgr(false, ctx_);

  const uint64 op_id = 1;
  const int output_num = 1;
  // Later `SerializeRemoteTensorHandle` is called on `another_remote_device_`
  // instead of the device used to create the handle (that is, `remote_device_`)
  // to trigger a second call to `RemoteAddress` inside
  // `SerializeRemoteTensorHandle`.
  TensorHandle* handle = TensorHandle::CreateLazyRemoteHandle(
      op_id, output_num, DT_FLOAT, remote_device_, /*is_ready=*/false, ctx_);

  std::unique_ptr<Thread> thread_worker_1;
  thread_worker_1.reset(tsl::Env::Default()->StartThread(
      tensorflow::ThreadOptions(), "thread_worker2",
      [&remote_mgr, &handle, this]() {
        // Grab tensor handle's lock for reading and then block because tensor
        // handle is not ready. But do not grab remote mgr's lock for reading
        // (which was not the case before).
        RemoteTensorHandle remote_handle;
        TF_ASSERT_OK(remote_mgr.SerializeRemoteTensorHandle(
            handle, /*wait_until_ready=*/true, &remote_handle,
            another_remote_device_, another_remote_device_->name()));
      }));

  std::unique_ptr<Thread> thread_worker_2;
  thread_worker_2.reset(tsl::Env::Default()->StartThread(
      tensorflow::ThreadOptions(), "thread_worker3",
      [&remote_mgr, &handle, this]() {
        // This sleep of 5s ensures that `AddOperationOutput` cannot get the
        // remote mgr's lock before `SerializeRemoteTensorHandle` have had a
        // chance to get to blocked state.
        absl::SleepFor(absl::Seconds(5));
        // Grab remote mgr's lock for writing (which would get stuck before) and
        // release it.
        remote_mgr.AddOperationOutput(handle, op_id, output_num);
        // Set the tensor handle to ready (which would not happen before because
        // `AddOperationOutput` is stuck) so that the other thread is now
        // unblocked.
        TF_ASSERT_OK(handle->SetRemoteShape(TensorShape({0}), remote_device_,
                                            ctx_->GetContextViewId()));
      }));

  thread_worker_1.reset();
  thread_worker_2.reset();

  handle->Unref();
}

TEST_F(RemoteMgrTest, SerializeRemoteTensorHandle) {
  RemoteMgr remote_mgr(false, ctx_);

  const uint64 op_id = 3;
  const int output_num = 1;
  TensorHandle* handle = TensorHandle::CreateLazyRemoteHandle(
      op_id, output_num, DT_FLOAT, remote_device_, /*is_ready=*/true, ctx_);
  RemoteTensorHandle remote_handle;
  TF_ASSERT_OK(remote_mgr.SerializeRemoteTensorHandle(
      handle, /*wait_until_ready=*/true, &remote_handle, remote_device_));
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

TEST_F(RemoteMgrTest, ErrorSourcesShouldExist) {
  RemoteMgr remote_mgr(false, ctx_);

  const uint64 op_id = 3;
  const int output_num = 1;
  TensorHandle* handle = TensorHandle::CreateLazyRemoteHandle(
      op_id, output_num, DT_FLOAT, remote_device_, /*is_ready=*/true, ctx_);
  RemoteTensorHandle remote_handle;
  remote_mgr.AddOperationOutput(handle, op_id, output_num);
  TF_ASSERT_OK(remote_mgr.SerializeRemoteTensorHandle(
      handle, /*wait_until_ready=*/true, &remote_handle, remote_device_));
  auto remote_handle_internal = RemoteTensorHandleInternal(remote_handle);
  TF_ASSERT_OK(remote_mgr.DeleteTensorHandle(remote_handle_internal));

  // Now that the tensor has been deleted, we cannot access the remote handle.
  Status s = remote_mgr.DeleteTensorHandle(remote_handle_internal);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.GetPayload(kErrorSource).has_value());

  TensorHandle* out;
  s = remote_mgr.GetTensorHandle(remote_handle_internal, &out);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.GetPayload(kErrorSource).has_value());

  s = remote_mgr.DeserializeRemoteTensorHandle(remote_handle, &out);
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(s.GetPayload(kErrorSource).has_value());
}

}  // namespace
}  // namespace eager
}  // namespace tensorflow

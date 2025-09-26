/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/collective_rma_local.h"

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

#define NUM_DEVS 3
static const int kStepId = 123;

class CollectiveRemoteAccessLocalTest : public ::testing::Test {
 protected:
  const string kTaskName = "/job:localhost/replica:0/task:0";

  CollectiveRemoteAccessLocalTest() {
    work_queue_ = std::make_shared<UnboundedWorkQueue>(Env::Default(), "test");
    ConfigProto cp;
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, kTaskName, &devices));
    device_mgr_ = std::make_unique<StaticDeviceMgr>(std::move(devices));
    drl_ = std::make_unique<DeviceResolverLocal>(device_mgr_.get());
    prl_ = std::make_unique<CollectiveParamResolverLocal>(
        cp, device_mgr_.get(), drl_.get(), /*nccl_communicator*/ nullptr,
        kTaskName);
    rma_ = std::make_unique<CollectiveRemoteAccessLocal>(device_mgr_.get(),
                                                          drl_.get(), kStepId);
    cm_ = std::make_unique<CancellationManager>();
  }

  ~CollectiveRemoteAccessLocalTest() override = default;

  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<DeviceResolverLocal> drl_;
  std::unique_ptr<CollectiveParamResolverLocal> prl_;
  std::unique_ptr<CollectiveRemoteAccessLocal> rma_;
  std::unique_ptr<CancellationManager> cm_;
};

TEST_F(CollectiveRemoteAccessLocalTest, PostRecvCPU0) {
  Device* cpu0 = nullptr;
  AllocatorAttributes attr;
  DeviceLocality dev_locality;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:0", &cpu0));
  Tensor sink_tensor(DT_FLOAT, TensorShape({8}));
  absl::Notification recv_note;
  absl::Status recv_status;
  rma_->RecvFromPeer(kTaskName + "/device:CPU:0", kTaskName, true /*is_local*/,
                     "key_0", cpu0 /*to_device*/, nullptr /*to_device_ctx*/,
                     attr /*to_alloc_attr*/, &sink_tensor, dev_locality,
                     0 /*stream_index*/, cm_.get(),
                     [&recv_note, &recv_status](const absl::Status& s) {
                       recv_status = s;
                       recv_note.Notify();
                     });
  Tensor source_tensor(DT_FLOAT, TensorShape({8}));
  for (int i = 0; i < 8; ++i) {
    source_tensor.flat<float>()(i) = i / 2;
  }
  // Tensors have distinct storage.
  EXPECT_NE(DMAHelper::base(&source_tensor), DMAHelper::base(&sink_tensor));
  absl::Notification send_note;
  absl::Status send_status;
  rma_->PostToPeer(kTaskName + "/device:CPU:0", kTaskName, "key_0",
                   cpu0 /*from_device*/, nullptr /*from_device_ctx*/,
                   attr /*to_alloc_attr*/, &source_tensor, dev_locality,
                   cm_.get(),
                   [&send_note, &send_status](const absl::Status& s) {
                     send_status = s;
                     send_note.Notify();
                   });
  recv_note.WaitForNotification();
  send_note.WaitForNotification();
  TF_EXPECT_OK(recv_status);
  TF_EXPECT_OK(send_status);
  // Sink tensor gets the source tensor values.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(sink_tensor.flat<float>()(i), i / 2);
  }
  // And still has distinct storage.
  EXPECT_NE(DMAHelper::base(&source_tensor), DMAHelper::base(&sink_tensor));
}

TEST_F(CollectiveRemoteAccessLocalTest, PostRecvCPU1_2) {
  Device* cpu2 = nullptr;
  AllocatorAttributes attr;
  DeviceLocality dev_locality;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:2", &cpu2));
  Tensor sink_tensor(DT_FLOAT, TensorShape({8}));
  absl::Notification recv_note;
  absl::Status recv_status;
  rma_->RecvFromPeer(kTaskName + "/device:CPU:1", kTaskName, true /*is_local*/,
                     "key_0", cpu2 /*to_device*/, nullptr /*to_device_ctx*/,
                     attr /*to_alloc_attr*/, &sink_tensor, dev_locality,
                     0 /*stream_index*/, cm_.get(),
                     [&recv_note, &recv_status](const absl::Status& s) {
                       recv_status = s;
                       recv_note.Notify();
                     });
  Tensor source_tensor(DT_FLOAT, TensorShape({8}));
  for (int i = 0; i < 8; ++i) {
    source_tensor.flat<float>()(i) = i / 2;
  }
  // Tensors have distinct storage.
  EXPECT_NE(DMAHelper::base(&source_tensor), DMAHelper::base(&sink_tensor));
  Device* cpu1 = nullptr;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:1", &cpu1));
  absl::Notification send_note;
  absl::Status send_status;
  rma_->PostToPeer(kTaskName + "/device:CPU:2", kTaskName, "key_0",
                   cpu1 /*from_device*/, nullptr /*from_device_ctx*/,
                   attr /*to_alloc_attr*/, &source_tensor, dev_locality,
                   cm_.get(),
                   [&send_note, &send_status](const absl::Status& s) {
                     send_status = s;
                     send_note.Notify();
                   });
  recv_note.WaitForNotification();
  send_note.WaitForNotification();
  TF_EXPECT_OK(recv_status);
  TF_EXPECT_OK(send_status);
  // Sink tensor gets the source tensor values.
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(sink_tensor.flat<float>()(i), i / 2);
  }
  // And still has distinct storage.
  EXPECT_NE(DMAHelper::base(&source_tensor), DMAHelper::base(&sink_tensor));
}

TEST_F(CollectiveRemoteAccessLocalTest, CheckHealth) {
  absl::Status status;
  absl::Notification done;
  rma_->CheckPeerHealth(kTaskName, /*timeout_in_ms=*/0,
                        [&status, &done](const absl::Status& s) {
                          status = s;
                          done.Notify();
                        });
  done.WaitForNotification();
  EXPECT_TRUE(absl::IsInternal(status));
}

TEST_F(CollectiveRemoteAccessLocalTest, RecvThenCancel) {
  Device* cpu0 = nullptr;
  AllocatorAttributes attr;
  DeviceLocality dev_locality;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:0", &cpu0));
  Tensor sink_tensor(DT_FLOAT, TensorShape({8}));
  absl::Notification recv_note;
  absl::Status recv_status;
  rma_->RecvFromPeer(kTaskName + "/device:CPU:0", kTaskName, true /*is_local*/,
                     "key_0", cpu0 /*to_device*/, nullptr /*to_device_ctx*/,
                     attr /*to_alloc_attr*/, &sink_tensor, dev_locality,
                     0 /*stream_index*/, cm_.get(),
                     [&recv_note, &recv_status](const absl::Status& s) {
                       recv_status = s;
                       recv_note.Notify();
                     });
  cm_->StartCancel();
  recv_note.WaitForNotification();
  EXPECT_TRUE(cm_->IsCancelled());
  EXPECT_TRUE(absl::IsCancelled(recv_status));
}

TEST_F(CollectiveRemoteAccessLocalTest, CancelThenRecv) {
  Device* cpu0 = nullptr;
  AllocatorAttributes attr;
  DeviceLocality dev_locality;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:0", &cpu0));
  Tensor sink_tensor(DT_FLOAT, TensorShape({8}));
  absl::Notification recv_note;
  absl::Status recv_status;
  cm_->StartCancel();
  rma_->RecvFromPeer(kTaskName + "/device:CPU:0", kTaskName, true /*is_local*/,
                     "key_0", cpu0 /*to_device*/, nullptr /*to_device_ctx*/,
                     attr /*to_alloc_attr*/, &sink_tensor, dev_locality,
                     0 /*stream_index*/, cm_.get(),
                     [&recv_note, &recv_status](const absl::Status& s) {
                       recv_status = s;
                       recv_note.Notify();
                     });
  recv_note.WaitForNotification();
  EXPECT_TRUE(cm_->IsCancelled());
  EXPECT_TRUE(absl::IsCancelled(recv_status));
}

TEST_F(CollectiveRemoteAccessLocalTest, PostThenCancel) {
  Device* cpu0 = nullptr;
  AllocatorAttributes attr;
  DeviceLocality dev_locality;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:0", &cpu0));
  Tensor source_tensor(DT_FLOAT, TensorShape({8}));
  absl::Notification send_note;
  absl::Status send_status;
  rma_->PostToPeer(kTaskName + "/device:CPU:0", kTaskName, "key_0",
                   cpu0 /*from_device*/, nullptr /*from_device_ctx*/,
                   attr /*to_alloc_attr*/, &source_tensor, dev_locality,
                   cm_.get(),
                   [&send_note, &send_status](const absl::Status& s) {
                     send_status = s;
                     send_note.Notify();
                   });
  cm_->StartCancel();
  send_note.WaitForNotification();
  EXPECT_TRUE(cm_->IsCancelled());
  EXPECT_TRUE(absl::IsCancelled(send_status));
}

TEST_F(CollectiveRemoteAccessLocalTest, CancelThenPost) {
  Device* cpu0 = nullptr;
  AllocatorAttributes attr;
  DeviceLocality dev_locality;
  TF_ASSERT_OK(device_mgr_->LookupDevice(kTaskName + "/device:CPU:0", &cpu0));
  Tensor source_tensor(DT_FLOAT, TensorShape({8}));
  absl::Notification send_note;
  absl::Status send_status;
  cm_->StartCancel();
  rma_->PostToPeer(kTaskName + "/device:CPU:0", kTaskName, "key_0",
                   cpu0 /*from_device*/, nullptr /*from_device_ctx*/,
                   attr /*to_alloc_attr*/, &source_tensor, dev_locality,
                   cm_.get(),
                   [&send_note, &send_status](const absl::Status& s) {
                     send_status = s;
                     send_note.Notify();
                   });
  send_note.WaitForNotification();
  EXPECT_TRUE(cm_->IsCancelled());
  EXPECT_TRUE(absl::IsCancelled(send_status));
}

}  // namespace
}  // namespace tensorflow

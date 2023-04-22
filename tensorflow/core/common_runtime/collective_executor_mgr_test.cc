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
#include "tensorflow/core/common_runtime/collective_executor_mgr.h"

#include "tensorflow/core/common_runtime/collective_param_resolver_local.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/nccl/collective_communicator.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

#define NUM_DEVS 3

TEST(MaybeCreateNcclCommunicatorm, ZeroGpus) {
  ConfigProto cp;
  (*cp.mutable_device_count())["GPU"] = 0;
  EXPECT_EQ(nullptr, MaybeCreateNcclCommunicator(cp));
}

class CollectiveExecutorMgrTest : public ::testing::Test {
 protected:
  CollectiveExecutorMgrTest() {
    ConfigProto cp;
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    string task_name = "/job:localhost/replica:0/task:0";
    device_count->insert({"CPU", NUM_DEVS});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, task_name, &devices));
    device_mgr_ = absl::make_unique<StaticDeviceMgr>(std::move(devices));
    std::unique_ptr<DeviceResolverInterface> drl(
        new DeviceResolverLocal(device_mgr_.get()));
    std::unique_ptr<ParamResolverInterface> prl(
        new CollectiveParamResolverLocal(cp, device_mgr_.get(), drl.get(),
                                         /*nccl_communicator*/ nullptr,
                                         task_name));
    cme_.reset(new CollectiveExecutorMgr(cp, device_mgr_.get(), std::move(drl),
                                         std::move(prl),
                                         MaybeCreateNcclCommunicator(cp)));
  }

  std::unique_ptr<CollectiveExecutorMgr> cme_;
  std::unique_ptr<DeviceMgr> device_mgr_;
};

TEST_F(CollectiveExecutorMgrTest, FindOrCreate) {
  CollectiveExecutor::Handle* h =
      new CollectiveExecutor::Handle(cme_->FindOrCreate(1), true);
  EXPECT_TRUE(h->get());
  CollectiveExecutor::Handle* h2 =
      new CollectiveExecutor::Handle(cme_->FindOrCreate(1), true);
  EXPECT_EQ(h->get(), h2->get());
  CollectiveExecutor* ce = h->get();
  delete h;
  delete h2;
  CollectiveExecutor::Handle h3(cme_->FindOrCreate(1), true);
  EXPECT_EQ(ce, h3.get());
  cme_->Cleanup(1);
}

TEST_F(CollectiveExecutorMgrTest, StepSequenceRelated) {
  EXPECT_EQ(CollectiveExecutor::kInvalidId, cme_->NextStepId(123));
  Notification ss_note;
  Status ss_status;
  cme_->RefreshStepIdSequenceAsync(123,
                                   [&ss_status, &ss_note](const Status& s) {
                                     ss_status = s;
                                     ss_note.Notify();
                                   });
  ss_note.WaitForNotification();
  EXPECT_FALSE(ss_status.ok());
  EXPECT_EQ(ss_status.error_message(),
            "CollectiveExecutorMgr does not implement RefreshStepIdSequence.");
  Notification gs_note;
  Status gs_status;
  GetStepSequenceRequest* req = nullptr;
  GetStepSequenceResponse* resp = nullptr;
  cme_->GetStepSequenceAsync(req, resp,
                             [&gs_status, &gs_note](const Status& s) {
                               gs_status = s;
                               gs_note.Notify();
                             });
  gs_note.WaitForNotification();
  EXPECT_FALSE(gs_status.ok());
  EXPECT_EQ(gs_status.error_message(),
            "CollectiveExecutorMgr does not implement GetStepSequence.");
}

}  // namespace
}  // namespace tensorflow

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/dispatcher_impl.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace testing {
namespace {

using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Test;

class DataServiceDispatcherImplTest : public Test {
 protected:
  void SetUp() override {
    experimental::DispatcherConfig config;
    test_dispatcher_impl_ = std::make_unique<DataServiceDispatcherImpl>(config);
    TF_ASSERT_OK(test_dispatcher_impl_->Start());
  }

  std::unique_ptr<DataServiceDispatcherImpl> test_dispatcher_impl_;
};

SnapshotRequest CreateDummySnapshotRequest() {
  SnapshotRequest request;
  *request.mutable_dataset() = RangeDataset(10);
  request.set_directory(LocalTempFilename());
  *request.mutable_metadata() = CreateDummyDistributedSnapshotMetadata();
  return request;
}

TEST_F(DataServiceDispatcherImplTest, SnapshotFailsIfAlreadyStarted) {
  SnapshotRequest request = CreateDummySnapshotRequest();
  SnapshotResponse response;
  TF_ASSERT_OK(test_dispatcher_impl_->Snapshot(&request, &response));
  EXPECT_THAT(test_dispatcher_impl_->Snapshot(&request, &response),
              StatusIs(error::INVALID_ARGUMENT, HasSubstr("already started")));
}

TEST_F(DataServiceDispatcherImplTest, SnapshotWritesMetadataFile) {
  SnapshotRequest request = CreateDummySnapshotRequest();
  SnapshotResponse response;
  TF_ASSERT_OK(test_dispatcher_impl_->Snapshot(&request, &response));
  TF_ASSERT_OK(Env::Default()->FileExists(
      io::JoinPath(request.directory(), "snapshot.metadata")));
}

TEST_F(DataServiceDispatcherImplTest, SnapshotDirectoryIsInWorkerHeartbeat) {
  SnapshotRequest snapshot_request = CreateDummySnapshotRequest();
  SnapshotResponse snapshot_response;
  TF_ASSERT_OK(
      test_dispatcher_impl_->Snapshot(&snapshot_request, &snapshot_response));
  WorkerHeartbeatRequest worker_heartbeat_request;
  WorkerHeartbeatResponse worker_heartbeat_response;
  TF_ASSERT_OK(test_dispatcher_impl_->WorkerHeartbeat(
      &worker_heartbeat_request, &worker_heartbeat_response));
  ASSERT_EQ(worker_heartbeat_response.snapshots_size(), 1);
  ASSERT_EQ(worker_heartbeat_response.snapshots(0).directory(),
            snapshot_request.directory());
}

}  // namespace
}  // namespace testing
}  // namespace data
}  // namespace tensorflow

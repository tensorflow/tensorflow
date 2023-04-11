/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/snapshot_manager.h"

#include <memory>
#include <string>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/status_to_from_proto.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/tsl/protobuf/status.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

template <class T>
T GetValue(const Tensor& tensor) {
  return tensor.unaligned_flat<T>().data()[0];
}

TEST(SnapshotManagerTest, CreateStreamAssignment) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest request;
  *request.mutable_dataset() = testing::RangeDataset(10);
  request.set_path(snapshot_path);
  *request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SnapshotManager> snapshot_manager,
                          SnapshotManager::Start(request, Env::Default()));
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  ASSERT_EQ(heartbeat_response.snapshot_tasks().size(), 1);
  EXPECT_EQ(heartbeat_response.snapshot_tasks(0).base_path(), snapshot_path);
  EXPECT_EQ(heartbeat_response.snapshot_tasks(0).stream_index(), 0);
  EXPECT_EQ(heartbeat_response.snapshot_tasks(0).num_sources(), 1);
}

TEST(SnapshotManagerTest, GetSnapshotSplit) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest request;
  *request.mutable_dataset() = testing::RangeDataset(10);
  request.set_path(snapshot_path);
  *request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SnapshotManager> snapshot_manager,
                          SnapshotManager::Start(request, Env::Default()));
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));

  const SnapshotTaskDef& task = heartbeat_response.snapshot_tasks(0);
  GetSnapshotSplitRequest get_split_request;
  GetSnapshotSplitResponse get_split_response;
  get_split_request.set_worker_address("localhost");
  get_split_request.set_base_path(task.base_path());
  get_split_request.set_stream_index(task.stream_index());
  get_split_request.set_source_index(0);

  for (int64_t i = 0; i < 10; ++i) {
    TF_ASSERT_OK(snapshot_manager->GetSnapshotSplit(get_split_request,
                                                    get_split_response));
    Tensor tensor;
    ASSERT_TRUE(tensor.FromProto(get_split_response.split()));
    EXPECT_EQ(GetValue<int64_t>(tensor), i);
  }
}

TEST(SnapshotManagerTest, Resume) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest request;
  *request.mutable_dataset() = testing::RangeDataset(10);
  request.set_path(snapshot_path);
  *request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SnapshotManager> snapshot_manager,
                          SnapshotManager::Start(request, Env::Default()));
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), SizeIs(1));

  // Resumes a snapshot manager.
  heartbeat_response.Clear();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> resumed_manager,
      SnapshotManager::Resume(snapshot_path, Env::Default()));
  TF_EXPECT_OK(
      resumed_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), SizeIs(1));
}

TEST(SnapshotManagerTest, SnapshotStreamError) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest snapshot_request;
  *snapshot_request.mutable_dataset() = testing::RangeDataset(10);
  snapshot_request.set_path(snapshot_path);
  *snapshot_request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(snapshot_request, Env::Default()));
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  const SnapshotTaskDef& task = heartbeat_response.snapshot_tasks(0);

  // Reports an error.
  heartbeat_response.Clear();
  SnapshotTaskProgress snapshot_task_progress;
  *snapshot_task_progress.mutable_snapshot_task() = task;
  *snapshot_task_progress.mutable_status() =
      tsl::StatusToProto(errors::NotFound("Not found"));
  (*heartbeat_request.mutable_snapshot_task_progress())[snapshot_path] =
      snapshot_task_progress;
  TF_EXPECT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  // When the snapshot manager is in an error state, it returns an empty
  // response to inform the workers to cancel the ongoing tasks.
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), IsEmpty());

  // Verifies the `ERROR` file is written.
  TF_ASSERT_OK(
      Env::Default()->FileExists(SnapshotErrorFilePath(snapshot_path)));
  StatusProto status_proto;
  TF_ASSERT_OK(ReadTextProto(
      Env::Default(), SnapshotErrorFilePath(snapshot_path), &status_proto));
  EXPECT_THAT(tsl::StatusFromProto(status_proto),
              StatusIs(error::NOT_FOUND, "Not found"));
}

TEST(SnapshotManagerTest, ResumeFromError) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest request;
  *request.mutable_dataset() = testing::RangeDataset(10);
  request.set_path(snapshot_path);
  *request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SnapshotManager> snapshot_manager,
                          SnapshotManager::Start(request, Env::Default()));
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  ASSERT_THAT(heartbeat_response.snapshot_tasks(), SizeIs(1));
  const SnapshotTaskDef& task = heartbeat_response.snapshot_tasks(0);

  // Reports an error.
  heartbeat_response.Clear();
  SnapshotTaskProgress snapshot_task_progress;
  *snapshot_task_progress.mutable_snapshot_task() = task;
  *snapshot_task_progress.mutable_status() =
      tsl::StatusToProto(errors::NotFound("Not found"));
  (*heartbeat_request.mutable_snapshot_task_progress())[snapshot_path] =
      snapshot_task_progress;
  TF_EXPECT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), IsEmpty());

  // The resumed snapshot manager should be in an error state, which returns an
  // empty response to inform the workers to cancel the ongoing tasks.
  heartbeat_response.Clear();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> resumed_manager,
      SnapshotManager::Resume(snapshot_path, Env::Default()));
  TF_EXPECT_OK(
      resumed_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), IsEmpty());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

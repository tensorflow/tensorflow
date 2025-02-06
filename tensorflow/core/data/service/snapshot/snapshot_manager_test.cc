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

#include <cstdint>
#include <memory>
#include <string>

#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/tsl/protobuf/status.pb.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher.pb.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::IsOkAndHolds;
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

  SnapshotAssignmentManager snapshot_assignment_manager(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(request, snapshot_assignment_manager,
                             Env::Default()));
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

  SnapshotAssignmentManager snapshot_assignment_manager(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(request, snapshot_assignment_manager,
                             Env::Default()));
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

TEST(SnapshotManagerTest, HandleStreamCompletion) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest request;
  *request.mutable_dataset() = testing::RangeDataset(10);
  request.set_path(snapshot_path);
  *request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();
  SnapshotAssignmentManager snapshot_assignment_manager(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(request, snapshot_assignment_manager,
                             Env::Default()));

  // Creates two streams.
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost:1");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  heartbeat_request.Clear();
  heartbeat_response.Clear();
  heartbeat_request.set_worker_address("localhost:2");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  ASSERT_EQ(heartbeat_response.snapshot_tasks().size(), 1);
  const SnapshotTaskDef& snapshot_task = heartbeat_response.snapshot_tasks(0);
  EXPECT_EQ(snapshot_task.base_path(), snapshot_path);
  EXPECT_EQ(snapshot_task.stream_index(), 1);
  EXPECT_EQ(snapshot_task.num_sources(), 1);

  // Reports stream completion.
  heartbeat_request.Clear();
  heartbeat_response.Clear();
  heartbeat_request.set_worker_address("localhost:1");
  SnapshotTaskProgress progress;
  *progress.mutable_snapshot_task() = snapshot_task;
  progress.set_completed(true);
  (*heartbeat_request.mutable_snapshot_task_progress())[snapshot_path] =
      progress;
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_TRUE(heartbeat_response.snapshot_tasks().empty());

  // The worker should not receive a stream in the next heartbeat.
  heartbeat_request.Clear();
  heartbeat_response.Clear();
  heartbeat_request.set_worker_address("localhost:1");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_TRUE(heartbeat_response.snapshot_tasks().empty());
}

TEST(SnapshotManagerTest, Resume) {
  std::string snapshot_path = testing::LocalTempFilename();
  SnapshotRequest request;
  *request.mutable_dataset() = testing::RangeDataset(10);
  request.set_path(snapshot_path);
  *request.mutable_metadata() =
      testing::CreateDummyDistributedSnapshotMetadata();

  SnapshotAssignmentManager snapshot_assignment_manager_1(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(request, snapshot_assignment_manager_1,
                             Env::Default()));
  WorkerHeartbeatRequest heartbeat_request;
  WorkerHeartbeatResponse heartbeat_response;
  heartbeat_request.set_worker_address("localhost");
  TF_ASSERT_OK(
      snapshot_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), SizeIs(1));

  // Resumes a snapshot manager.
  heartbeat_response.Clear();
  SnapshotAssignmentManager snapshot_assignment_manager_2(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> resumed_manager,
      SnapshotManager::Resume(snapshot_path, snapshot_assignment_manager_2,
                              Env::Default()));
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

  SnapshotAssignmentManager snapshot_assignment_manager(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(snapshot_request, snapshot_assignment_manager,
                             Env::Default()));
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

  SnapshotAssignmentManager snapshot_assignment_manager_1(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> snapshot_manager,
      SnapshotManager::Start(request, snapshot_assignment_manager_1,
                             Env::Default()));
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
  SnapshotAssignmentManager snapshot_assignment_manager_2(
      /*worker_max_concurrent_snapshots=*/2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SnapshotManager> resumed_manager,
      SnapshotManager::Resume(snapshot_path, snapshot_assignment_manager_2,
                              Env::Default()));
  TF_EXPECT_OK(
      resumed_manager->WorkerHeartbeat(heartbeat_request, heartbeat_response));
  EXPECT_THAT(heartbeat_response.snapshot_tasks(), IsEmpty());
}

TEST(SnapshotAssignmentManagerTest, LoadBalanceSnapshots) {
  SnapshotAssignmentManager snapshot_assignment_manager(
      /*worker_max_concurrent_snapshots=*/2);
  snapshot_assignment_manager.AddSnapshot("snapshot_1");
  snapshot_assignment_manager.AddSnapshot("snapshot_2");
  snapshot_assignment_manager.AddSnapshot("snapshot_3");

  // Worker 1: snapshot 3
  // Worker 2: N/A
  EXPECT_THAT(snapshot_assignment_manager.TryAddAssignment(
                  "snapshot_3", "worker_1", /*stream_index=*/0),
              IsOkAndHolds(true));
  EXPECT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_1"),
              ElementsAre("snapshot_3", _));
  ASSERT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_2"),
              ElementsAre(Not("snapshot_3")));

  // Worker 1: snapshots 2, 3
  // Worker 2: N/A
  EXPECT_THAT(snapshot_assignment_manager.TryAddAssignment(
                  "snapshot_2", "worker_1", /*stream_index=*/0),
              IsOkAndHolds(true));
  ASSERT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_1"),
              UnorderedElementsAre("snapshot_2", "snapshot_3"));
  EXPECT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_2"),
              ElementsAre("snapshot_1"));

  // Worker 1: snapshots 2, 3
  // Worker 2: snapshot 2
  EXPECT_THAT(snapshot_assignment_manager.TryAddAssignment(
                  "snapshot_1", "worker_1", /*stream_index=*/0),
              IsOkAndHolds(false));
  EXPECT_THAT(snapshot_assignment_manager.TryAddAssignment(
                  "snapshot_2", "worker_2", /*stream_index=*/0),
              IsOkAndHolds(true));
  ASSERT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_1"),
              UnorderedElementsAre("snapshot_2", "snapshot_3"));
  EXPECT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_2"),
              ElementsAre("snapshot_2", "snapshot_1"));

  // Worker 1: snapshot 3
  // Worker 2: snapshot 2
  snapshot_assignment_manager.RemoveAssignment("snapshot_2", "worker_1",
                                               /*stream_index=*/0);
  EXPECT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_1"),
              ElementsAre("snapshot_3", "snapshot_1"));
  ASSERT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_2"),
              ElementsAre("snapshot_2", "snapshot_1"));

  // Worker 1: N/A
  // Worker 2: snapshot 2
  snapshot_assignment_manager.RemoveAssignment("snapshot_3", "worker_1",
                                               /*stream_index=*/0);
  ASSERT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_1"),
              ElementsAre("snapshot_1"));
  ASSERT_THAT(snapshot_assignment_manager.LoadBalanceSnapshots("worker_2"),
              ElementsAre("snapshot_2", "snapshot_1"));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

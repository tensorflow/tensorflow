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
#include "tensorflow/core/data/service/dispatcher_client.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/dataset_store.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tsl/platform/path.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::experimental::DistributedSnapshotMetadata;
using ::tensorflow::data::testing::CreateDummyDistributedSnapshotMetadata;
using ::tensorflow::data::testing::EqualsProto;
using ::tensorflow::data::testing::InfiniteDataset;
using ::tensorflow::data::testing::LocalTempFilename;
using ::tensorflow::data::testing::RangeDataset;
using ::tensorflow::testing::StatusIs;
using ::testing::AllOf;
using ::testing::HasSubstr;

constexpr const char kProtocol[] = "grpc";

DataServiceMetadata GetDefaultMetadata() {
  StructuredValue decoded_spec;
  TensorShapeProto::Dim* dim =
      decoded_spec.mutable_tensor_shape_value()->add_dim();
  dim->set_size(1);
  dim->set_name(absl::StrCat("dim"));

  DataServiceMetadata metadata;
  metadata.set_element_spec(decoded_spec.SerializeAsString());
  metadata.set_compression(DataServiceMetadata::COMPRESSION_SNAPPY);
  metadata.set_cardinality(kUnknownCardinality);
  return metadata;
}

class DispatcherClientTest : public ::testing::Test {
 protected:
  absl::Status SetUpTfDataService(int64_t num_workers,
                                  int64_t worker_max_concurrent_snapshots = 0) {
    TestCluster::Config config;
    config.num_workers = num_workers;
    config.work_dir = tsl::io::JoinPath(tsl::testing::TmpDir(), "work_dir");
    config.worker_max_concurrent_snapshots = worker_max_concurrent_snapshots;
    test_cluster_ = std::make_unique<TestCluster>(config);
    TF_RETURN_IF_ERROR(test_cluster_->Initialize());
    dispatcher_client_ = std::make_unique<DataServiceDispatcherClient>(
        test_cluster_->DispatcherAddress(), kProtocol);
    return absl::OkStatus();
  }

  // Creates a dataset and returns the dataset ID.
  absl::StatusOr<std::string> RegisterDataset(
      const DatasetDef& dataset, const DataServiceMetadata& metadata,
      const std::optional<std::string>& requested_dataset_id = std::nullopt) {
    std::string dataset_id;
    TF_RETURN_IF_ERROR(dispatcher_client_->RegisterDataset(
        dataset, metadata, requested_dataset_id, dataset_id));
    return dataset_id;
  }

  // Starts snapshots and returns the directories.
  absl::StatusOr<absl::flat_hash_set<std::string>> StartDummySnapshots(
      int64_t num_snapshots) {
    DistributedSnapshotMetadata metadata =
        CreateDummyDistributedSnapshotMetadata();
    // Create a set of local file paths to which snapshots will be materialized.
    absl::flat_hash_set<std::string> directories;
    for (int64_t i = 0; i < num_snapshots; ++i) {
      directories.insert(LocalTempFilename());
    }
    for (const auto& directory : directories) {
      TF_RETURN_IF_ERROR(
          dispatcher_client_->Snapshot(RangeDataset(10), directory, metadata));
    }
    return directories;
  }

  std::unique_ptr<TestCluster> test_cluster_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_client_;
};

TEST_F(DispatcherClientTest, GetDataServiceMetadata) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  metadata.set_cardinality(10);
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(RangeDataset(10), metadata));

  DataServiceMetadata result;
  TF_ASSERT_OK(dispatcher_client_->GetDataServiceMetadata(dataset_id, result));
  EXPECT_THAT(result, EqualsProto(metadata));
}

TEST_F(DispatcherClientTest, DatasetDoesNotExist) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  EXPECT_THAT(
      dispatcher_client_->GetDataServiceMetadata(
          /*dataset_id=*/"not-found", metadata),
      absl_testing::StatusIs(error::NOT_FOUND,
                             HasSubstr("Dataset id not-found not found")));
}

TEST_F(DispatcherClientTest, SnapshotAlreadyStarted) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string directory = LocalTempFilename();
  TF_ASSERT_OK(
      dispatcher_client_->Snapshot(RangeDataset(10), directory, metadata));
  EXPECT_THAT(
      dispatcher_client_->Snapshot(RangeDataset(10), directory, metadata),
      absl_testing::StatusIs(error::ALREADY_EXISTS,
                             HasSubstr("already started")));
}

TEST_F(DispatcherClientTest, GetDataServiceConfig) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceConfig config;
  TF_ASSERT_OK(dispatcher_client_->GetDataServiceConfig(config));
  EXPECT_EQ(config.deployment_mode(), DEPLOYMENT_MODE_COLOCATED);
}

TEST_F(DispatcherClientTest, SnapshotSkeletonWritten) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  TF_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<std::string> paths,
                          StartDummySnapshots(/*num_snapshots=*/3));
  for (const auto& path : paths) {
    TF_ASSERT_OK(Env::Default()->FileExists(CommittedChunksDirectory(path)));
    TF_ASSERT_OK(Env::Default()->FileExists(StreamsDirectory(path)));
  }
}

TEST_F(DispatcherClientTest, SnapshotMetadataAndDatasetDefWritten) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  TF_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<std::string> paths,
                          StartDummySnapshots(/*num_snapshots=*/3));
  for (const auto& path : paths) {
    TF_ASSERT_OK(
        Env::Default()->FileExists(io::JoinPath(path, "snapshot.metadata")));
    TF_ASSERT_OK(
        Env::Default()->FileExists(io::JoinPath(path, "dataset_def.proto")));
  }
}

TEST_F(DispatcherClientTest, SnapshotsInHeartbeat) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1,
                                  /*worker_max_concurrent_snapshots=*/3));
  TF_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<std::string> paths,
                          StartDummySnapshots(/*num_snapshots=*/3));
  WorkerHeartbeatRequest worker_heartbeat_request;
  worker_heartbeat_request.set_worker_address(test_cluster_->WorkerAddress(0));

  for (int64_t i = 1; i <= 3; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        WorkerHeartbeatResponse worker_heartbeat_response,
        dispatcher_client_->WorkerHeartbeat(worker_heartbeat_request));
    ASSERT_EQ(worker_heartbeat_response.snapshot_tasks_size(), i);
    for (const auto& snapshot_task :
         worker_heartbeat_response.snapshot_tasks()) {
      ASSERT_TRUE(paths.count(snapshot_task.base_path()));
      ASSERT_EQ(snapshot_task.stream_index(), 0);
    }
  }
}

TEST_F(DispatcherClientTest, GetSnapshotSplit) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  TF_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<std::string> paths,
                          StartDummySnapshots(/*num_snapshots=*/3));
  WorkerHeartbeatRequest worker_heartbeat_request;
  worker_heartbeat_request.set_worker_address(test_cluster_->WorkerAddress(0));
  TF_ASSERT_OK_AND_ASSIGN(
      WorkerHeartbeatResponse worker_heartbeat_response,
      dispatcher_client_->WorkerHeartbeat(worker_heartbeat_request));
  for (int64_t i = 0; i < 5; ++i) {
    for (const auto& snapshot_task :
         worker_heartbeat_response.snapshot_tasks()) {
      GetSnapshotSplitRequest get_snapshot_split_request;
      Tensor split;
      int64_t local_split_index = 0;
      bool end_of_splits = false;
      TF_ASSERT_OK(dispatcher_client_->GetSnapshotSplit(
          test_cluster_->WorkerAddress(0), snapshot_task.base_path(),
          snapshot_task.stream_index(),
          /*source_index=*/0, /*repetition_index=*/0, split, local_split_index,
          end_of_splits));
      EXPECT_EQ(local_split_index, i);
      EXPECT_FALSE(end_of_splits);
    }
  }
}

TEST_F(DispatcherClientTest, GetSnapshotSplitMultipleStreams) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/3,
                                  /*worker_max_concurrent_snapshots=*/1));
  TF_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<std::string> paths,
                          StartDummySnapshots(/*num_snapshots=*/3));

  absl::flat_hash_set<std::string> snapshots_in_progress;
  for (int64_t i = 0; i < 3; ++i) {
    WorkerHeartbeatRequest worker_heartbeat_request;
    worker_heartbeat_request.set_worker_address(
        test_cluster_->WorkerAddress(i));
    TF_ASSERT_OK_AND_ASSIGN(
        WorkerHeartbeatResponse worker_heartbeat_response,
        dispatcher_client_->WorkerHeartbeat(worker_heartbeat_request));
    EXPECT_EQ(worker_heartbeat_response.snapshot_tasks().size(), 1);
    for (const auto& snapshot_task :
         worker_heartbeat_response.snapshot_tasks()) {
      snapshots_in_progress.insert(snapshot_task.base_path());
      GetSnapshotSplitRequest get_snapshot_split_request;
      Tensor split;
      int64_t local_split_index = 0;
      bool end_of_splits = false;
      TF_ASSERT_OK(dispatcher_client_->GetSnapshotSplit(
          test_cluster_->WorkerAddress(i), snapshot_task.base_path(),
          snapshot_task.stream_index(),
          /*source_index=*/0, /*repetition_index=*/0, split, local_split_index,
          end_of_splits));
      EXPECT_EQ(local_split_index, 0);
      EXPECT_FALSE(end_of_splits);
    }
  }

  // Each worker writes one snapshot; each snapshot has been assigned a worker.
  EXPECT_EQ(snapshots_in_progress, paths);
}

TEST_F(DispatcherClientTest, RegisterDatasetWithExplicitId) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  metadata.set_cardinality(10);
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string dataset_id1,
      RegisterDataset(RangeDataset(10), metadata,
                      /*requested_dataset_id=*/"dataset_id"));
  EXPECT_EQ(dataset_id1, "dataset_id");

  // Registers a dataset with the same dataset ID.
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string dataset_id2,
      RegisterDataset(RangeDataset(10), metadata,
                      /*requested_dataset_id=*/"dataset_id"));
  EXPECT_EQ(dataset_id1, dataset_id2);
}

TEST_F(DispatcherClientTest, DatasetsDoNotMatch) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  metadata.set_cardinality(10);
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string dataset_id1,
      RegisterDataset(RangeDataset(10), metadata,
                      /*requested_dataset_id=*/"dataset_id"));
  EXPECT_EQ(dataset_id1, "dataset_id");

  // Registers a dataset with the same dataset ID but different metadata.
  metadata.set_cardinality(kInfiniteCardinality);
  EXPECT_THAT(
      RegisterDataset(InfiniteDataset(), metadata,
                      /*requested_dataset_id=*/"dataset_id"),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          HasSubstr(
              "Datasets with the same ID should have the same structure")));
}

TEST_F(DispatcherClientTest, EnableCrossTrainerCache) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  metadata.set_cardinality(kInfiniteCardinality);
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(InfiniteDataset(), metadata));

  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  std::string job_name = "job";
  int64_t job_id;
  TF_ASSERT_OK(dispatcher_client_->GetOrCreateJob(
      dataset_id, processing_mode, job_name,
      /*num_consumers=*/std::nullopt,
      /*use_cross_trainer_cache=*/true, TARGET_WORKERS_AUTO, job_id));
  int64_t iteration_client_id;
  TF_ASSERT_OK(dispatcher_client_->GetOrCreateIteration(
      job_id, /*repetition=*/0, iteration_client_id));

  WorkerHeartbeatRequest worker_heartbeat_request;
  worker_heartbeat_request.set_worker_address(test_cluster_->WorkerAddress(0));
  TF_ASSERT_OK_AND_ASSIGN(
      WorkerHeartbeatResponse worker_heartbeat_response,
      dispatcher_client_->WorkerHeartbeat(worker_heartbeat_request));
  ASSERT_EQ(worker_heartbeat_response.new_tasks_size(), 1);
  EXPECT_TRUE(worker_heartbeat_response.new_tasks(0).use_cross_trainer_cache());
}

TEST_F(DispatcherClientTest, CreateNamedJob) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  metadata.set_cardinality(10);
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(RangeDataset(10), metadata));

  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  std::string job_name = "job";
  int64_t job_id_1 = -1;
  TF_ASSERT_OK(dispatcher_client_->GetOrCreateJob(
      dataset_id, processing_mode, job_name,
      /*num_consumers=*/std::nullopt,
      /*use_cross_trainer_cache=*/true, TARGET_WORKERS_AUTO, job_id_1));

  int64_t job_id_2 = -2;
  // Creating the same job should succeed and receive the same job id.
  TF_ASSERT_OK(dispatcher_client_->GetOrCreateJob(
      dataset_id, processing_mode, job_name,
      /*num_consumers=*/std::nullopt,
      /*use_cross_trainer_cache=*/true, TARGET_WORKERS_AUTO, job_id_2));
  ASSERT_EQ(job_id_1, job_id_2);
}

TEST_F(DispatcherClientTest, NamedJobsDoNotMatch) {
  TF_ASSERT_OK(SetUpTfDataService(/*num_workers=*/1));
  DataServiceMetadata metadata = GetDefaultMetadata();
  metadata.set_cardinality(10);
  TF_ASSERT_OK_AND_ASSIGN(const std::string dataset_id,
                          RegisterDataset(RangeDataset(10), metadata));

  int64_t job_id = 0;
  ProcessingModeDef processing_mode;
  processing_mode.set_sharding_policy(ProcessingModeDef::OFF);
  std::string job_name = "job";
  TF_ASSERT_OK(dispatcher_client_->GetOrCreateJob(
      dataset_id, processing_mode, job_name,
      /*num_consumers=*/std::nullopt,
      /*use_cross_trainer_cache=*/false, TARGET_WORKERS_AUTO, job_id));

  // Creating the same iteration with a different argument should fail.
  processing_mode.set_sharding_policy(ProcessingModeDef::DYNAMIC);
  EXPECT_THAT(
      dispatcher_client_->GetOrCreateJob(dataset_id, processing_mode, job_name,
                                         /*num_consumers=*/std::nullopt,
                                         /*use_cross_trainer_cache=*/true,
                                         TARGET_WORKERS_AUTO, job_id),
      absl_testing::StatusIs(
          error::INVALID_ARGUMENT,
          AllOf(HasSubstr("but found an existing job with different "
                          "parameters: "),
                HasSubstr("Existing processing mode: <"),
                HasSubstr("Existing cross-trainer cache: <disabled>"))));
}

class DispatcherClientTest_DatasetId
    : public DispatcherClientTest,
      public ::testing::WithParamInterface<std::optional<std::string>> {};

TEST_P(DispatcherClientTest_DatasetId, SyncDatasetStoreWithDispatcherState) {
  TestCluster::Config config;
  config.num_workers = 1;
  config.work_dir = tsl::io::JoinPath(tsl::testing::TmpDir(), "work_dir");

  test_cluster_ = std::make_unique<TestCluster>(config);
  TF_ASSERT_OK(test_cluster_->Initialize());
  dispatcher_client_ = std::make_unique<DataServiceDispatcherClient>(
      test_cluster_->DispatcherAddress(), kProtocol);

  DatasetDef dataset_def = RangeDataset(10);
  std::optional<std::string> requested_dataset_id = GetParam();
  std::string dataset_id;
  TF_ASSERT_OK(dispatcher_client_->RegisterDataset(
      dataset_def, GetDefaultMetadata(),
      /*requested_dataset_id=*/std::nullopt, dataset_id));
  EXPECT_EQ(dataset_id, "1000");

  // Writes an inconsistent dataset file. It should be discarded when the user
  // registers a new dataset.
  std::string datasets_dir = tsl::io::JoinPath(config.work_dir, "datasets");
  FileSystemDatasetStore dataset_store(datasets_dir);
  TF_ASSERT_OK(dataset_store.Put("1001", dataset_def));
  if (requested_dataset_id.has_value()) {
    TF_ASSERT_OK(dataset_store.Put(*requested_dataset_id, dataset_def));
  }

  TF_ASSERT_OK(dispatcher_client_->RegisterDataset(
      dataset_def, GetDefaultMetadata(),
      /*requested_dataset_id=*/requested_dataset_id, dataset_id));
  if (requested_dataset_id.has_value()) {
    EXPECT_EQ(dataset_id, *requested_dataset_id);
  } else {
    EXPECT_EQ(dataset_id, "1001");
  }
}

INSTANTIATE_TEST_SUITE_P(DatasetId, DispatcherClientTest_DatasetId,
                         ::testing::Values(std::nullopt, "dataset_id"));

}  // namespace
}  // namespace data
}  // namespace tensorflow

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
#include "tensorflow/core/data/service/snapshot/snapshot_split_provider.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/time/time.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using testing::CreateDummyDistributedSnapshotMetadata;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using testing::LocalTempFilename;
using testing::RangeDataset;
using tsl::testing::IsOkAndHolds;

constexpr const char kProtocol[] = "grpc";

class TestSnapshotCluster {
 public:
  explicit TestSnapshotCluster(int64_t num_workers) {
    TestCluster::Config config;
    config.num_workers = num_workers;
    config.worker_heartbeat_interval_ms = 100;
    test_cluster_ = std::make_unique<TestCluster>(config);
    TF_CHECK_OK(test_cluster_->Initialize());
    dispatcher_client_ = std::make_unique<DataServiceDispatcherClient>(
        test_cluster_->DispatcherAddress(), kProtocol);
  }

  Status RestartWorker(int64_t worker_index) {
    int port = test_cluster_->WorkerBoundPort(0);
    test_cluster_->StopWorker(0);
    return test_cluster_->AddWorker(port);
  }

  std::string DispatcherAddress() const {
    return test_cluster_->DispatcherAddress();
  }

  TestCluster& test_cluster() const { return *test_cluster_; }

  DataServiceDispatcherClient& dispatcher() const {
    return *dispatcher_client_;
  }

 private:
  std::unique_ptr<TestCluster> test_cluster_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_client_;
};

Status WaitUntilFileExists(const std::string& file_path) {
  while (true) {
    Status status = Env::Default()->FileExists(file_path);
    if (!errors::IsNotFound(status)) {
      TF_RETURN_IF_ERROR(status);
    }
    if (status.ok()) {
      return OkStatus();
    }
    Env::Default()->SleepForMicroseconds(
        absl::ToInt64Microseconds(absl::Seconds(1)));
  }
  return OkStatus();
}

Status WaitUntilSnapshotComplete(const std::string& base_path) {
  // TODO(b/258691097): Wait for the DONE in the snapshot base directory.
  TF_RETURN_IF_ERROR(
      WaitUntilFileExists(StreamDoneFilePath(base_path, /*stream_index=*/0)));
  std::string streams_directory = StreamsDirectory(base_path);
  std::vector<std::string> streams;
  TF_RETURN_IF_ERROR(Env::Default()->GetChildren(streams_directory, &streams));
  for (const std::string& stream : streams) {
    std::string done_file =
        tsl::io::JoinPath(streams_directory, stream, "DONE");
    TF_RETURN_IF_ERROR(WaitUntilFileExists(done_file));
  }
  return OkStatus();
}

// Reads the records from a distributed tf.data snapshot written at `base_path`.
template <class T>
StatusOr<std::vector<T>> ReadSnapshot(const std::string& base_path,
                                      const std::string& compression) {
  experimental::DistributedSnapshotMetadata metadata;
  metadata.set_compression(compression);
  SnapshotReaderParams params{base_path, metadata, DataTypeVector{DT_INT64},
                              Env::Default()};
  SnapshotReader reader(params);
  std::vector<T> result;
  while (true) {
    TF_ASSIGN_OR_RETURN(GetNextResult next, reader.GetNext());
    if (next.end_of_sequence) {
      return result;
    }
    result.push_back(next.tensors[0].unaligned_flat<T>().data()[0]);
  }
  return result;
}

Status ClearSnapshot(const std::string& base_path, int64_t stream_index,
                     int64_t source_index, bool clear_split_files) {
  int64_t undeleted_files, undeleted_dirs;
  TF_RETURN_IF_ERROR(Env::Default()->DeleteRecursively(
      CommittedChunksDirectory(base_path), &undeleted_files, &undeleted_dirs));
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(
      CommittedChunksDirectory(base_path)));
  if (Env::Default()
          ->FileExists(CheckpointsDirectory(base_path, stream_index))
          .ok()) {
    TF_RETURN_IF_ERROR(Env::Default()->DeleteRecursively(
        CheckpointsDirectory(base_path, stream_index), &undeleted_files,
        &undeleted_dirs));
  }
  TF_RETURN_IF_ERROR(
      Env::Default()->DeleteFile(StreamDoneFilePath(base_path, stream_index)));
  if (clear_split_files) {
    TF_RETURN_IF_ERROR(Env::Default()->DeleteRecursively(
        SourceDirectory(base_path, stream_index, source_index),
        &undeleted_files, &undeleted_dirs));
    TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(
        SourceDirectory(base_path, stream_index, source_index)));
  }
  return OkStatus();
}

TEST(SnapshotSplitProviderTest, GetSplitFromDispatcher) {
  TestSnapshotCluster data_service(/*num_workers=*/1);
  DatasetDef dataset = RangeDataset(10);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitUntilSnapshotComplete(snapshot_path));
  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotSplitProviderTest, GetSplitFromFiles) {
  // The first pass generates split files.
  TestSnapshotCluster data_service(/*num_workers=*/1);
  DatasetDef dataset = RangeDataset(10);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitUntilSnapshotComplete(snapshot_path));
  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));

  // Clears the snapshot while keeping the split files. When the worker
  // restarts, it will read the split files to get the splits.
  TF_ASSERT_OK(ClearSnapshot(snapshot_path, /*stream_index=*/0,
                             /*source_index=*/0,
                             /*clear_split_files=*/false));
  TF_ASSERT_OK(data_service.RestartWorker(0));
  TF_ASSERT_OK(WaitUntilSnapshotComplete(snapshot_path));
  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST(SnapshotSplitProviderTest, SplitNotFound) {
  // The first pass generates split files.
  TestSnapshotCluster data_service(/*num_workers=*/1);
  DatasetDef dataset = RangeDataset(10);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitUntilSnapshotComplete(snapshot_path));
  EXPECT_THAT(ReadSnapshot<int64_t>(snapshot_path, tsl::io::compression::kNone),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));

  // Clears the snapshot and split files. The dispatcher replies with the last
  // split, but the previous splits are not found in the source directory. In
  // this case, the workers fail.
  TF_ASSERT_OK(ClearSnapshot(snapshot_path, /*stream_index=*/0,
                             /*source_index=*/0,
                             /*clear_split_files=*/true));
  TF_ASSERT_OK(data_service.RestartWorker(0));
  std::string error_file_path = tsl::io::JoinPath(
      StreamDirectory(snapshot_path, /*stream_index=*/0), "ERROR");
  TF_ASSERT_OK(WaitUntilFileExists(error_file_path));
  std::string error_message;
  TF_ASSERT_OK(
      ReadFileToString(Env::Default(), error_file_path, &error_message));
  EXPECT_THAT(error_message,
              HasSubstr("not all splits between [0, 9] are found"));
}

// TODO(b/266126556): Add a test for checkpointing the split provider.

}  // namespace
}  // namespace data
}  // namespace tensorflow

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
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/data/snapshot_utils.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
namespace data {
namespace {

using testing::CreateDummyDistributedSnapshotMetadata;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using testing::LocalTempFilename;
using testing::RangeDataset;
using tsl::testing::IsOkAndHolds;

constexpr const char kProtocol[] = "grpc";

class DistributedSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TestCluster::Config config;
    config.num_workers = 1;
    config.worker_heartbeat_interval_ms = 100;
    test_cluster_ = std::make_unique<TestCluster>(config);
    TF_ASSERT_OK(test_cluster_->Initialize());
    dispatcher_client_ = std::make_unique<DataServiceDispatcherClient>(
        test_cluster_->DispatcherAddress(), kProtocol);
  }

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

template <class T>
StatusOr<std::vector<T>> ReadRecords(const std::string& file_path,
                                     const std::string& compression,
                                     int64_t num_elements) {
  static constexpr int kTFRecordReader = 2;
  DataTypeVector dtypes(num_elements, DT_INT64);
  std::unique_ptr<snapshot_util::Reader> reader;
  TF_RETURN_IF_ERROR(snapshot_util::Reader::Create(Env::Default(), file_path,
                                                   compression, kTFRecordReader,
                                                   dtypes, &reader));

  std::vector<Tensor> tensors;
  TF_RETURN_IF_ERROR(reader->ReadTensors(&tensors));

  std::vector<T> result;
  for (const Tensor& tensor : tensors) {
    result.push_back(tensor.unaligned_flat<T>().data()[0]);
  }
  return result;
}

TEST_F(DistributedSnapshotTest, WriteSnapshot) {
  DatasetDef dataset = RangeDataset(10);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(dispatcher_client_->Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitUntilFileExists(
      StreamDoneFilePath(snapshot_path, /*stream_index=*/0)));
  EXPECT_THAT(ReadRecords<int64_t>(
                  tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path),
                                    "chunk_0_0"),
                  tsl::io::compression::kNone, /*num_elements=*/10),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST_F(DistributedSnapshotTest, WriteMultipleSnapshots) {
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  // Create a set of local file paths to which snapshots will be materialized.
  std::vector<std::string> snapshots = {LocalTempFilename(),
                                        LocalTempFilename()};
  TF_ASSERT_OK(
      dispatcher_client_->Snapshot(RangeDataset(5), snapshots[0], metadata));
  TF_ASSERT_OK(
      dispatcher_client_->Snapshot(RangeDataset(10), snapshots[1], metadata));

  TF_ASSERT_OK(WaitUntilFileExists(
      StreamDoneFilePath(snapshots[0], /*stream_index=*/0)));
  EXPECT_THAT(ReadRecords<int64_t>(
                  tsl::io::JoinPath(CommittedChunksDirectory(snapshots[0]),
                                    "chunk_0_0"),
                  tsl::io::compression::kNone, /*num_elements=*/5),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4)));

  TF_ASSERT_OK(WaitUntilFileExists(
      StreamDoneFilePath(snapshots[1], /*stream_index=*/0)));
  EXPECT_THAT(ReadRecords<int64_t>(
                  tsl::io::JoinPath(CommittedChunksDirectory(snapshots[1]),
                                    "chunk_0_0"),
                  tsl::io::compression::kNone, /*num_elements=*/10),
              IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
}

TEST_F(DistributedSnapshotTest, EmptyDataset) {
  DatasetDef dataset = RangeDataset(0);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(dispatcher_client_->Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitUntilFileExists(
      StreamDoneFilePath(snapshot_path, /*stream_index=*/0)));
  EXPECT_THAT(ReadRecords<int64_t>(
                  tsl::io::JoinPath(CommittedChunksDirectory(snapshot_path),
                                    "chunk_0_0"),
                  tsl::io::compression::kNone, /*num_elements=*/0),
              IsOkAndHolds(IsEmpty()));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

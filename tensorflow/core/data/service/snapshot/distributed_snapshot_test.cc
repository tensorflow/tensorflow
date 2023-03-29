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
#include <tuple>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/snapshot/snapshot_reader.h"
#include "tensorflow/core/data/service/snapshot/test_utils.h"
#include "tensorflow/core/data/service/test_cluster.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/io/compression.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using testing::ChooseFromDatasets;
using testing::CreateDummyDistributedSnapshotMetadata;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using testing::LocalTempFilename;
using testing::RangeDataset;
using ::testing::UnorderedElementsAre;
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

  DataServiceDispatcherClient& dispatcher() const {
    return *dispatcher_client_;
  }

 private:
  std::unique_ptr<TestCluster> test_cluster_;
  std::unique_ptr<DataServiceDispatcherClient> dispatcher_client_;
};

tsl::Status WaitForFileExists(const std::string& file_path) {
  while (true) {
    tsl::Status status = Env::Default()->FileExists(file_path);
    if (!errors::IsNotFound(status)) {
      TF_RETURN_IF_ERROR(status);
    }
    if (status.ok()) {
      return tsl::OkStatus();
    }
    Env::Default()->SleepForMicroseconds(
        absl::ToInt64Microseconds(absl::Seconds(1)));
  }
  return tsl::OkStatus();
}

tsl::Status WaitForSnapshotComplete(const std::string& base_path) {
  return WaitForFileExists(SnapshotDoneFilePath(base_path));
}

class DistributedSnapshotTest : public ::testing::TestWithParam<int64_t> {
 protected:
  int64_t NumWorkers() const { return GetParam(); }
};

TEST_P(DistributedSnapshotTest, WriteSnapshot) {
  TestSnapshotCluster data_service(NumWorkers());
  DatasetDef dataset = RangeDataset(10);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshot_path));
  if (NumWorkers() == 1) {
    EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path,
                                               tsl::io::compression::kNone),
                IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
  } else {  // More than 1 workers: The element order is non-deterministic.
    EXPECT_THAT(
        testing::ReadSnapshot<int64_t>(snapshot_path,
                                       tsl::io::compression::kNone),
        IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
  }
}

TEST_P(DistributedSnapshotTest, WriteMultipleSnapshots) {
  TestSnapshotCluster data_service(NumWorkers());
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::vector<std::string> snapshots = {
      LocalTempFilename(), LocalTempFilename(), LocalTempFilename()};
  TF_ASSERT_OK(data_service.dispatcher().Snapshot(RangeDataset(0), snapshots[0],
                                                  metadata));
  TF_ASSERT_OK(data_service.dispatcher().Snapshot(RangeDataset(10),
                                                  snapshots[1], metadata));
  TF_ASSERT_OK(data_service.dispatcher().Snapshot(RangeDataset(20),
                                                  snapshots[2], metadata));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshots[0]));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshots[1]));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshots[2]));
  EXPECT_THAT(
      testing::ReadSnapshot<int64_t>(snapshots[0], tsl::io::compression::kNone),
      IsOkAndHolds(IsEmpty()));
  EXPECT_THAT(
      testing::ReadSnapshot<int64_t>(snapshots[1], tsl::io::compression::kNone),
      IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)));
  EXPECT_THAT(
      testing::ReadSnapshot<int64_t>(snapshots[2], tsl::io::compression::kNone),
      IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                        12, 13, 14, 15, 16, 17, 18, 19)));
}

TEST_P(DistributedSnapshotTest, ChooseFromDatasets) {
  // Tests snapshotting this dataset:
  // datasets = [tf.data.Dataset.from_tensor_slices(["a", "a", "a", "a", "a"]),
  //             tf.data.Dataset.from_tensor_slices(["b", "b", "b", "b", "b"]),
  //             tf.data.Dataset.from_tensor_slices(["c", "c", "c", "c", "c"])]
  // choice_dataset = tf.data.Dataset.range(3).repeat()
  // dataset = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)
  TestSnapshotCluster data_service(NumWorkers());
  TF_ASSERT_OK_AND_ASSIGN(DatasetDef dataset, ChooseFromDatasets());
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshot_path));
  EXPECT_THAT(
      testing::ReadSnapshot<tstring>(snapshot_path,
                                     tsl::io::compression::kNone),
      IsOkAndHolds(UnorderedElementsAre("a", "b", "c", "a", "b", "c", "a", "b",
                                        "c", "a", "b", "c", "a", "b", "c")));
}

TEST_P(DistributedSnapshotTest, EmptyDataset) {
  TestSnapshotCluster data_service(NumWorkers());
  DatasetDef dataset = RangeDataset(0);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshot_path));
  EXPECT_THAT(testing::ReadSnapshot<int64_t>(snapshot_path,
                                             tsl::io::compression::kNone),
              IsOkAndHolds(IsEmpty()));
}

INSTANTIATE_TEST_SUITE_P(NumWorkers, DistributedSnapshotTest,
                         ::testing::Values(1, 5));

class DistributedSnapshotCompressionTest
    : public ::testing::TestWithParam<std::tuple<int64_t, std::string>> {
 protected:
  int64_t NumWorkers() const { return std::get<0>(GetParam()); }
  std::string Compression() const { return std::get<1>(GetParam()); }
};

TEST_P(DistributedSnapshotCompressionTest, Compression) {
  TestSnapshotCluster data_service(NumWorkers());
  DatasetDef dataset = RangeDataset(20);
  experimental::DistributedSnapshotMetadata metadata =
      CreateDummyDistributedSnapshotMetadata();
  metadata.set_compression(Compression());
  std::string snapshot_path = LocalTempFilename();
  TF_ASSERT_OK(
      data_service.dispatcher().Snapshot(dataset, snapshot_path, metadata));
  TF_ASSERT_OK(WaitForSnapshotComplete(snapshot_path));
  EXPECT_THAT(
      testing::ReadSnapshot<int64_t>(snapshot_path, Compression()),
      IsOkAndHolds(UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                        12, 13, 14, 15, 16, 17, 18, 19)));
}

INSTANTIATE_TEST_SUITE_P(
    NumWorkersAndCompression, DistributedSnapshotCompressionTest,
    ::testing::Combine(::testing::Values(1, 5),
                       ::testing::Values(tsl::io::compression::kGzip,
                                         tsl::io::compression::kSnappy,
                                         tsl::io::compression::kZlib)));

}  // namespace
}  // namespace data
}  // namespace tensorflow

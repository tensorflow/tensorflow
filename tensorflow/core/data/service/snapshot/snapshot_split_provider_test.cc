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

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/io/compression.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/data/serialization_utils.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_client.h"
#include "tensorflow/core/data/service/snapshot/file_utils.h"
#include "tensorflow/core/data/service/snapshot/path_utils.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/protobuf/snapshot.pb.h"
#include "tsl/platform/path.h"

namespace tensorflow {
namespace data {
namespace {

using ::testing::_;
using testing::CreateDummyDistributedSnapshotMetadata;
using ::testing::DoAll;
using ::testing::HasSubstr;
using testing::LocalTempFilename;
using ::testing::Return;
using ::testing::SetArgReferee;
using tsl::testing::StatusIs;

class MockDispatcherClient : public DataServiceDispatcherClient {
 public:
  explicit MockDispatcherClient()
      : DataServiceDispatcherClient(/*address=*/"localhost",
                                    /*protocol=*/"grpc") {}

  MOCK_METHOD(absl::Status, GetSnapshotSplit,
              (const std::string& worker_address, const std::string& base_path,
               int64_t stream_index, int64_t source_index,
               int64_t repetition_index, Tensor& split,
               int64_t& local_split_index, bool& end_of_splits),
              (override));
};

SnapshotTaskDef TestSnapshotTask() {
  SnapshotTaskDef snapshot_task;
  snapshot_task.set_base_path(LocalTempFilename());
  snapshot_task.set_stream_index(0);
  snapshot_task.set_num_sources(1);
  *snapshot_task.mutable_metadata() = CreateDummyDistributedSnapshotMetadata();
  return snapshot_task;
}

absl::Status WriteSplits(const SnapshotTaskDef& snapshot_task,
                         int64_t num_splits) {
  std::string source_dir =
      RepetitionDirectory(snapshot_task.base_path(),
                          snapshot_task.stream_index(), /*source_index=*/0,
                          /*repetition_index=*/0);
  TF_RETURN_IF_ERROR(Env::Default()->RecursivelyCreateDir(source_dir));
  for (int64_t i = 0; i < num_splits; ++i) {
    std::string split_filename = absl::StrCat("split_", i, "_", i);
    std::string split_path = tsl::io::JoinPath(source_dir, split_filename);
    Tensor split(int64_t{i});
    TF_RETURN_IF_ERROR(AtomicallyWriteTFRecords(
        split_path, {split}, tsl::io::compression::kNone, Env::Default()));
  }
  return absl::OkStatus();
}

TEST(SnapshotSplitProviderTest, GetSplitFromDispatcher) {
  const SnapshotTaskDef snapshot_task = TestSnapshotTask();
  Tensor split(int64_t{0});
  auto mock_dispatcher_ptr = std::make_unique<MockDispatcherClient>();
  MockDispatcherClient* mock_dispatcher = mock_dispatcher_ptr.get();
  // The dispatcher sends split 0 to the worker.
  EXPECT_CALL(*mock_dispatcher, GetSnapshotSplit(_, _, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgReferee<5>(split),
                      SetArgReferee<6>(0),      // local_split_index
                      SetArgReferee<7>(false),  // end_of_splits
                      Return(absl::OkStatus())));

  Tensor result;
  bool end_of_splits = false;
  SnapshotSplitProvider split_provider(
      "worker_address", snapshot_task, /*source_index=*/0,
      /*timeout=*/absl::Seconds(10), std::move(mock_dispatcher_ptr),
      Env::Default());
  TF_EXPECT_OK(split_provider.GetNext(&result, &end_of_splits));
  test::ExpectTensorEqual<int64_t>(result, split);
  EXPECT_FALSE(end_of_splits);
}

TEST(SnapshotSplitProviderTest, GetSplitFromFile) {
  const SnapshotTaskDef snapshot_task = TestSnapshotTask();
  Tensor split(int64_t{9});
  auto mock_dispatcher_ptr = std::make_unique<MockDispatcherClient>();
  MockDispatcherClient* mock_dispatcher = mock_dispatcher_ptr.get();
  // The dispatcher sends split 9 to the worker. The worker should get previous
  // splits from the split files.
  EXPECT_CALL(*mock_dispatcher, GetSnapshotSplit(_, _, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgReferee<5>(split),
                      SetArgReferee<6>(9),      // local_split_index
                      SetArgReferee<7>(false),  // end_of_splits
                      Return(absl::OkStatus())));
  TF_ASSERT_OK(WriteSplits(snapshot_task, /*num_splits=*/10));

  SnapshotSplitProvider split_provider(
      "worker_address", snapshot_task, /*source_index=*/0,
      /*timeout=*/absl::Seconds(10), std::move(mock_dispatcher_ptr),
      Env::Default());

  for (int64_t i = 0; i < 10; ++i) {
    Tensor result;
    bool end_of_splits = false;
    TF_EXPECT_OK(split_provider.GetNext(&result, &end_of_splits));
    test::ExpectTensorEqual<int64_t>(result, Tensor(int64_t{i}));
    EXPECT_FALSE(end_of_splits);
  }
}

TEST(SnapshotSplitProviderTest, EndOfSplits) {
  const SnapshotTaskDef snapshot_task = TestSnapshotTask();
  auto mock_dispatcher_ptr = std::make_unique<MockDispatcherClient>();
  MockDispatcherClient* mock_dispatcher = mock_dispatcher_ptr.get();
  // The dispatcher sends `end_of_splits` to the worker.
  EXPECT_CALL(*mock_dispatcher, GetSnapshotSplit(_, _, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgReferee<6>(0),     // local_split_index
                      SetArgReferee<7>(true),  // end_of_splits
                      Return(absl::OkStatus())));

  SnapshotSplitProvider split_provider(
      "worker_address", snapshot_task, /*source_index=*/0,
      /*timeout=*/absl::Seconds(10), std::move(mock_dispatcher_ptr),
      Env::Default());
  Tensor result;
  bool end_of_splits = false;
  TF_EXPECT_OK(split_provider.GetNext(&result, &end_of_splits));
  EXPECT_TRUE(end_of_splits);
}

TEST(SnapshotSplitProviderTest, SplitNotFound) {
  const SnapshotTaskDef snapshot_task = TestSnapshotTask();
  Tensor split(int64_t{10});
  auto mock_dispatcher_ptr = std::make_unique<MockDispatcherClient>();
  MockDispatcherClient* mock_dispatcher = mock_dispatcher_ptr.get();
  // The dispatcher sends split 10, but no splits are written.
  EXPECT_CALL(*mock_dispatcher, GetSnapshotSplit(_, _, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgReferee<5>(split),
                      SetArgReferee<6>(10),     // local_split_index
                      SetArgReferee<7>(false),  // end_of_splits
                      Return(absl::OkStatus())));
  TF_ASSERT_OK(WriteSplits(snapshot_task, /*num_splits=*/0));

  SnapshotSplitProvider split_provider(
      "worker_address", snapshot_task, /*source_index=*/0,
      /*timeout=*/absl::Seconds(10), std::move(mock_dispatcher_ptr),
      Env::Default());
  Tensor result;
  bool end_of_splits = false;
  EXPECT_THAT(split_provider.GetNext(&result, &end_of_splits),
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  HasSubstr("not all splits between [0, 10] are found")));
}

std::string full_name(const std::string& name) {
  return FullName("test", name);
}

TEST(SnapshotSplitProviderTest, SaveRestore) {
  const SnapshotTaskDef snapshot_task = TestSnapshotTask();
  Tensor split(int64_t{9});
  auto mock_dispatcher_ptr = std::make_unique<MockDispatcherClient>();
  MockDispatcherClient* mock_dispatcher = mock_dispatcher_ptr.get();
  EXPECT_CALL(*mock_dispatcher, GetSnapshotSplit(_, _, _, _, _, _, _, _))
      .WillOnce(DoAll(SetArgReferee<5>(split),
                      SetArgReferee<6>(9),      // local_split_index
                      SetArgReferee<7>(false),  // end_of_splits
                      Return(absl::OkStatus())));
  TF_ASSERT_OK(WriteSplits(snapshot_task, /*num_splits=*/10));

  SnapshotSplitProvider split_provider(
      "worker_address", snapshot_task, /*source_index=*/0,
      /*timeout=*/absl::Seconds(10), std::move(mock_dispatcher_ptr),
      Env::Default());

  // Reads splits 0--4 and then saves.
  for (int64_t i = 0; i < 5; ++i) {
    Tensor result;
    bool end_of_splits = false;
    TF_EXPECT_OK(split_provider.GetNext(&result, &end_of_splits));
    test::ExpectTensorEqual<int64_t>(result, Tensor(int64_t{i}));
    EXPECT_FALSE(end_of_splits);
  }

  VariantTensorDataWriter writer;
  TF_ASSERT_OK(split_provider.Save(full_name, &writer));
  std::vector<const VariantTensorData*> variants;
  writer.GetData(&variants);
  VariantTensorDataReader reader(variants);

  // Reads splits 5--9.
  SnapshotSplitProvider restored_split_provider(
      "worker_address", snapshot_task, /*source_index=*/0,
      /*timeout=*/absl::Seconds(10), std::make_unique<MockDispatcherClient>(),
      Env::Default());
  TF_ASSERT_OK(restored_split_provider.Restore(full_name, &reader));
  for (int64_t i = 5; i <= 9; ++i) {
    Tensor result;
    bool end_of_splits = false;
    TF_EXPECT_OK(split_provider.GetNext(&result, &end_of_splits));
    test::ExpectTensorEqual<int64_t>(result, Tensor(int64_t{i}));
    EXPECT_FALSE(end_of_splits);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow

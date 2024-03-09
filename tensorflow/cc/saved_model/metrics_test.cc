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

#include "tensorflow/cc/saved_model/metrics.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "json/json.h"
#include "json/reader.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace metrics {
// The value of the cells for each metric persists across tests.

TEST(MetricsTest, TestSavedModelWrite) {
  EXPECT_EQ(SavedModelWriteApi("foo").value(), 0);
  SavedModelWriteApi("foo").IncrementBy(1);
  EXPECT_EQ(SavedModelWriteApi("foo").value(), 1);

  EXPECT_EQ(SavedModelWriteCount("1").value(), 0);
  SavedModelWriteCount("1").IncrementBy(1);
  EXPECT_EQ(SavedModelWriteCount("1").value(), 1);
}

TEST(MetricsTest, TestSavedModelRead) {
  SavedModelReadApi("bar").IncrementBy(1);
  EXPECT_EQ(SavedModelReadApi("bar").value(), 1);
  SavedModelReadCount("2").IncrementBy(1);
  EXPECT_EQ(SavedModelReadCount("2").value(), 1);

  SavedModelReadApi("baz").IncrementBy(1);
  EXPECT_EQ(SavedModelReadApi("baz").value(), 1);
  SavedModelReadCount("2").IncrementBy(1);
  EXPECT_EQ(SavedModelReadCount("2").value(), 2);
}

TEST(MetricsTest, TestCheckpointRead) {
  EXPECT_EQ(CheckpointReadDuration("foo").value().num(), 0);
  CheckpointReadDuration("foo").Add(100);
  EXPECT_EQ(CheckpointReadDuration("foo").value().num(), 1);
}

TEST(MetricsTest, TestCheckpointWrite) {
  EXPECT_EQ(CheckpointWriteDuration("foo").value().num(), 0);
  CheckpointWriteDuration("foo").Add(100);
  EXPECT_EQ(CheckpointWriteDuration("foo").value().num(), 1);
}

TEST(MetricsTest, TestAsyncCheckpointWrite) {
  EXPECT_EQ(AsyncCheckpointWriteDuration("foo").value().num(), 0);
  AsyncCheckpointWriteDuration("foo").Add(100);
  EXPECT_EQ(AsyncCheckpointWriteDuration("foo").value().num(), 1);
}

TEST(MetricsTest, TestTrainingTimeSaved) {
  EXPECT_EQ(TrainingTimeSaved("foo").value(), 0);
  TrainingTimeSaved("foo").IncrementBy(100);
  EXPECT_EQ(TrainingTimeSaved("foo").value(), 100);
}

TEST(MetricsTest, TestCheckpointSize) {
  EXPECT_EQ(CheckpointSize("foo", 10).value(), 0);
  CheckpointSize("foo", 10).IncrementBy(1);
  EXPECT_EQ(CheckpointSize("foo", 10).value(), 1);
}

TEST(MetricsTest, TestWriteFingerprint) {
  EXPECT_EQ(SavedModelWriteFingerprint().value(), "");
  SavedModelWriteFingerprint().Set("foo");
  EXPECT_EQ(SavedModelWriteFingerprint().value(), "foo");
  SavedModelWriteFingerprint().Set("bar");
  EXPECT_EQ(SavedModelWriteFingerprint().value(), "bar");
}

TEST(MetricsTest, TestWritePath) {
  EXPECT_EQ(SavedModelWritePath().value(), "");
  SavedModelWritePath().Set("foo");
  EXPECT_EQ(SavedModelWritePath().value(), "foo");
  SavedModelWritePath().Set("bar");
  EXPECT_EQ(SavedModelWritePath().value(), "bar");
}

TEST(MetricsTest, TestWritePathAndSingleprint) {
  EXPECT_EQ(SavedModelWritePathAndSingleprint().value(), "");
  SavedModelWritePathAndSingleprint().Set("foo");
  EXPECT_EQ(SavedModelWritePathAndSingleprint().value(), "foo");
  SavedModelWritePathAndSingleprint().Set("bar");
  EXPECT_EQ(SavedModelWritePathAndSingleprint().value(), "bar");

  EXPECT_EQ(
      MakeSavedModelPathAndSingleprint("path", "singleprint").value_or(""),
      "path:singleprint");
}

TEST(MetricsTest, TestInvalidMakePathAndSingleprint) {
  EXPECT_THAT(MakeSavedModelPathAndSingleprint("", "singleprint"),
              testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(MakeSavedModelPathAndSingleprint("path", ""),
              testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MetricsTest, TestReadFingerprint) {
  EXPECT_EQ(SavedModelReadFingerprint().value(), "");
  SavedModelReadFingerprint().Set("foo");
  EXPECT_EQ(SavedModelReadFingerprint().value(), "foo");
  SavedModelReadFingerprint().Set("bar");
  EXPECT_EQ(SavedModelReadFingerprint().value(), "bar");
}

TEST(MetricsTest, TestReadPath) {
  EXPECT_EQ(SavedModelReadPath().value(), "");
  SavedModelReadPath().Set("foo");
  EXPECT_EQ(SavedModelReadPath().value(), "foo");
  SavedModelReadPath().Set("bar");
  EXPECT_EQ(SavedModelReadPath().value(), "bar");
}

TEST(MetricsTest, TestReadPathAndSingleprint) {
  EXPECT_EQ(SavedModelReadPathAndSingleprint().value(), "");
  SavedModelReadPathAndSingleprint().Set("foo");
  EXPECT_EQ(SavedModelReadPathAndSingleprint().value(), "foo");
  SavedModelReadPathAndSingleprint().Set("bar");
  EXPECT_EQ(SavedModelReadPathAndSingleprint().value(), "bar");

  TF_ASSERT_OK_AND_ASSIGN(
      auto path_singleprint,
      ParseSavedModelPathAndSingleprint("path/model:name:singleprint"));
  auto [path, singleprint] = path_singleprint;
  EXPECT_EQ(path, "path/model:name");
  EXPECT_EQ(singleprint, "singleprint");
}

TEST(MetricsTest, TestMakeFingerprintJson) {
  FingerprintDef fingerprint;
  fingerprint.set_saved_model_checksum(1);
  fingerprint.set_graph_def_program_hash(2);
  fingerprint.set_signature_def_hash(3);
  fingerprint.set_saved_object_graph_hash(4);
  fingerprint.set_checkpoint_hash(5);

  std::string serialized_fingerprint_json = MakeFingerprintJson(fingerprint);

  EXPECT_EQ(
      serialized_fingerprint_json,
      "{\n\t\"checkpoint_hash\" : 5,\n\t\"graph_def_program_hash\" : "
      "2,\n\t\"saved_model_checksum\" : 1,\n\t\"saved_object_graph_hash\" : "
      "4,\n\t\"signature_def_hash\" : 3\n}");

  Json::Value fingerprint_json = Json::objectValue;
  Json::Reader reader = Json::Reader();
  reader.parse(serialized_fingerprint_json, fingerprint_json);
  EXPECT_EQ(fingerprint_json["saved_model_checksum"].asUInt64(), 1);
  EXPECT_EQ(fingerprint_json["graph_def_program_hash"].asUInt64(), 2);
  EXPECT_EQ(fingerprint_json["signature_def_hash"].asUInt64(), 3);
  EXPECT_EQ(fingerprint_json["saved_object_graph_hash"].asUInt64(), 4);
  EXPECT_EQ(fingerprint_json["checkpoint_hash"].asUInt64(), 5);
}

TEST(MetricsTest, TestFoundFingerprintOnLoad) {
  EXPECT_EQ(SavedModelFoundFingerprintOnLoad().value(), "");

  SavedModelFoundFingerprintOnLoad().Set(kFingerprintFound);
  EXPECT_EQ(SavedModelFoundFingerprintOnLoad().value(), "FOUND");
  SavedModelFoundFingerprintOnLoad().Set(kFingerprintNotFound);
  EXPECT_EQ(SavedModelFoundFingerprintOnLoad().value(), "NOT_FOUND");
  SavedModelFoundFingerprintOnLoad().Set(kFingerprintError);
  EXPECT_EQ(SavedModelFoundFingerprintOnLoad().value(), "ERROR");
}

TEST(MetricsTest, TestShardingCallbackDuration) {
  EXPECT_EQ(ShardingCallbackDuration().value(), 0);
  ShardingCallbackDuration().IncrementBy(100);
  EXPECT_EQ(ShardingCallbackDuration().value(), 100);
}

TEST(MetricsTest, TestNumCheckpointShardsWritten) {
  EXPECT_EQ(NumCheckpointShardsWritten().value(), 0);
  NumCheckpointShardsWritten().IncrementBy(10);
  EXPECT_EQ(NumCheckpointShardsWritten().value(), 10);
}

TEST(MetricsTest, TestShardingCallbackDescription) {
  EXPECT_EQ(ShardingCallbackDescription().value(), "");
  ShardingCallbackDescription().Set("foo");
  EXPECT_EQ(ShardingCallbackDescription().value(), "foo");
}

}  // namespace metrics
}  // namespace tensorflow

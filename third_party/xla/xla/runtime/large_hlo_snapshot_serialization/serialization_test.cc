/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/runtime/large_hlo_snapshot_serialization/serialization.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using ::tsl::proto_testing::EqualsProto;

HloUnoptimizedSnapshot CreateSnapshot() {
  HloUnoptimizedSnapshot snapshot;
  *snapshot.mutable_hlo_module() = HloModuleProto();
  snapshot.mutable_hlo_module()->set_name("test_hlo_module");
  for (int i = 0; i < 3; ++i) {
    HloInputs* partition_metadata = snapshot.add_partitions();
    for (int j = 0; j < 3; ++j) {
      Shape shape = ShapeUtil::MakeShape(xla::PrimitiveType::F32, {i, i, i});
      Literal literal = LiteralUtil::CreateR1<float>(std::vector<float>(i, i));
      *partition_metadata->add_arguments() = literal.ToProto();
    }
  }
  return snapshot;
}

absl::StatusOr<HloUnoptimizedSnapshot> SerializeAndDeserialize(
    const HloUnoptimizedSnapshot& snapshot) {
  std::string serialized_snapshot;
  tsl::protobuf::io::StringOutputStream output_stream(&serialized_snapshot);
  TF_RETURN_IF_ERROR(SerializeHloUnoptimizedSnapshot(snapshot, &output_stream));

  tsl::protobuf::io::ArrayInputStream input_stream(serialized_snapshot.data(),
                                                   serialized_snapshot.size());
  return DeserializeHloUnoptimizedSnapshot(&input_stream);
}

TEST(LargeHloSnapshotSerializationTest, SerializeAndDeserialize) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();

  TF_ASSERT_OK_AND_ASSIGN(HloUnoptimizedSnapshot deserialized_snapshot,
                          SerializeAndDeserialize(snapshot));
  EXPECT_THAT(deserialized_snapshot, EqualsProto(snapshot));
}

TEST(LargeHloSnapshotSerializationTest, SerializeAndDeserializeEmptyModule) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();
  *snapshot.mutable_hlo_module() = HloModuleProto();

  TF_ASSERT_OK_AND_ASSIGN(HloUnoptimizedSnapshot deserialized_snapshot,
                          SerializeAndDeserialize(snapshot));

  EXPECT_THAT(deserialized_snapshot, EqualsProto(snapshot));
}

TEST(LargeHloSnapshotSerializationTest, SerializeAndDeserializeEmptyPartition) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();
  snapshot.clear_partitions();

  TF_ASSERT_OK_AND_ASSIGN(HloUnoptimizedSnapshot deserialized_snapshot,
                          SerializeAndDeserialize(snapshot));

  EXPECT_THAT(deserialized_snapshot, EqualsProto(snapshot));
}

TEST(LargeHloSnapshotSerializationTest, SerializeAndDeserializeBrokenSnapshot) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();

  std::string serialized_snapshot;
  tsl::protobuf::io::StringOutputStream output_stream(&serialized_snapshot);
  TF_ASSERT_OK(SerializeHloUnoptimizedSnapshot(snapshot, &output_stream));

  serialized_snapshot[0] = '~';

  HloUnoptimizedSnapshot deserialized_snapshot;
  tsl::protobuf::io::ArrayInputStream input_stream(serialized_snapshot.data(),
                                                   serialized_snapshot.size());
  auto status = DeserializeHloUnoptimizedSnapshot(&input_stream).status();
  EXPECT_THAT(status.message(), "Failed to deserialize metadata");
}

TEST(LargeHloSnapshotSerializationTest, SerializeAndDeserializeBrokenLiteral) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();

  std::string serialized_snapshot;
  tsl::protobuf::io::StringOutputStream output_stream(&serialized_snapshot);
  TF_ASSERT_OK(SerializeHloUnoptimizedSnapshot(snapshot, &output_stream));

  serialized_snapshot.resize(serialized_snapshot.size() - 3);
  HloUnoptimizedSnapshot deserialized_snapshot;
  tsl::protobuf::io::ArrayInputStream input_stream(serialized_snapshot.data(),
                                                   serialized_snapshot.size());
  auto status = DeserializeHloUnoptimizedSnapshot(&input_stream).status();
  EXPECT_THAT(status.message(),
              HasSubstr("Failed to deserialize argument with"));
}

TEST(LargeHloSnapshotSerializationTest,
     SerializeAndDeserializeUnsupportedVersion) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();
  snapshot.set_version(1);

  std::string serialized_snapshot;
  tsl::protobuf::io::StringOutputStream output_stream(&serialized_snapshot);
  TF_ASSERT_OK(SerializeHloUnoptimizedSnapshot(snapshot, &output_stream));

  HloUnoptimizedSnapshot deserialized_snapshot;
  tsl::protobuf::io::ArrayInputStream input_stream(serialized_snapshot.data(),
                                                   serialized_snapshot.size());
  auto status = DeserializeHloUnoptimizedSnapshot(&input_stream).status();
  EXPECT_THAT(status.message(), HasSubstr("Unsupported snapshot version"));
}

TEST(LargeHloSnapshotSerializationTest,
     SerializeAndDeserializeDifferentLiteralSizes) {
  HloUnoptimizedSnapshot snapshot = CreateSnapshot();
  *snapshot.mutable_hlo_module() = HloModuleProto();

  int large_size = 1000 * 1000 * 100;
  Literal large_literal =
      LiteralUtil::CreateR1<float>(std::vector<float>(large_size, 3.14f));
  *snapshot.mutable_partitions()->begin()->add_arguments() =
      large_literal.ToProto();

  TF_ASSERT_OK_AND_ASSIGN(HloUnoptimizedSnapshot deserialized_snapshot,
                          SerializeAndDeserialize(snapshot));

  EXPECT_THAT(deserialized_snapshot, EqualsProto(snapshot));
}

}  // namespace
}  // namespace xla

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

#include "xla/util/split_proto/split_proto_reader.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "riegeli/records/record_reader.h"
#include "riegeli/records/record_writer.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_proto.pb.h"

namespace xla {
using absl_testing::IsOkAndHolds;

namespace {

using ::absl_testing::StatusIs;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

std::unique_ptr<riegeli::Reader> CreateReader(absl::string_view data) {
  return std::make_unique<riegeli::StringReader<absl::string_view>>(data);
}

TEST(SplitProtoReaderTest, ReadSplitProtoWithProtoMergeRecord) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(
        result_proto_type: "xla.gpu.GpuExecutableProto"
        records { proto_merge_record {} }
        records { proto_merge_record {} }
        records { proto_merge_record {} }
      )pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "test_text")pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(thunk { thunk_info { thunk_id: 1 } })pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "overridden_test_text")pb"));
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  ASSERT_OK(ReadSplitProto(CreateReader(data), result_proto));

  EXPECT_THAT(result_proto,
              EqualsProto(R"pb(asm_text: "overridden_test_text"
                               thunk { thunk_info { thunk_id: 1 } })pb"));
}

TEST(SplitProtoReaderTest, ReadSplitProtoWithFieldOverrideRecord) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(result_proto_type: "xla.gpu.GpuExecutableProto"
           records { proto_merge_record {} }
           records {
             field_override_record {
               field_path { field_number: 3 }  # asm_text field number
               field_type: TYPE_STRING
             }
           }
      )pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "original_text")pb"));
  record_writer.WriteRecord("overridden_text");
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  ASSERT_OK(ReadSplitProto(CreateReader(data), result_proto));

  EXPECT_THAT(result_proto, EqualsProto(R"pb(
                asm_text: "overridden_text"
              )pb"));
}

TEST(SplitProtoReaderTest, MismatchedProtoName) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(result_proto_type: "AnotherProto"
           records { proto_merge_record {} })pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "test_text")pb"));
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  EXPECT_THAT(ReadSplitProto(CreateReader(data), result_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SplitProtoReaderTest, UnsupportedFieldOverrideType) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(result_proto_type: "xla.gpu.GpuExecutableProto"
           records { proto_merge_record {} }
           records {
             field_override_record {
               field_path { field_number: 7 }  # thunk field number
               field_type: TYPE_UNKNOWN
             }
           })pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(thunk { thunk_info { thunk_id: 1 } })pb"));
  record_writer.WriteRecord("some_data");
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  EXPECT_THAT(ReadSplitProto(CreateReader(data), result_proto),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(SplitProtoReaderTest, InvalidFieldNumberInFieldOverride) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(result_proto_type: "xla.gpu.GpuExecutableProto"
           records {
             field_override_record {
               field_path { field_number: 9999 }  # Invalid field number
               field_type: TYPE_STRING
             }
           })pb"));
  record_writer.WriteRecord("some_data");
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  EXPECT_THAT(ReadSplitProto(CreateReader(data), result_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SplitProtoReaderTest, NonRiegeliFile) {
  gpu::GpuExecutableProto result_proto;

  EXPECT_THAT(ReadSplitProto(CreateReader("invalid_data"), result_proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      ReadSplitProto(CreateReader(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
                                      R"pb(asm_text: "test_text")pb")
                                      .SerializeAsString()),
                     result_proto),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SplitProtoReaderTest, MissingRecordsInFile) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(
        result_proto_type: "xla.gpu.GpuExecutableProto"
        records { proto_merge_record {} })pb"));
  // No ProtoMergeRecord is written, so the reader will expect more records.
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  EXPECT_THAT(ReadSplitProto(CreateReader(data), result_proto),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(SplitProtoReaderTest, UnknownRecordType) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(
        result_proto_type: "xla.gpu.GpuExecutableProto"
        records {
          # No record type is set, which would happen if the reader doesn't
          # recognize a new record type yet.
        }
        records { proto_merge_record {} }
      )pb"));
  record_writer.WriteRecord("some record we don't know about");
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "test_text")pb"));
  record_writer.Close();

  gpu::GpuExecutableProto result_proto;
  ASSERT_OK(ReadSplitProto(CreateReader(data), result_proto));
  EXPECT_THAT(result_proto, EqualsProto(R"pb(asm_text: "test_text")pb"));
}

}  // namespace

TEST(IsSplitProtoTest, ValidSplitProto) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(
        result_proto_type: "xla.gpu.GpuExecutableProto"
        records { proto_merge_record {} }
      )pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "test_text")pb"));
  record_writer.Close();

  riegeli::StringReader reader(data);
  EXPECT_THAT(IsSplitProto(reader), IsOkAndHolds(true));
}

TEST(IsSplitProtoTest, EmptyFile) {
  std::string data;
  riegeli::StringReader reader(data);
  EXPECT_THAT(IsSplitProto(reader), IsOkAndHolds(false));
}

TEST(IsSplitProtoTest, NotRiegeliFormat) {
  std::string data = "This is not a riegeli file";
  riegeli::StringReader reader(data);
  EXPECT_THAT(IsSplitProto(reader), IsOkAndHolds(false));
}

TEST(IsSplitProtoTest, RiegeliButNotSplitProto) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "test_text")pb"));
  record_writer.Close();

  riegeli::StringReader reader(data);
  EXPECT_THAT(IsSplitProto(reader), IsOkAndHolds(false));
}

TEST(IsSplitProtoTest, ReaderPositionResets) {
  std::string data;
  riegeli::RecordWriter record_writer{riegeli::StringWriter(&data)};
  SplitProtoManifest manifest = ParseTextProtoOrDie<SplitProtoManifest>(
      R"pb(
        result_proto_type: "xla.gpu.GpuExecutableProto"
        records { proto_merge_record {} }
        records { proto_merge_record {} }
      )pb");
  record_writer.WriteRecord(manifest);
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(asm_text: "test_text")pb"));
  record_writer.WriteRecord(ParseTextProtoOrDie<gpu::GpuExecutableProto>(
      R"pb(binary: "test_binary")pb"));
  record_writer.Close();

  riegeli::StringReader reader(data);
  EXPECT_THAT(IsSplitProto(reader), IsOkAndHolds(true));

  // Check if the reader position is reset by reading the first record again.
  riegeli::RecordReader<riegeli::Reader&> record_reader(reader);
  SplitProtoManifest manifest_after_check;
  EXPECT_TRUE(record_reader.ReadRecord(manifest_after_check));
  EXPECT_THAT(manifest_after_check, EqualsProto(manifest));
}

}  // namespace xla

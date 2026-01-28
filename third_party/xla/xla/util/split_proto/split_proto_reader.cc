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
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/reflection.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/records/record_position.h"
#include "riegeli/records/record_reader.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util/split_proto/split_proto.pb.h"

namespace xla {

namespace {

template <typename T, typename Src>
absl::Status ReadRecord(riegeli::RecordReader<Src>& record_reader, T& record) {
  if (!record_reader.ReadRecord(record)) {
    return record_reader.ok()
               ? absl::InternalError(
                     "Manifest indicates there are more records, but the "
                     "file has ended")
               : record_reader.status();
  }
  return absl::OkStatus();
}

template <typename Src>
absl::Status HandleProtoMergeRecord(riegeli::RecordReader<Src>& record_reader,
                                    google::protobuf::Message& proto) {
  absl::string_view record_data;
  TF_RETURN_IF_ERROR(ReadRecord(record_reader, record_data));

  if (!proto.MergeFromString(record_data)) {
    return absl::InternalError("Failed to parse proto merge record");
  }
  return absl::OkStatus();
}

absl::Status ReadOverrideFieldRecord(google::protobuf::Message& proto,
                                     std::string record_data,
                                     const Record& record) {
  Record::FieldOverrideRecord override_record = record.field_override_record();

  if (override_record.field_type() !=
      Record::FieldOverrideRecord::TYPE_STRING) {
    return absl::UnimplementedError(absl::StrFormat(
        "Field override record type %d is not supported by the current "
        "implementation",
        override_record.field_type()));
  }
  if (override_record.field_path().empty()) {
    return absl::InvalidArgumentError(
        "Field override record must have at least one field path part");
  }
  if (override_record.field_path_size() > 1 ||
      override_record.field_path(0).index_case() !=
          Record::FieldOverrideRecord::FieldPathPart::INDEX_NOT_SET) {
    return absl::UnimplementedError(
        "Only non-repeated unnested field overrides are supported by the "
        "current implementation");
  }

  const google::protobuf::FieldDescriptor* field_descriptor =
      proto.GetDescriptor()->FindFieldByNumber(
          override_record.field_path(0).field_number());
  if (field_descriptor == nullptr) {
    return absl::InvalidArgumentError("Field not found in proto");
  }
  if (field_descriptor->is_repeated()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Field override record for repeated fields is not supported by the "
        "current implementation"));
  }

  proto.GetReflection()->SetString(&proto, field_descriptor,
                                   std::move(record_data));
  return absl::OkStatus();
}

template <typename Src>
absl::Status HandleFieldOverrideRecord(
    riegeli::RecordReader<Src>& record_reader, google::protobuf::Message& proto,
    const Record& record) {
  std::string record_data;
  TF_RETURN_IF_ERROR(ReadRecord(record_reader, record_data));

  TF_RETURN_IF_ERROR(
      ReadOverrideFieldRecord(proto, std::move(record_data), record));
  return absl::OkStatus();
}

}  // namespace

absl::Status ReadSplitProto(std::unique_ptr<riegeli::Reader> reader,
                            google::protobuf::Message& proto) {
  riegeli::RecordReader record_reader(std::move(reader));

  SplitProtoManifest manifest;
  if (!record_reader.ReadRecord(manifest)) {
    // If the read data isn't a valid riegeli file, RecordReader will return
    // false, but the status will be OK. So manually check for this case, so we
    // don't silently ignore the error.
    return record_reader.status().ok()
               ? absl::InvalidArgumentError(
                     "Failed to read manifest, you are likely reading a non "
                     "split proto file")
               : record_reader.status();
  }
  if (manifest.result_proto_type() != proto.GetTypeName()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Manifest proto name does not match the expected proto name: %s vs %s",
        manifest.result_proto_type(), proto.GetTypeName()));
  }

  for (const Record& record : manifest.records()) {
    switch (record.record_type_case()) {
      case Record::kProtoMergeRecord: {
        TF_RETURN_IF_ERROR(HandleProtoMergeRecord<>(record_reader, proto));
        break;
      }
      case Record::kFieldOverrideRecord: {
        TF_RETURN_IF_ERROR(
            HandleFieldOverrideRecord<>(record_reader, proto, record));
        break;
      }
      default:
        // This is a record type that the binary doesn't know about yet, so we
        // just ignore it by reading the record and ignoring its data.
        // Unfortunately, we can't just skip the record, without reading it.
        absl::string_view record_data;
        record_reader.ReadRecord(record_data);
        LOG(WARNING) << "Ignoring split proto unknown record type";
        break;
    }
  }

  if (!record_reader.Close()) {
    return record_reader.status();
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> IsSplitProto(riegeli::Reader& reader) {
  riegeli::RecordReader<riegeli::Reader&> record_reader(reader);
  if (!record_reader.CheckFileFormat()) {
    return false;
  }
  riegeli::RecordPosition initial_pos = record_reader.pos();

  SplitProtoManifest manifest;
  bool read_ok = record_reader.ReadRecord(manifest);
  bool manifest_was_read = !manifest.result_proto_type().empty();

  // Resets the reader back, leaving it as it was when we started.
  record_reader.Seek(initial_pos);
  if (!record_reader.ok()) {
    return record_reader.status();
  }

  // Make its an actual manifest, and not an empty or other proto.
  return read_ok && manifest_was_read;
}

}  // namespace xla

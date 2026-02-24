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

#include "xla/util/split_proto/split_executable_and_options_writer.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "google/protobuf/util/field_mask_util.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/util/split_proto/split_proto.pb.h"
#include "xla/util/split_proto/split_proto_riegeli_options.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

namespace {

using ::google::protobuf::FieldMask;
using ::google::protobuf::util::FieldMaskUtil;

SplitProtoManifest BuildManifest() {
  SplitProtoManifest manifest;
  *manifest.mutable_result_proto_type() =
      ExecutableAndOptionsProto::descriptor()->full_name();

  // FieldOverrideRecord for serialized_executable
  Record::FieldOverrideRecord* override_serialized_executable =
      manifest.add_records()->mutable_field_override_record();
  override_serialized_executable->mutable_field_path()->Add()->set_field_number(
      1);
  override_serialized_executable->set_field_type(
      Record::FieldOverrideRecord::TYPE_STRING);

  // ProtoMergeRecord for the rest of the fields
  manifest.add_records()->mutable_proto_merge_record();

  return manifest;
}

template <typename T, typename Src>
absl::Status WriteRecord(riegeli::RecordWriter<Src>& record_writer, T& record) {
  if (!record_writer.WriteRecord(record)) {
    return record_writer.status().ok()
               ? absl::InternalError("Failed to write record")
               : record_writer.status();
  }
  return absl::OkStatus();
}

ExecutableAndOptionsProto GetProtoWithoutSerializedExecutable(
    const ExecutableAndOptionsProto& executable_and_options) {
  FieldMask serialized_executable_field;
  serialized_executable_field.add_paths("serialized_executable");

  FieldMask all_fields_but_serialized_executable;
  FieldMaskUtil::Subtract<ExecutableAndOptionsProto>(
      FieldMaskUtil::GetFieldMaskForAllFields<ExecutableAndOptionsProto>(),
      serialized_executable_field, &all_fields_but_serialized_executable);

  ExecutableAndOptionsProto result;
  FieldMaskUtil::MergeMessageTo(executable_and_options,
                                all_fields_but_serialized_executable,
                                FieldMaskUtil::MergeOptions(), &result);
  return result;
}

}  // namespace

absl::Status WriteSplitExecutableAndOptions(
    const ExecutableAndOptionsProto& executable_and_options,
    std::unique_ptr<riegeli::Writer> writer) {
  riegeli::RecordWriter record_writer(std::move(writer),
                                      GetSplitProtoRiegeliOptions());
  SplitProtoManifest manifest = BuildManifest();
  RETURN_IF_ERROR(WriteRecord(record_writer, manifest));

  // Write the serialized_executable field
  RETURN_IF_ERROR(WriteRecord(record_writer,
                              executable_and_options.serialized_executable()));

  // Write the rest of the fields
  ExecutableAndOptionsProto proto_without_serialized_executable =
      GetProtoWithoutSerializedExecutable(executable_and_options);
  RETURN_IF_ERROR(
      WriteRecord(record_writer, proto_without_serialized_executable));

  if (!record_writer.Close()) {
    return record_writer.status();
  }
  return absl::OkStatus();
}

}  // namespace xla

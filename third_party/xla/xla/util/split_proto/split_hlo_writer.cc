/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/util/split_proto/split_hlo_writer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/message_lite.h"
#include "google/protobuf/util/field_mask_util.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/util/split_proto/split_proto.pb.h"
#include "xla/util/split_proto/split_proto_riegeli_options.h"
#include "xla/util/split_proto/split_proto_write_record.h"

namespace xla {
namespace {

using ::google::protobuf::FieldMask;
using ::google::protobuf::util::FieldMaskUtil;

SplitProtoManifest BuildManifest(int32_t num_of_computations) {
  SplitProtoManifest manifest;
  *manifest.mutable_result_proto_type() = HloProto::descriptor()->full_name();

  // ProtoMergeRecord for all fields but the HLO computations.
  manifest.add_records()->mutable_proto_merge_record();

  for (int32_t i = 0; i < num_of_computations; ++i) {
    manifest.add_records()->mutable_proto_merge_record();
  }

  return manifest;
}

HloProto GetProtoWithoutComputations(const HloProto& hlo_proto) {
  FieldMask hlo_computations_field;
  hlo_computations_field.add_paths("hlo_module.computations");

  FieldMask all_fields_but_hlo_computations;
  FieldMaskUtil::Subtract<HloProto>(
      FieldMaskUtil::GetFieldMaskForAllFields<HloProto>(),
      hlo_computations_field, &all_fields_but_hlo_computations);

  HloProto result;
  FieldMaskUtil::MergeMessageTo(hlo_proto, all_fields_but_hlo_computations,
                                FieldMaskUtil::MergeOptions(), &result);
  return result;
}

}  // namespace

absl::Status WriteSplitHloProto(
    const HloProto& hlo_proto, std::unique_ptr<riegeli::Writer> writer) {
  riegeli::RecordWriter record_writer(std::move(writer),
                                      GetSplitProtoRiegeliOptions());
  SplitProtoManifest manifest =
      BuildManifest(hlo_proto.hlo_module().computations_size());
  RETURN_IF_ERROR(WriteRecord(record_writer, manifest));

  // Write the rest of the fields
  RETURN_IF_ERROR(WriteRecord(record_writer,
                              GetProtoWithoutComputations(hlo_proto)));

  // Write the HLO computations.
  for (const HloComputationProto& computation :
       hlo_proto.hlo_module().computations()) {
    HloProto sub_proto;
    *sub_proto.mutable_hlo_module()->add_computations() = computation;
    RETURN_IF_ERROR(WriteRecord(record_writer, sub_proto));
  }

  if (!record_writer.Close()) {
    return record_writer.status();
  }
  return absl::OkStatus();
}

}  // namespace xla

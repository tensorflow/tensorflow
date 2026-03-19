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

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/literal.h"
#include "xla/runtime/large_hlo_snapshot_serialization/coded_stream_iterators.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"

namespace xla {

constexpr int kMaxSupportedSnapshotVersion = 0;
constexpr int kMaxSupportedLiteralVersion = 0;

absl::Status SerializeHloUnoptimizedSnapshot(
    const HloUnoptimizedSnapshot& snapshot,
    tsl::protobuf::io::ZeroCopyOutputStream* zero_copy_output_stream) {
  // Prepare metadata
  HloUnoptimizedSnapshot metadata_proto;
  *metadata_proto.mutable_hlo_module() = snapshot.hlo_module();
  metadata_proto.set_version(snapshot.version());

  for (const auto& partition : snapshot.partitions()) {
    HloInputs* partition_metadata = metadata_proto.add_partitions();
    for (const auto& argument : partition.arguments()) {
      TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(argument.shape()));
      TF_ASSIGN_OR_RETURN(int64_t serialized_size,
                          ShapeUtil::SerializedSize(shape));
      partition_metadata->add_arguments_descriptors()->set_argument_size_bytes(
          serialized_size);
    }
  }

  // Serialize metadata
  tsl::protobuf::io::CodedOutputStream output_stream(zero_copy_output_stream);
  CodedStreamOutputIterator output_it(&output_stream);
  tsl::protobuf::util::SerializeDelimitedToCodedStream(metadata_proto,
                                                       &output_stream);

  // Serialize literals
  for (const auto& hlo_input : snapshot.partitions()) {
    for (const auto& literal_proto : hlo_input.arguments()) {
      TF_ASSIGN_OR_RETURN(Literal literal,
                          xla::Literal::CreateFromProto(literal_proto));
      TF_RETURN_IF_ERROR(literal.Serialize(output_it));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<HloUnoptimizedSnapshot> DeserializeHloUnoptimizedSnapshot(
    tsl::protobuf::io::ZeroCopyInputStream* zero_copy_input_stream) {
  HloUnoptimizedSnapshot metadata;
  {
    tsl::protobuf::io::CodedInputStream input_stream(zero_copy_input_stream);

    // Deserialize metadata
    if (!tsl::protobuf::util::ParseDelimitedFromCodedStream(
            &metadata, &input_stream,
            /*clean_eof=*/nullptr)) {
      return absl::InternalError("Failed to deserialize metadata");
    }
  }

  HloUnoptimizedSnapshot snapshot_with_args;
  if (metadata.version() > kMaxSupportedSnapshotVersion) {
    return absl::InternalError(
        absl::StrCat("Unsupported snapshot version: ", metadata.version()));
  }

  *snapshot_with_args.mutable_hlo_module() =
      std::move(*metadata.mutable_hlo_module());

  // Deserialize literals
  CodedStreamInputIterator input_it_end;
  for (const auto& partition : metadata.partitions()) {
    HloInputs* partition_metadata = snapshot_with_args.add_partitions();
    for (const auto& descriptor : partition.arguments_descriptors()) {
      tsl::protobuf::io::CodedInputStream input_stream(zero_copy_input_stream);

      if (descriptor.version() > kMaxSupportedLiteralVersion) {
        return absl::InternalError(absl::StrCat(
            "Unsupported argument descriptor version: ", descriptor.version()));
      }

      const uint64_t argument_size = descriptor.argument_size_bytes();
      CodedStreamInputIterator input_it_begin(&input_stream, argument_size);
      auto literal_or_status =
          Literal::Deserialize(input_it_begin, input_it_end);
      if (!literal_or_status.ok()) {
        return absl::InternalError(absl::StrCat(
            "Failed to deserialize argument with size ", argument_size, ": ",
            literal_or_status.status().message()));
      }
      *partition_metadata->add_arguments() =
          literal_or_status.value().ToProto();
    }
  }
  tsl::protobuf::io::CodedInputStream input_stream(zero_copy_input_stream);
  if (input_stream.BytesUntilTotalBytesLimit() > 0) {
    return absl::InternalError("Unexpected extra data in the stream");
  }

  return snapshot_with_args;
}

}  // namespace xla

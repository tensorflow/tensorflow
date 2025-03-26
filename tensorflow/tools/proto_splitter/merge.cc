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

#include "tensorflow/tools/proto_splitter/merge.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/base/object.h"  // from @riegeli
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::tools::proto_splitter {

using ::tensorflow::proto_splitter::ChunkedField;
using ::tensorflow::proto_splitter::ChunkedMessage;
using ::tensorflow::proto_splitter::ChunkInfo;
using ::tensorflow::proto_splitter::ChunkMetadata;
using ::tensorflow::proto_splitter::FieldIndex;
using tools::proto_splitter::GetChunkMetadata;
using tools::proto_splitter::GetRiegeliReader;
using tools::proto_splitter::OnlyContainsPb;
using tsl::protobuf::FieldDescriptor;
using tsl::protobuf::Message;
using tsl::protobuf::Reflection;

absl::Status Merger::Merge(const std::vector<std::unique_ptr<Message>>& chunks,
                           const ChunkedMessage& chunked_message,
                           Message* merged_message) {
  riegeli::RecordReader<riegeli::FdReader<>> null_reader{riegeli::kClosed};

  if (chunked_message.has_chunk_index()) {
    // Chunks referenced by fields should be merged into the parent chunk.
    merged_message->MergeFrom(*chunks[chunked_message.chunk_index()].get());
  }

  // Use each chunked_field within the chunked_message to merge its
  // corresponding chunk into merged_message.
  for (const auto& chunked_field : chunked_message.chunked_fields()) {
    absl::Status s = ProcessField(chunked_field, merged_message, {}, chunks,
                                  null_reader, MergerOp::MERGE);
    if (!s.ok()) return s;
  }

  return absl::OkStatus();
}

absl::Status Merger::Read(std::string prefix, Message* merged_message) {
  uint64_t start_time = Env::Default()->NowMicros();

  TF_ASSIGN_OR_RETURN(bool only_contains_pb, OnlyContainsPb(prefix));
  if (only_contains_pb) {
    return ReadPb(absl::StrCat(prefix, ".pb"), merged_message);
  }

  // Create riegeli reader for file.cpb
  TF_ASSIGN_OR_RETURN(auto reader,
                      GetRiegeliReader(absl::StrCat(prefix, ".cpb")));

  auto read_metadata = GetChunkMetadata(reader);
  if (!read_metadata.ok()) {
    reader.Close();
    return absl::FailedPreconditionError(
        absl::StrCat("Couldn't read ChunkMetadata from chunked proto.\n",
                     read_metadata.status().ToString()));
  }
  ChunkMetadata chunk_metadata = read_metadata.value();

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  // Read the remaining chunks.
  absl::Status s =
      ReadFields(chunk_metadata.message(), reader, chunks_info, merged_message);

  reader.Close();

  uint64_t end_time = Env::Default()->NowMicros();
  LOG(INFO) << "Finished reading and merging chunked proto, took "
            << HumanReadableDuration(end_time - start_time) << ".";
  return s;
}

absl::Status Merger::ReadPartial(absl::string_view prefix,
                                 const ChunkMetadata& chunk_metadata,
                                 Message* merged_message) {
  uint64_t start_time = Env::Default()->NowMicros();

  TF_ASSIGN_OR_RETURN(bool only_contains_pb, OnlyContainsPb(prefix));
  if (only_contains_pb) {
    return absl::FailedPreconditionError(
        absl::StrCat("Attempting to read part of a chunked proto .cpb file, "
                     "but only found a regular proto: ",
                     prefix, ".pb"));
  }

  // Create riegeli reader for file.cpb
  TF_ASSIGN_OR_RETURN(auto reader,
                      GetRiegeliReader(absl::StrCat(prefix, ".cpb")));

  std::vector<ChunkInfo> chunks_info = std::vector<ChunkInfo>(
      chunk_metadata.chunks().begin(), chunk_metadata.chunks().end());

  // Read the remaining chunks.
  absl::Status s =
      ReadFields(chunk_metadata.message(), reader, chunks_info, merged_message);

  reader.Close();

  uint64_t end_time = Env::Default()->NowMicros();
  LOG(INFO) << "Finished reading and merging chunked proto, took "
            << HumanReadableDuration(end_time - start_time) << ".";
  return s;
}

absl::Status Merger::ReadPb(const std::string& pb_file,
                            Message* merged_message) {
  uint64_t start_time = Env::Default()->NowMicros();
  TF_ASSIGN_OR_RETURN(bool file_exists,
                      internal::FileExists(Env::Default(), pb_file));
  if (!file_exists)
    return absl::NotFoundError(absl::StrCat("File not found: ", pb_file));
  LOG(INFO) << "Reading binary proto from " << pb_file;
  auto ret = ReadBinaryProto(Env::Default(), pb_file, merged_message);
  uint64_t end_time = Env::Default()->NowMicros();
  LOG(INFO) << "Finished reading binary proto, took "
            << HumanReadableDuration(end_time - start_time) << ".";
  return ret;
}

absl::Status Merger::ReadFields(
    const ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<ChunkInfo>& chunks_info,
    tsl::protobuf::Message* merged_message) {
  if (chunked_message.has_chunk_index()) {
    // Chunks referenced by fields should be merged into the parent chunk.
    TF_ASSIGN_OR_RETURN(
        std::string chunk,
        ReadChunk(reader, chunks_info[chunked_message.chunk_index()]));
    if (!merged_message->MergeFromString(chunk)) {
      return absl::FailedPreconditionError(
          "Couldn't merge chunk into message.");
    }
  }

  // Sort the chunked_fields by depth and index.
  // For example, this ensures that GraphDef.library is merged before
  //   GraphDef.library.function[0], which will be merged before
  //   GraphDef.library.function[1].
  std::vector<ChunkedField> chunked_fields(
      chunked_message.chunked_fields().begin(),
      chunked_message.chunked_fields().end());
  absl::Status sort_status = absl::OkStatus();
  std::sort(
      chunked_fields.begin(), chunked_fields.end(),
      [&sort_status](ChunkedField cf1, ChunkedField cf2) {
        int tag_depth =
            std::min(cf1.field_tag().size(), cf2.field_tag().size());
        for (int depth = 0; depth < tag_depth; ++depth) {
          FieldIndex tag1 = cf1.field_tag()[depth];
          FieldIndex tag2 = cf2.field_tag()[depth];
          if (tag1.has_field() && tag2.has_field()) {
            uint32_t field1 = tag1.field();
            uint32_t field2 = tag2.field();
            if (field1 != field2) return field1 < field2;
          } else if (tag1.has_index() && tag2.has_index()) {
            uint64_t index1 = tag1.index();
            uint64_t index2 = tag2.index();
            if (index1 != index2) return index1 < index2;
          } else if (tag1.has_map_key() && tag2.has_map_key()) {
            return false;
          } else {
            sort_status = absl::FailedPreconditionError("Field tag mismatch");
            return false;
          }
        }
        if (cf1.field_tag().size() == cf2.field_tag().size()) {
          // If the fields are the same, merge the earlier chunks first.
          return cf1.message().chunk_index() < cf2.message().chunk_index();
        }
        return cf1.field_tag().size() < cf2.field_tag().size();
      });
  if (!sort_status.ok()) return sort_status;

  // Use each chunked_field within the chunked_message to merge its
  // corresponding chunk into merged_message.
  for (const auto& chunked_field : chunked_fields) {
    absl::Status s = ProcessField(chunked_field, merged_message, chunks_info,
                                  {}, reader, MergerOp::READ);
    if (!s.ok()) return s;
  }
  return absl::OkStatus();
}

absl::Status Merger::ProcessField(
    const ChunkedField& chunked_field, Message* merged_message,
    const std::vector<ChunkInfo>& chunks_info,
    const std::vector<std::unique_ptr<Message>>& chunks,
    riegeli::RecordReader<riegeli::FdReader<>>& reader, MergerOp op) {
  std::string chunk;
  switch (op) {
    case MergerOp::READ: {
      TF_ASSIGN_OR_RETURN(
          chunk, ReadChunk(reader,
                           chunks_info[chunked_field.message().chunk_index()]));
      break;
    }
    case MergerOp::MERGE: {
      chunk =
          chunks[chunked_field.message().chunk_index()]->SerializeAsString();
      break;
    }
  }

  if (chunked_field.field_tag().empty()) {
    // Chunk is not a field within the parent, but instead a portion of the
    // parent itself. Needs to be concatenated.
    merged_message->MergeFromString(chunk);
    return absl::OkStatus();
  }

  uint64_t field_index;
  Message* curr_message = merged_message;
  // Use field tags to navigate the merged_message, constructing necessary
  // fields along the way.
  TF_ASSIGN_OR_RETURN(const std::vector<Field> fields,
                      GetFieldTypes(chunked_field.field_tag()));
  const FieldDescriptor* field_desc = nullptr;
  for (const auto& field : fields) {
    merged_message = curr_message;
    field_desc = merged_message->GetDescriptor()->FindFieldByNumber(
        std::get<int>(field.first));
    auto res = GetMutableField(merged_message, field);
    if (!res.ok()) {
      if (!absl::IsNotFound(res.status())) return res.status();
      // Add missing field.
      if (field_desc->is_map()) {
        TF_RETURN_IF_ERROR(
            AddMapEntry(curr_message, field_desc, field.second.value()));
        res = GetMutableField(curr_message, field);
      } else {
        curr_message->GetReflection()->AddMessage(curr_message, field_desc);
        res = GetMutableField(curr_message, field);
      }
    }

    auto [parent, mutable_field, mutable_field_index] = res.value();
    if (mutable_field->is_repeated() && mutable_field_index != -1) {
      field_index = mutable_field_index;
      // Update merged_message to repeated element.
      curr_message = parent->GetReflection()->MutableRepeatedMessage(
          parent, mutable_field, std::max(0, mutable_field_index));
      if (mutable_field->is_map()) {
        // messages of map type have the value at field #2
        field_desc = mutable_field->message_type()->FindFieldByNumber(2);
        merged_message = curr_message;
        curr_message = curr_message->GetReflection()->MutableMessage(
            curr_message, field_desc);
      }
    } else if (mutable_field->type() == FieldDescriptor::Type::TYPE_MESSAGE) {
      // Update merged_message.
      curr_message =
          parent->GetReflection()->MutableMessage(parent, mutable_field);
    }
  }
  // merged_message now points to the Message whose field (described by
  // field_desc) will be added/set.

  const Reflection* reflection = merged_message->GetReflection();
  if (field_desc->is_repeated()) {
    // field may contain multiple elements
    auto message_callback = [&reflection, &merged_message, &field_index, &op,
                             &chunks, &chunked_field, &reader, &chunks_info,
                             &field_desc]() -> absl::Status {
      for (int _ = reflection->FieldSize(*merged_message, field_desc);
           _ <= field_index; _++) {
        reflection->AddMessage(merged_message, field_desc);
      }
      switch (op) {
        case MergerOp::MERGE:
          TF_RETURN_IF_ERROR(
              Merge(chunks, chunked_field.message(),
                    reflection->MutableRepeatedMessage(
                        merged_message, field_desc, field_index)));
          break;
        case MergerOp::READ:
          TF_RETURN_IF_ERROR(
              ReadFields(chunked_field.message(), reader, chunks_info,
                         reflection->MutableRepeatedMessage(
                             merged_message, field_desc, field_index)));
          break;
        default:
          return absl::InternalError("Encountered unknown MergerOp.");
      }
      return absl::OkStatus();
    };
    TF_RETURN_IF_ERROR(SetRepeatedFieldElement(
        merged_message, field_desc, field_index, chunk, message_callback));
  } else {
    // regular field
    auto message_callback = [&reflection, &merged_message, &op, &chunks,
                             &chunked_field, &reader, &chunks_info,
                             &field_desc]() -> absl::Status {
      switch (op) {
        case MergerOp::MERGE:
          TF_RETURN_IF_ERROR(
              Merge(chunks, chunked_field.message(),
                    reflection->MutableMessage(merged_message, field_desc)));
          break;
        case MergerOp::READ:
          TF_RETURN_IF_ERROR(ReadFields(
              chunked_field.message(), reader, chunks_info,
              reflection->MutableMessage(merged_message, field_desc)));
          break;
        default:
          return absl::InternalError("Encountered unknown MergerOp.");
      }
      return absl::OkStatus();
    };
    TF_RETURN_IF_ERROR(
        SetFieldElement(merged_message, field_desc, chunk, message_callback));
  }

  return absl::OkStatus();
}

}  // namespace tensorflow::tools::proto_splitter

#include "tensorflow/tools/proto_splitter/cc/composable_splitter_base.h"

#include <unistd.h>

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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/cord_writer.h"  // from @riegeli
#include "riegeli/bytes/fd_writer.h"  // from @riegeli
#include "riegeli/bytes/string_writer.h"  // from @riegeli
#include "riegeli/records/record_writer.h"  // from @riegeli
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/tools/proto_splitter/cc/max_size.h"
#include "tensorflow/tools/proto_splitter/cc/split.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#define IS_OSS true

namespace tensorflow {
namespace tools::proto_splitter {

using ::tensorflow::proto_splitter::ChunkMetadata;

VersionDef ComposableSplitterBase::Version() {
  VersionDef version;
  version.set_splitter_version(1);
  version.set_join_version(0);
  return version;
}

void ComposableSplitterBase::SetInitialSize(size_t size) { size_ = size; }

size_t ComposableSplitterBase::GetInitialSize() {
  if (size_ == 0) {
    size_ = message_->ByteSizeLong();
  }
  return size_;
}

absl::StatusOr<ChunkedProto> ComposableSplitterBase::Split() {
  if (parent_splitter_ != nullptr) {
    return absl::UnimplementedError(
        "The `Split` function behavior for children ComposableSplitter has not "
        "been defined. Please call `parent_splitter.Split()` instead.");
  }
  if (!built_) {
    LOG(INFO) << "Splitting message '" << message_->GetDescriptor()->full_name()
              << "' (size " << HumanReadableBytes(GetInitialSize()) << ") "
              << "into chunks of size " << HumanReadableBytes(GetMaxSize());
    uint64_t start_time = Env::Default()->NowMicros();
    TF_RETURN_IF_ERROR(BuildChunks());
    TF_RETURN_IF_ERROR(FixChunks());
    uint64_t end_time = Env::Default()->NowMicros();

    std::string chunk_msg;
    if (chunked_message_.chunked_fields().empty()) {
      chunk_msg = "No chunks were generated.";
    } else {
      chunk_msg = absl::StrCat(
          "Generated ", chunked_message_.chunked_fields_size(), " chunks.");
    }
    LOG(INFO) << "Finished chunking '" << message_->GetDescriptor()->full_name()
              << "', took " << HumanReadableDuration(end_time - start_time)
              << ". " << chunk_msg;
    built_ = true;
  }
  return (ChunkedProto){.chunks = &chunks_,
                        .chunked_message = &chunked_message_};
}

template <typename T>
static absl::Status WriteToRecordWriter(
    riegeli::RecordWriter<T>& writer, const std::vector<MessageBytes>& chunks,
    ChunkedMessage& chunked_message,
    const ::tensorflow::proto_splitter::VersionDef& version) {
  // Export Riegeli / chunked file.
  ChunkMetadata metadata;
  *metadata.mutable_message() = chunked_message;
  *metadata.mutable_version() = version;
  auto* metadata_chunks = metadata.mutable_chunks();

  for (const auto& chunk : chunks) {
    auto* chunk_metadata = metadata_chunks->Add();
    if (std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(
            chunk)) {
      const auto& msg_chunk =
          std::get<std::shared_ptr<tsl::protobuf::Message>>(chunk);
      LOG(INFO) << "Writing chunk of size " << msg_chunk->ByteSizeLong();
      writer.WriteRecord(*msg_chunk);
      chunk_metadata->set_size(msg_chunk->ByteSizeLong());
      chunk_metadata->set_type(
          ::tensorflow::proto_splitter::ChunkInfo::MESSAGE);
    } else if (std::holds_alternative<tsl::protobuf::Message*>(chunk)) {
      auto* msg_chunk = std::get<tsl::protobuf::Message*>(chunk);
      writer.WriteRecord(*msg_chunk);
      chunk_metadata->set_size(msg_chunk->ByteSizeLong());
      chunk_metadata->set_type(
          ::tensorflow::proto_splitter::ChunkInfo::MESSAGE);
    } else {
      const auto& str_chunk = std::get<std::string>(chunk);
      writer.WriteRecord(str_chunk);
      chunk_metadata->set_size(str_chunk.size());
      chunk_metadata->set_type(::tensorflow::proto_splitter::ChunkInfo::BYTES);
    }
    chunk_metadata->set_offset(writer.LastPos().get().numeric());
  }
  writer.WriteRecord(metadata);
  return absl::OkStatus();
}

absl::Status ComposableSplitterBase::CheckIfWriteImplemented() {
  if (parent_splitter_ != nullptr) {
    return absl::UnimplementedError(
        "The `Write` function behavior for children ComposableSplitter has not "
        "been defined. Please call `parent_splitter.Write()` instead.");
  }
  return absl::OkStatus();
}

absl::Status ComposableSplitterBase::Write(std::string file_prefix) {
  TF_RETURN_IF_ERROR(CheckIfWriteImplemented());

  auto split_results = Split();
  if (!split_results.ok()) return split_results.status();

  std::vector<MessageBytes>* chunks = split_results.value().chunks;
  ChunkedMessage* chunked_message = split_results.value().chunked_message;

  tsl::Env* env = tsl::Env::Default();
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(
      std::string{tensorflow::io::Dirname(file_prefix)}));

  std::string output_path;
  if (chunked_message->chunked_fields().empty()) {
    // Export regular pb.
    output_path = absl::StrCat(file_prefix, ".pb");
    TF_RETURN_IF_ERROR(
        tensorflow::WriteBinaryProto(env, output_path, *message_));
  } else {
    // Export Riegeli / chunked file.
    output_path = absl::StrCat(file_prefix, ".cpb");
    using WriterType = riegeli::FdWriter<>;
    riegeli::RecordWriter<WriterType> writer((WriterType(output_path)));
    if (!writer.is_open()) return writer.status();
    TF_RETURN_IF_ERROR(WriteToRecordWriter<WriterType>(
        writer, *chunks, *chunked_message, Version()));
    if (!writer.Close()) return writer.status();
  }
  LOG(INFO) << "Splitter output written to " << output_path;
  return absl::OkStatus();
}

absl::StatusOr<std::tuple<std::string, bool>>
ComposableSplitterBase::WriteToString() {
  TF_RETURN_IF_ERROR(CheckIfWriteImplemented());

  auto split_results = Split();
  if (!split_results.ok()) return split_results.status();
  std::vector<MessageBytes>* chunks = split_results.value().chunks;
  ChunkedMessage* chunked_message = split_results.value().chunked_message;

  std::string output;
  if (chunked_message->chunked_fields().empty()) {
    // Export regular pb.
    if (!message_->SerializeToString(&output))
      return absl::InvalidArgumentError("Serialization to string failed");
    LOG(INFO) << "Splitter output written to string";
    return std::make_tuple(output, false);
  } else {
    // Export Riegeli / chunked file.
    using WriterType = riegeli::StringWriter<>;
    riegeli::RecordWriter<WriterType> writer((WriterType(&output)));
    if (!writer.is_open()) return writer.status();
    TF_RETURN_IF_ERROR(WriteToRecordWriter<WriterType>(
        writer, *chunks, *chunked_message, Version()));
    if (!writer.Close()) return writer.status();
    LOG(INFO) << "Splitter output written to string";
    return std::make_tuple(output, true);
  }
}

#if !IS_OSS
absl::StatusOr<std::tuple<absl::Cord, bool>>
ComposableSplitterBase::WriteToCord() {
  TF_RETURN_IF_ERROR(CheckIfWriteImplemented());

  auto split_results = Split();
  if (!split_results.ok()) return split_results.status();
  std::vector<MessageBytes>* chunks = split_results.value().chunks;
  ChunkedMessage* chunked_message = split_results.value().chunked_message;

  absl::Cord output;
  if (chunked_message->chunked_fields().empty()) {
    // Export regular pb.
    if (!message_->SerializeToCord(&output))
      return absl::InvalidArgumentError("Serialization to absl::Cord failed");
    LOG(INFO) << "Splitter output written to absl::Cord";
    return std::make_tuple(output, false);
  } else {
    // Export Riegeli / chunked file.
    using WriterType = riegeli::CordWriter<>;
    riegeli::RecordWriter<WriterType> writer((WriterType(&output)));
    if (!writer.is_open()) return writer.status();
    TF_RETURN_IF_ERROR(WriteToRecordWriter<WriterType>(
        writer, *chunks, *chunked_message, Version()));
    if (!writer.Close()) return writer.status();
    LOG(INFO) << "Splitter output written to absl::Cord";
    return std::make_tuple(output, true);
  }
}
#endif

absl::Status ComposableSplitterBase::SetMessageAsBaseChunk() {
  if (!chunks_.empty()) {
    return absl::FailedPreconditionError(
        "Cannot set `message` as the base chunk since there are already "
        "created chunks.");
  }

  chunks_.push_back(message_);
  chunked_message_.set_chunk_index(0);
  chunks_order_.push_back(0);
  add_chunk_order_.push_back(0);
  return absl::OkStatus();
}

absl::Status ComposableSplitterBase::AddChunk(
    std::unique_ptr<MessageBytes> chunk, std::vector<FieldType>* fields,
    int* index) {
  if (parent_splitter_ != nullptr) {
    std::vector<FieldType> all_fields(fields_in_parent_->begin(),
                                      fields_in_parent_->end());
    all_fields.insert(all_fields.end(), fields->begin(), fields->end());
    return parent_splitter_->AddChunk(std::move(chunk), &all_fields, index);
  }

  auto new_chunk_index = chunks_.size();
  auto new_field = chunked_message_.add_chunked_fields();
  new_field->mutable_message()->set_chunk_index(new_chunk_index);
  TF_RETURN_IF_ERROR(
      AddFieldTag(*message_->GetDescriptor(), *fields, *new_field));

  // Add chunk at the end or insert at the index position.
  if (index == nullptr) {
    chunks_.push_back(*chunk);
    chunks_order_.push_back(new_chunk_index);
  } else {
    auto it = chunks_.begin();
    std::advance(it, *index);
    chunks_.insert(it, *chunk);

    fix_chunk_order_ = true;
    auto it2 = chunks_order_.begin();
    std::advance(it2, *index);
    chunks_order_.insert(it2, new_chunk_index);
  }
  add_chunk_order_.push_back(new_chunk_index);
  return absl::OkStatus();
}

absl::Status ComposableSplitterBase::FixChunks() {
  if (!fix_chunk_order_) return absl::OkStatus();

  // Use `add_chunk_order_` and `chunks_order_` to update the chunk indices.
  absl::flat_hash_map<int, int> chunk_indices;
  for (int i = 0; i < chunks_order_.size(); ++i) {
    chunk_indices[chunks_order_[i]] = i;
  }

  std::deque<ChunkedMessage*> to_fix = {&chunked_message_};
  while (!to_fix.empty()) {
    auto msg = to_fix.front();
    to_fix.pop_front();
    for (int i = 0; i < msg->chunked_fields_size(); ++i) {
      to_fix.push_back(msg->mutable_chunked_fields(i)->mutable_message());
    }

    if (!msg->has_chunk_index()) continue;
    int current_chunk_idx = msg->chunk_index();
    int new_chunk_index = chunk_indices[add_chunk_order_[current_chunk_idx]];
    msg->set_chunk_index(new_chunk_index);
  }
  fix_chunk_order_ = false;
  return absl::OkStatus();
}

}  // namespace tools::proto_splitter
}  // namespace tensorflow

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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_MERGE_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_MERGE_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow::tools::proto_splitter {

class Merger {
 private:
  enum MergerOp { MERGE, READ };

 public:
  // Merges the provided `chunks` into `merged_message` using `chunked_message`.
  // Example usage:
  //   std::vector<tsl::protobuf::Message> chunks = GetMyChunks();
  //   ::proto_splitter::ChunkedMessage chunked_message = GetMyChunkedMessage();
  //   my_project::MyProto my_proto;
  //   Merger::Merge(chunks, chunked_message, &my_proto);
  // TODO(b/282775853): Integrate Splitter return type with Merge input type
  static absl::Status Merge(
      const std::vector<std::unique_ptr<tsl::protobuf::Message>>& chunks,
      const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
      tsl::protobuf::Message* merged_message);

  // Reads a TF SavedModel chunked protobuf from `prefix` (must be .pb or .cpb)
  // into `merged_message`. The proto format of `merged_message` must match the
  // format of the proto written to `prefix`.
  // Example usage:
  //   my_project::MyProto my_proto;
  //   Merger::Read("path/to/saved_model", &my_proto);
  static absl::Status Read(std::string prefix,
                           tsl::protobuf::Message* merged_message);

  // Like `Merger::Read`, but only reads what's specified in `chunk_metadata`.
  static absl::Status ReadPartial(
      absl::string_view prefix,
      const ::tensorflow::proto_splitter::ChunkMetadata& chunk_metadata,
      tsl::protobuf::Message* merged_message);

 private:
  // Reads a normal saved_model.pb proto in.
  static absl::Status ReadPb(const std::string& pb_file,
                             tsl::protobuf::Message* merged_message);

  // Uses metadata contained in `chunked_message` to fill `merged_message` with
  // data accessed by the `reader` using `chunks_info`.
  static absl::Status ReadFields(
      const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
      riegeli::RecordReader<riegeli::FdReader<>>& reader,
      const std::vector<::tensorflow::proto_splitter::ChunkInfo>&
          chunks_info,  // TODO(adamcogdell): this can just be a
                        // RepeatedPtrField
      tsl::protobuf::Message* merged_message);

  // Processes a single `chunked_field` within a `chunked_message`. If the field
  // itself is a `chunked_message` that contains additional `chunked_fields`,
  // either MergeFields or ReadFields is called to recursively (depending on the
  // value of `op`) to add those fields to `merged_message`. Otherwise, the
  // field is simply added to `merged_message` using reflection.
  static absl::Status ProcessField(
      const ::tensorflow::proto_splitter::ChunkedField& chunked_field,
      tsl::protobuf::Message* merged_message,
      const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info,
      const std::vector<std::unique_ptr<tsl::protobuf::Message>>& chunks,
      riegeli::RecordReader<riegeli::FdReader<>>& reader, MergerOp op);
};

}  // namespace tensorflow::tools::proto_splitter

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_MERGE_H_

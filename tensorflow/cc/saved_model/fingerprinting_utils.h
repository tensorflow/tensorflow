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

#ifndef TENSORFLOW_CC_SAVED_MODEL_FINGERPRINTING_UTILS_H_
#define TENSORFLOW_CC_SAVED_MODEL_FINGERPRINTING_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/fd_reader.h"  // from @riegeli
#include "riegeli/records/record_reader.h"  // from @riegeli
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tensorflow/core/protobuf/saved_object_graph.pb.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"

namespace tensorflow::saved_model::fingerprinting {

namespace fingerprinting_utils_internal {

using ::tensorflow::protobuf::Map;
using ::tensorflow::protobuf::Message;
using ::tensorflow::protobuf::RepeatedPtrField;

// Number of sequential FieldIndex matches of `a` in `b`. (Length of initial
// subsequence.)
// Example: `a = {4, 2}`, `b = {4, 2, 1, 3}`, `fieldTagMatches(a, b) == 2`
absl::StatusOr<int> fieldTagMatches(
    const RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>& a,
    const RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>& b);

// Pull out the relevant data within `chunked_message`. A `chunked_field` is
// relevant if its `field_tags` are an initial subsequence any of the
// `target_fields` in the provided `target_fields_list`.
absl::StatusOr<::tensorflow::proto_splitter::ChunkedMessage>
PruneChunkedMessage(
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    std::vector<::tensorflow::proto_splitter::ChunkInfo> chunks_info,
    std::vector<RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>>
        target_fields_list);

// Deterministically serializes the proto `message`.
std::string SerializeProto(const Message& message);

// Uses metadata contained in `chunked_message` to hash fields within the
// data accessed by the `reader` using `chunks_info`.
absl::StatusOr<uint64_t> HashFields(
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info,
    const RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>&
        field_tags,
    Message* merged_message);

// Gets the field tags for `graph_def`.::tensorflow
inline RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>
GraphDefFieldTags();

// Gets the field tags for `signature_def`.
inline RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>
SignatureDefFieldTags();

// Gets the field tags for `saved_object_graph`.
inline RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>
SavedObjectGraphFieldTags();

// Returns a `SavedModel` containing only fields (up to those) specified by
// `GraphDefFieldTags()`, `SignatureDefFieldTags()`, and
// `SavedObjectGraphFieldTags()`.
absl::StatusOr<tensorflow::SavedModel> PrunedSavedModel(
    absl::string_view export_dir,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info,
    ::tensorflow::proto_splitter::ChunkMetadata& chunk_metadata);

// Hashes the contents of `message` specified by `field_tags`.
absl::StatusOr<uint64_t> HashMessage(
    Message* message,
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info,
    const RepeatedPtrField<::tensorflow::proto_splitter::FieldIndex>&
        field_tags);

// Hashes the contents of `graph_def`.
absl::StatusOr<uint64_t> HashGraphDef(
    tensorflow::GraphDef* graph_def,
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info);

// Hashes the contents of `signature_def`.
absl::StatusOr<uint64_t> HashSignatureDef(
    const Map<std::string, ::tensorflow::SignatureDef>& signature_def_map,
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info);

// Hashes the contents of `saved_object_graph`.
absl::StatusOr<uint64_t> HashSavedObjectGraph(
    tensorflow::SavedObjectGraph* saved_object_graph,
    const ::tensorflow::proto_splitter::ChunkedMessage& chunked_message,
    riegeli::RecordReader<riegeli::FdReader<>>& reader,
    const std::vector<::tensorflow::proto_splitter::ChunkInfo>& chunks_info);

}  // namespace fingerprinting_utils_internal

// Returns a random UUID (128 bits random) as a string.
std::string CreateRandomUUID();

// Returns the hash of the checkpoint .index file, 0 if there is none.
uint64_t HashCheckpointIndexFile(absl::string_view model_dir);

// Creates a FingerprintDef proto from a chunked SavedModel and the checkpoint
// meta file (.index) in `export_dir`.
absl::StatusOr<FingerprintDef> CreateFingerprintDefCpb(
    absl::string_view export_dir, std::string cpb_file);

}  // namespace tensorflow::saved_model::fingerprinting

#endif  // TENSORFLOW_CC_SAVED_MODEL_FINGERPRINTING_UTILS_H_

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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_COMPOSABLE_SPLITTER_BASE_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_COMPOSABLE_SPLITTER_BASE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "tensorflow/tools/proto_splitter/cc/split.h"
#include "tensorflow/tools/proto_splitter/cc/util.h"
#include "tensorflow/tools/proto_splitter/chunk.pb.h"
#include "tsl/platform/protobuf.h"

#define IS_OSS true

namespace tensorflow {
namespace tools::proto_splitter {

// Parts of the ComposableSplitter that are independent of the template (to
// avoid linking issues).
class ComposableSplitterBase : public Splitter {
 public:
  explicit ComposableSplitterBase(tsl::protobuf::Message* message)
      : built_(false), message_(message), parent_splitter_(nullptr) {}

  explicit ComposableSplitterBase(tsl::protobuf::Message* message,
                                  ComposableSplitterBase* parent_splitter,
                                  std::vector<FieldType>* fields_in_parent)
      : built_(false),
        message_(message),
        parent_splitter_(parent_splitter),
        fields_in_parent_(fields_in_parent) {}

  // Splits a proto message.
  // Returns a pair of (
  //   chunks: List of messages/bytes,
  //   ChunkedMessage: Metadata about the chunked fields.)
  // If the message is not split, `chunks` should only contain the original
  // message.
  absl::StatusOr<ChunkedProto> Split() override;

  // Serializes a proto to disk.
  // The writer writes all chunks into a Riegeli file. The chunk metadata
  // (ChunkMetadata) is written at the very end.
  //   file_prefix: string prefix of the filepath. The writer will automatically
  //     attach a `.pb` or `.cpb` (chunked pb) suffix depending on whether the
  //     proto is split.
  absl::Status Write(std::string file_prefix) override;
  // The bool field record whether it's saved as a chunked protobuf (true) or
  // regular protobuf (false).
  absl::StatusOr<std::tuple<std::string, bool>> WriteToString();
#if !IS_OSS
  absl::StatusOr<std::tuple<absl::Cord, bool>> WriteToCord();
#endif

  VersionDef Version() override;

  // Builds the Splitter object by generating chunks from the proto.
  // Subclasses of `ComposableChunks` should only need to override this method.
  // This method should be called once per Splitter to create the chunks.
  // Users should call the methods `Split` or `Write` instead.
  virtual absl::Status BuildChunks() = 0;

  // Set or get the size of the initial user-provided proto.
  void SetInitialSize(size_t size);
  size_t GetInitialSize();

 protected:
  // Initializes `chunks_` and `chunked_message_`, with the user-provided
  // message as the initial chunk. This step is optional.
  absl::Status SetMessageAsBaseChunk();

  // Adds a new chunk and updates the ChunkedMessage proto. If set, the index
  // indicates where to insert the chunk.
  absl::Status AddChunk(std::unique_ptr<MessageBytes> chunk,
                        std::vector<FieldType>* fields, int* index = nullptr);

  std::vector<FieldType>* fields_in_parent() { return fields_in_parent_; }

 private:
  // After BuildChunks(), the chunk_index of each nested ChunkedMessage is set
  // to the length of the list when the chunk was added. This would be fine if
  // the chunks were always added to the end of the list. However, this is not
  // always the case the indices must be updated.
  absl::Status FixChunks();
  absl::Status CheckIfWriteImplemented();

  bool built_;
  tsl::protobuf::Message* message_;
  std::vector<MessageBytes> chunks_;
  ::tensorflow::proto_splitter::ChunkedMessage chunked_message_;
  ComposableSplitterBase* parent_splitter_;
  std::vector<FieldType>* fields_in_parent_;
  size_t size_ = 0;

  // Keep list of chunk keys in the order in which they were added to
  // the list and the actual order of chunks. If the user inserts a
  // chunk out of order, then these arrays will be used to fix the message chunk
  // indices.
  std::vector<int> add_chunk_order_;
  std::vector<int> chunks_order_;
  bool fix_chunk_order_ = false;
};

}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_COMPOSABLE_SPLITTER_BASE_H_

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/lib/io/table_builder.h"

#include <assert.h>

#include "absl/strings/string_view.h"
#include "xla/tsl/lib/hash/crc32c.h"
#include "xla/tsl/lib/io/block_builder.h"
#include "xla/tsl/lib/io/format.h"
#include "xla/tsl/lib/io/table_options.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/coding.h"
#include "tsl/platform/snappy.h"

namespace tsl {
namespace table {

namespace {

void FindShortestSeparator(string* start, absl::string_view limit) {
  // Find length of common prefix
  size_t min_length = std::min(start->size(), limit.size());
  size_t diff_index = 0;
  while ((diff_index < min_length) &&
         ((*start)[diff_index] == limit[diff_index])) {
    diff_index++;
  }

  if (diff_index >= min_length) {
    // Do not shorten if one string is a prefix of the other
  } else {
    uint8 diff_byte = static_cast<uint8>((*start)[diff_index]);
    if (diff_byte < static_cast<uint8>(0xff) &&
        diff_byte + 1 < static_cast<uint8>(limit[diff_index])) {
      (*start)[diff_index]++;
      start->resize(diff_index + 1);
      assert(absl::string_view(*start).compare(limit) < 0);
    }
  }
}

void FindShortSuccessor(string* key) {
  // Find first character that can be incremented
  size_t n = key->size();
  for (size_t i = 0; i < n; i++) {
    const uint8 byte = (*key)[i];
    if (byte != static_cast<uint8>(0xff)) {
      (*key)[i] = byte + 1;
      key->resize(i + 1);
      return;
    }
  }
  // *key is a run of 0xffs.  Leave it alone.
}
}  // namespace

struct TableBuilder::Rep {
  Options options;
  Options index_block_options;
  WritableFile* file;
  uint64 offset;
  absl::Status status;
  BlockBuilder data_block;
  BlockBuilder index_block;
  string last_key;
  int64_t num_entries;
  bool closed;  // Either Finish() or Abandon() has been called.

  // We do not emit the index entry for a block until we have seen the
  // first key for the next data block.  This allows us to use shorter
  // keys in the index block.  For example, consider a block boundary
  // between the keys "the quick brown fox" and "the who".  We can use
  // "the r" as the key for the index block entry since it is >= all
  // entries in the first block and < all entries in subsequent
  // blocks.
  //
  // Invariant: r->pending_index_entry is true only if data_block is empty.
  bool pending_index_entry;
  BlockHandle pending_handle;  // Handle to add to index block

  string compressed_output;

  Rep(const Options& opt, WritableFile* f)
      : options(opt),
        index_block_options(opt),
        file(f),
        offset(0),
        data_block(&options),
        index_block(&index_block_options),
        num_entries(0),
        closed(false),
        pending_index_entry(false) {
    index_block_options.block_restart_interval = 1;
  }
};

TableBuilder::TableBuilder(const Options& options, WritableFile* file)
    : rep_(new Rep(options, file)) {}

TableBuilder::~TableBuilder() {
  assert(rep_->closed);  // Catch errors where caller forgot to call Finish()
  delete rep_;
}

void TableBuilder::Add(absl::string_view key, absl::string_view value) {
  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->num_entries > 0) {
    assert(key.compare(absl::string_view(r->last_key)) > 0);
    // See if this key+value would make our current block overly large.  If
    // so, emit the current block before adding this key/value
    const int kOverlyLargeBlockRatio = 2;
    const size_t this_entry_bytes = key.size() + value.size();
    if (this_entry_bytes >= kOverlyLargeBlockRatio * r->options.block_size) {
      Flush();
    }
  }

  if (r->pending_index_entry) {
    assert(r->data_block.empty());
    FindShortestSeparator(&r->last_key, key);
    string handle_encoding;
    r->pending_handle.EncodeTo(&handle_encoding);
    r->index_block.Add(r->last_key, absl::string_view(handle_encoding));
    r->pending_index_entry = false;
  }

  r->last_key.assign(key.data(), key.size());
  r->num_entries++;
  r->data_block.Add(key, value);

  const size_t estimated_block_size = r->data_block.CurrentSizeEstimate();
  if (estimated_block_size >= r->options.block_size) {
    Flush();
  }
}

void TableBuilder::Flush() {
  Rep* r = rep_;
  assert(!r->closed);
  if (!ok()) return;
  if (r->data_block.empty()) return;
  assert(!r->pending_index_entry);
  WriteBlock(&r->data_block, &r->pending_handle);
  if (ok()) {
    r->pending_index_entry = true;
    // We don't flush the underlying file as that can be slow.
  }
}

void TableBuilder::WriteBlock(BlockBuilder* block, BlockHandle* handle) {
  // File format contains a sequence of blocks where each block has:
  //    block_data: uint8[n]
  //    type: uint8
  //    crc: uint32
  assert(ok());
  Rep* r = rep_;
  absl::string_view raw = block->Finish();

  absl::string_view block_contents;
  CompressionType type = r->options.compression;
  // TODO(postrelease): Support more compression options: zlib?
  switch (type) {
    case kNoCompression:
      block_contents = raw;
      break;

    case kSnappyCompression: {
      string* compressed = &r->compressed_output;
      if (port::Snappy_Compress(raw.data(), raw.size(), compressed) &&
          compressed->size() < raw.size() - (raw.size() / 8u)) {
        block_contents = *compressed;
      } else {
        // Snappy not supported, or compressed less than 12.5%, so just
        // store uncompressed form
        block_contents = raw;
        type = kNoCompression;
      }
      break;
    }
  }
  WriteRawBlock(block_contents, type, handle);
  r->compressed_output.clear();
  block->Reset();
}

void TableBuilder::WriteRawBlock(absl::string_view block_contents,
                                 CompressionType type, BlockHandle* handle) {
  Rep* r = rep_;
  handle->set_offset(r->offset);
  handle->set_size(block_contents.size());
  r->status = r->file->Append(block_contents);
  if (r->status.ok()) {
    char trailer[kBlockTrailerSize];
    trailer[0] = type;
    uint32 crc = crc32c::Value(block_contents.data(), block_contents.size());
    crc = crc32c::Extend(crc, trailer, 1);  // Extend crc to cover block type
    core::EncodeFixed32(trailer + 1, crc32c::Mask(crc));
    r->status = r->file->Append(absl::string_view(trailer, kBlockTrailerSize));
    if (r->status.ok()) {
      r->offset += block_contents.size() + kBlockTrailerSize;
    }
  }
}

absl::Status TableBuilder::status() const { return rep_->status; }

absl::Status TableBuilder::Finish() {
  Rep* r = rep_;
  Flush();
  assert(!r->closed);
  r->closed = true;

  BlockHandle metaindex_block_handle, index_block_handle;

  // Write metaindex block
  if (ok()) {
    BlockBuilder meta_index_block(&r->options);
    // TODO(postrelease): Add stats and other meta blocks
    WriteBlock(&meta_index_block, &metaindex_block_handle);
  }

  // Write index block
  if (ok()) {
    if (r->pending_index_entry) {
      FindShortSuccessor(&r->last_key);
      string handle_encoding;
      r->pending_handle.EncodeTo(&handle_encoding);
      r->index_block.Add(r->last_key, absl::string_view(handle_encoding));
      r->pending_index_entry = false;
    }
    WriteBlock(&r->index_block, &index_block_handle);
  }

  // Write footer
  if (ok()) {
    Footer footer;
    footer.set_metaindex_handle(metaindex_block_handle);
    footer.set_index_handle(index_block_handle);
    string footer_encoding;
    footer.EncodeTo(&footer_encoding);
    r->status = r->file->Append(footer_encoding);
    if (r->status.ok()) {
      r->offset += footer_encoding.size();
    }
  }
  return r->status;
}

void TableBuilder::Abandon() {
  Rep* r = rep_;
  assert(!r->closed);
  r->closed = true;
}

uint64 TableBuilder::NumEntries() const { return rep_->num_entries; }

uint64 TableBuilder::FileSize() const { return rep_->offset; }

}  // namespace table
}  // namespace tsl

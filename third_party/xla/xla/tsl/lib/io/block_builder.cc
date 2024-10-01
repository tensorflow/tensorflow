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

// BlockBuilder generates blocks where keys are prefix-compressed:
//
// When we store a key, we drop the prefix shared with the previous
// string.  This helps reduce the space requirement significantly.
// Furthermore, once every K keys, we do not apply the prefix
// compression and store the entire key.  We call this a "restart
// point".  The tail end of the block stores the offsets of all of the
// restart points, and can be used to do a binary search when looking
// for a particular key.  Values are stored as-is (without compression)
// immediately following the corresponding key.
//
// An entry for a particular key-value pair has the form:
//     shared_bytes: varint32
//     unshared_bytes: varint32
//     value_length: varint32
//     key_delta: char[unshared_bytes]
//     value: char[value_length]
// shared_bytes == 0 for restart points.
//
// The trailer of the block has the form:
//     restarts: uint32[num_restarts]
//     num_restarts: uint32
// restarts[i] contains the offset within the block of the ith restart point.

#include "xla/tsl/lib/io/block_builder.h"

#include <assert.h>

#include <algorithm>

#include "xla/tsl/lib/io/table_builder.h"
#include "tsl/platform/coding.h"

namespace tsl {
namespace table {

BlockBuilder::BlockBuilder(const Options* options)
    : options_(options), restarts_(), counter_(0), finished_(false) {
  assert(options->block_restart_interval >= 1);
  restarts_.push_back(0);  // First restart point is at offset 0
}

void BlockBuilder::Reset() {
  buffer_.clear();
  restarts_.clear();
  restarts_.push_back(0);  // First restart point is at offset 0
  counter_ = 0;
  finished_ = false;
  last_key_.clear();
}

size_t BlockBuilder::CurrentSizeEstimate() const {
  return (buffer_.size() +                     // Raw data buffer
          restarts_.size() * sizeof(uint32) +  // Restart array
          sizeof(uint32));                     // Restart array length
}

absl::string_view BlockBuilder::Finish() {
  // Append restart array
  CHECK_LE(restarts_.size(), std::numeric_limits<uint32_t>::max());
  for (const auto r : restarts_) {
    core::PutFixed32(&buffer_, r);
  }
  // Downcast safe because of the CHECK.
  core::PutFixed32(&buffer_, static_cast<uint32_t>(restarts_.size()));
  finished_ = true;
  return absl::string_view(buffer_);
}

void BlockBuilder::Add(const absl::string_view& key,
                       const absl::string_view& value) {
  absl::string_view last_key_piece(last_key_);
  assert(!finished_);
  assert(counter_ <= options_->block_restart_interval);
  assert(buffer_.empty()  // No values yet?
         || key.compare(last_key_piece) > 0);
  size_t shared = 0;
  if (counter_ < options_->block_restart_interval) {
    // See how much sharing to do with previous string
    const size_t min_length = std::min(last_key_piece.size(), key.size());
    while ((shared < min_length) && (last_key_piece[shared] == key[shared])) {
      shared++;
    }
  } else {
    // Restart compression
    CHECK_LE(buffer_.size(), std::numeric_limits<uint32_t>::max());
    restarts_.push_back(static_cast<uint32_t>(buffer_.size()));
    counter_ = 0;
  }
  const size_t non_shared = key.size() - shared;

  CHECK_LE(shared, std::numeric_limits<uint32_t>::max());
  CHECK_LE(non_shared, std::numeric_limits<uint32_t>::max());
  CHECK_LE(value.size(), std::numeric_limits<uint32_t>::max());

  // Add "<shared><non_shared><value_size>" to buffer_
  core::PutVarint32(&buffer_, static_cast<uint32_t>(shared));
  core::PutVarint32(&buffer_, static_cast<uint32_t>(non_shared));
  core::PutVarint32(&buffer_, static_cast<uint32_t>(value.size()));

  // Add string delta to buffer_ followed by value
  buffer_.append(key.data() + shared, non_shared);
  buffer_.append(value.data(), static_cast<uint32_t>(value.size()));

  // Update state
  last_key_.resize(shared);
  last_key_.append(key.data() + shared, non_shared);
  assert(absl::string_view(last_key_) == key);
  counter_++;
}

}  // namespace table
}  // namespace tsl

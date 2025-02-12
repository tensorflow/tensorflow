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

#include "xla/tsl/lib/io/format.h"

#include <limits>

#include "xla/tsl/lib/hash/crc32c.h"
#include "xla/tsl/lib/io/block.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/coding.h"
#include "tsl/platform/raw_coding.h"
#include "tsl/platform/snappy.h"

namespace tsl {
namespace table {

void BlockHandle::EncodeTo(string* dst) const {
  // Sanity check that all fields have been set
  assert(offset_ != ~static_cast<uint64>(0));
  assert(size_ != ~static_cast<uint64>(0));
  core::PutVarint64(dst, offset_);
  core::PutVarint64(dst, size_);
}

absl::Status BlockHandle::DecodeFrom(absl::string_view* input) {
  if (core::GetVarint64(input, &offset_) && core::GetVarint64(input, &size_)) {
    return absl::OkStatus();
  } else {
    return errors::DataLoss("bad block handle");
  }
}

void Footer::EncodeTo(string* dst) const {
#ifndef NDEBUG
  const size_t original_size = dst->size();
#endif
  metaindex_handle_.EncodeTo(dst);
  index_handle_.EncodeTo(dst);
  dst->resize(2 * BlockHandle::kMaxEncodedLength);  // Padding
  core::PutFixed32(dst, static_cast<uint32>(kTableMagicNumber & 0xffffffffu));
  core::PutFixed32(dst, static_cast<uint32>(kTableMagicNumber >> 32));
  assert(dst->size() == original_size + kEncodedLength);
}

absl::Status Footer::DecodeFrom(absl::string_view* input) {
  const char* magic_ptr = input->data() + kEncodedLength - 8;
  const uint32 magic_lo = core::DecodeFixed32(magic_ptr);
  const uint32 magic_hi = core::DecodeFixed32(magic_ptr + 4);
  const uint64 magic =
      ((static_cast<uint64>(magic_hi) << 32) | (static_cast<uint64>(magic_lo)));
  if (magic != kTableMagicNumber) {
    return errors::DataLoss("not an sstable (bad magic number)");
  }

  absl::Status result = metaindex_handle_.DecodeFrom(input);
  if (result.ok()) {
    result = index_handle_.DecodeFrom(input);
  }
  if (result.ok()) {
    // We skip over any leftover data (just padding for now) in "input"
    const char* end = magic_ptr + 8;
    *input = absl::string_view(end, input->data() + input->size() - end);
  }
  return result;
}

absl::Status ReadBlock(RandomAccessFile* file, const BlockHandle& handle,
                       BlockContents* result) {
  result->data = absl::string_view();
  result->cacheable = false;
  result->heap_allocated = false;

  // Read the block contents as well as the type/crc footer.
  // See table_builder.cc for the code that built this structure.
  size_t n = static_cast<size_t>(handle.size());

  if (kBlockTrailerSize > std::numeric_limits<size_t>::max() - n) {
    return errors::DataLoss("handle.size() too big");
  }

  char* buf = new char[n + kBlockTrailerSize];
  absl::string_view contents;
  absl::Status s =
      file->Read(handle.offset(), n + kBlockTrailerSize, &contents, buf);
  if (!s.ok()) {
    delete[] buf;
    return s;
  }
  if (contents.size() != n + kBlockTrailerSize) {
    delete[] buf;
    return errors::DataLoss("truncated block read");
  }

  // Check the crc of the type and the block contents
  const char* data = contents.data();  // Pointer to where Read put the data
  // This checksum verification is optional.  We leave it on for now
  const bool verify_checksum = true;
  if (verify_checksum) {
    const uint32 crc = crc32c::Unmask(core::DecodeFixed32(data + n + 1));
    const uint32 actual = crc32c::Value(data, n + 1);
    if (actual != crc) {
      delete[] buf;
      s = errors::DataLoss("block checksum mismatch");
      return s;
    }
  }

  switch (data[n]) {
    case kNoCompression:
      if (data != buf) {
        // File implementation gave us pointer to some other data.
        // Use it directly under the assumption that it will be live
        // while the file is open.
        delete[] buf;
        result->data = absl::string_view(data, n);
        result->heap_allocated = false;
        result->cacheable = false;  // Do not double-cache
      } else {
        result->data = absl::string_view(buf, n);
        result->heap_allocated = true;
        result->cacheable = true;
      }

      // Ok
      break;
    case kSnappyCompression: {
      size_t ulength = 0;
      if (!port::Snappy_GetUncompressedLength(data, n, &ulength)) {
        delete[] buf;
        return errors::DataLoss("corrupted compressed block contents");
      }
      char* ubuf = new char[ulength];
      if (!port::Snappy_Uncompress(data, n, ubuf)) {
        delete[] buf;
        delete[] ubuf;
        return errors::DataLoss("corrupted compressed block contents");
      }
      delete[] buf;
      result->data = absl::string_view(ubuf, ulength);
      result->heap_allocated = true;
      result->cacheable = true;
      break;
    }
    default:
      delete[] buf;
      return errors::DataLoss("bad block type");
  }

  return absl::OkStatus();
}

}  // namespace table
}  // namespace tsl

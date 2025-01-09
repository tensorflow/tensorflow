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

#include "xla/tsl/lib/io/inputbuffer.h"

#include <algorithm>

#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace tsl {
namespace io {

InputBuffer::InputBuffer(RandomAccessFile* file, size_t buffer_bytes)
    : file_(file),
      file_pos_(0),
      size_(buffer_bytes),
      buf_(new char[size_]),
      pos_(buf_),
      limit_(buf_) {}

InputBuffer::~InputBuffer() { delete[] buf_; }

absl::Status InputBuffer::FillBuffer() {
  absl::string_view data;
  absl::Status s = file_->Read(file_pos_, size_, &data, buf_);
  if (data.data() != buf_) {
    memmove(buf_, data.data(), data.size());
  }
  pos_ = buf_;
  limit_ = pos_ + data.size();
  file_pos_ += data.size();
  return s;
}

template <typename T>
absl::Status InputBuffer::ReadLine(T* result) {
  result->clear();
  absl::Status s;
  do {
    size_t buf_remain = limit_ - pos_;
    char* newline = static_cast<char*>(memchr(pos_, '\n', buf_remain));
    if (newline != nullptr) {
      size_t result_len = newline - pos_;
      result->append(pos_, result_len);
      pos_ = newline + 1;
      if (!result->empty() && result->back() == '\r') {
        result->resize(result->size() - 1);
      }
      return absl::OkStatus();
    }
    if (buf_remain > 0) result->append(pos_, buf_remain);
    // Get more data into buffer
    s = FillBuffer();
    DCHECK_EQ(pos_, buf_);
  } while (limit_ != buf_);
  if (!result->empty() && result->back() == '\r') {
    result->resize(result->size() - 1);
  }
  if (errors::IsOutOfRange(s) && !result->empty()) {
    return absl::OkStatus();
  }
  return s;
}

template Status InputBuffer::ReadLine<std::string>(std::string* result);
template Status InputBuffer::ReadLine<tstring>(tstring* result);

absl::Status InputBuffer::ReadNBytes(int64_t bytes_to_read,
                                     std::string* result) {
  result->clear();
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }
  result->resize(bytes_to_read);
  size_t bytes_read = 0;
  absl::Status status = ReadNBytes(bytes_to_read, &(*result)[0], &bytes_read);
  if (bytes_read < bytes_to_read) result->resize(bytes_read);
  return status;
}

absl::Status InputBuffer::ReadNBytes(int64_t bytes_to_read, char* result,
                                     size_t* bytes_read) {
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }
  absl::Status status;
  *bytes_read = 0;
  while (*bytes_read < static_cast<size_t>(bytes_to_read)) {
    if (pos_ == limit_) {
      // Get more data into buffer.
      status = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    // Do not go over the buffer boundary.
    const int64_t bytes_to_copy =
        std::min<int64_t>(limit_ - pos_, bytes_to_read - *bytes_read);
    // Copies buffered data into the destination.
    memcpy(result + *bytes_read, pos_, bytes_to_copy);
    pos_ += bytes_to_copy;
    *bytes_read += bytes_to_copy;
  }
  if (errors::IsOutOfRange(status) &&
      (*bytes_read == static_cast<size_t>(bytes_to_read))) {
    return absl::OkStatus();
  }
  return status;
}

absl::Status InputBuffer::ReadVarint32Fallback(uint32* result) {
  absl::Status s = ReadVarintFallback(result, core::kMaxVarint32Bytes);
  if (errors::IsDataLoss(s)) {
    return errors::DataLoss("Stored data is too large to be a varint32.");
  }
  return s;
}

absl::Status InputBuffer::ReadVarint64Fallback(uint64* result) {
  absl::Status s = ReadVarintFallback(result, core::kMaxVarint64Bytes);
  if (errors::IsDataLoss(s)) {
    return errors::DataLoss("Stored data is too large to be a varint64.");
  }
  return s;
}

template <typename T>
absl::Status InputBuffer::ReadVarintFallback(T* result, int max_bytes) {
  uint8 scratch = 0;
  auto* p = reinterpret_cast<char*>(&scratch);
  size_t unused_bytes_read = 0;

  *result = 0;
  for (int index = 0; index < max_bytes; index++) {
    int shift = 7 * index;
    TF_RETURN_IF_ERROR(ReadNBytes(1, p, &unused_bytes_read));
    *result |= (static_cast<T>(scratch) & 127) << shift;
    if (!(scratch & 128)) return absl::OkStatus();
  }
  return errors::DataLoss("Stored data longer than ", max_bytes, " bytes.");
}

absl::Status InputBuffer::SkipNBytes(int64_t bytes_to_skip) {
  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can only skip forward, not ",
                                   bytes_to_skip);
  }
  int64_t bytes_skipped = 0;
  absl::Status s;
  while (bytes_skipped < bytes_to_skip) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    const int64_t bytes_to_advance =
        std::min<int64_t>(limit_ - pos_, bytes_to_skip - bytes_skipped);
    bytes_skipped += bytes_to_advance;
    pos_ += bytes_to_advance;
  }
  if (errors::IsOutOfRange(s) && bytes_skipped == bytes_to_skip) {
    return absl::OkStatus();
  }
  return s;
}

absl::Status InputBuffer::Seek(int64_t position) {
  if (position < 0) {
    return errors::InvalidArgument("Seeking to a negative position: ",
                                   position);
  }
  // Position of the buffer within file.
  const int64_t bufpos = file_pos_ - static_cast<int64_t>(limit_ - buf_);
  if (position >= bufpos && position < file_pos_) {
    // Seeks to somewhere inside the buffer.
    pos_ = buf_ + (position - bufpos);
    DCHECK(pos_ >= buf_ && pos_ < limit_);
  } else {
    // Seeks to somewhere outside.  Discards the buffered data.
    pos_ = limit_ = buf_;
    file_pos_ = position;
  }
  return absl::OkStatus();
}

absl::Status InputBuffer::Hint(int64_t bytes_to_read) {
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }

  // The internal buffer is too small. Do nothing.
  if (bytes_to_read > size_) {
    return absl::OkStatus();
  }

  const int64_t bytes_remain_in_buf = static_cast<int64_t>(limit_ - pos_);

  // There are enough data in the buffer. Do nothing.
  if (bytes_to_read <= bytes_remain_in_buf) {
    return absl::OkStatus();
  }

  // Additional read from file is necessary. Make some room.
  memmove(buf_, pos_, bytes_remain_in_buf);
  pos_ = buf_;
  limit_ = buf_ + bytes_remain_in_buf;
  bytes_to_read -= bytes_remain_in_buf;

  // Read the remaining bytes from file.
  absl::string_view data;
  absl::Status s = file_->Read(file_pos_, bytes_to_read, &data, limit_);
  if (data.data() != limit_) {
    memmove(limit_, data.data(), data.size());
  }
  limit_ += data.size();
  file_pos_ += data.size();

  if (errors::IsOutOfRange(s) && data.size() == bytes_to_read) {
    return absl::OkStatus();
  } else {
    return s;
  }
}

}  // namespace io
}  // namespace tsl

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/snappy/snappy_inputbuffer.h"

namespace tensorflow {
namespace io {
SnappyInputBuffer::SnappyInputBuffer(
    RandomAccessFile* file,
    size_t input_buffer_bytes,  // size of input_buffer_
    size_t output_buffer_bytes  // size of output_buffer_
    )
    : file_(file),
      input_buffer_capacity_(input_buffer_bytes),
      output_buffer_capacity_(output_buffer_bytes),
      input_buffer_(new char[input_buffer_capacity_]),
      output_buffer_(new char[output_buffer_capacity_]),
      next_in_(input_buffer_.get()),
      bytes_read_(0) {}

Status SnappyInputBuffer::ReadNBytes(int64_t bytes_to_read, tstring* result) {
  result->clear();
  result->resize_uninitialized(bytes_to_read);

  char* result_ptr = result->mdata();

  // Read as many bytes as possible from cache.
  size_t bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
  bytes_to_read -= bytes_read;
  result_ptr += bytes_read;

  while (bytes_to_read > 0) {
    // At this point we can be sure that cache has been emptied.
    DCHECK_EQ(avail_out_, 0);

    // Now that the cache is empty we need to inflate more data.
    TF_RETURN_IF_ERROR(Inflate());

    bytes_read = ReadBytesFromCache(bytes_to_read, result_ptr);
    bytes_to_read -= bytes_read;
    result_ptr += bytes_read;
  }

  return OkStatus();
}

int64_t SnappyInputBuffer::Tell() const { return bytes_read_; }

Status SnappyInputBuffer::Reset() {
  file_pos_ = 0;
  avail_in_ = 0;
  avail_out_ = 0;
  next_in_ = input_buffer_.get();
  bytes_read_ = 0;
  return OkStatus();
}

size_t SnappyInputBuffer::ReadBytesFromCache(size_t bytes_to_read,
                                             char* result_ptr) {
  size_t can_read_bytes = std::min(bytes_to_read, avail_out_);
  if (can_read_bytes > 0) {
    memcpy(result_ptr, next_out_, can_read_bytes);
    next_out_ += can_read_bytes;
    avail_out_ -= can_read_bytes;
  }
  bytes_read_ += can_read_bytes;
  return can_read_bytes;
}

Status SnappyInputBuffer::Inflate() {
  // Read length of compressed block.
  uint32 compressed_block_length;
  TF_RETURN_IF_ERROR(ReadCompressedBlockLength(&compressed_block_length));

  // If the entire block is not in cache do a read from file.
  if (avail_in_ < compressed_block_length) {
    TF_RETURN_IF_ERROR(ReadFromFile());
    if (avail_in_ < compressed_block_length) {
      if (compressed_block_length > input_buffer_capacity_) {
        return errors::ResourceExhausted(
            "Input buffer(size: ", input_buffer_capacity_,
            " bytes) too small. Should be larger ", "than ",
            compressed_block_length, " bytes.");
      } else {
        return errors::DataLoss(
            strings::StrCat("Failed to read ", compressed_block_length,
                            " bytes from file. Possible data corruption."));
      }
    }
  }

  size_t uncompressed_length;
  if (!port::Snappy_GetUncompressedLength(next_in_, compressed_block_length,
                                          &uncompressed_length)) {
    return errors::DataLoss("Parsing error in Snappy_GetUncompressedLength");
  }

  // Output buffer must have been cleared before uncompressing more input.
  DCHECK_EQ(avail_out_, 0);

  // Output buffer must be large enough to fit the uncompressed block.
  DCHECK_GE(output_buffer_capacity_, uncompressed_length);
  next_out_ = output_buffer_.get();

  bool status = port::Snappy_Uncompress(next_in_, compressed_block_length,
                                        output_buffer_.get());
  if (!status) {
    return errors::DataLoss("Snappy_Uncompress failed");
  }
  next_in_ += compressed_block_length;
  avail_in_ -= compressed_block_length;
  avail_out_ += uncompressed_length;
  return OkStatus();
}

Status SnappyInputBuffer::ReadCompressedBlockLength(uint32* length) {
  *length = 0;
  size_t bytes_to_read = 4;
  while (bytes_to_read > 0) {
    if (avail_in_ == 0) {
      TF_RETURN_IF_ERROR(ReadFromFile());
    }
    size_t readable = std::min(bytes_to_read, avail_in_);

    for (size_t i = 0; i < readable; i++) {
      // The "unsigned char" type cast is intentional to avoid implicit type
      // casting of the signed char to unsigned int during bitwise OR which
      // causes weird overflow errors.
      *length = (*length << 8) | static_cast<unsigned char>(next_in_[0]);
      bytes_to_read--;
      next_in_++;
      avail_in_--;
    }
  }
  return OkStatus();
}

Status SnappyInputBuffer::ReadFromFile() {
  int bytes_to_read = input_buffer_capacity_;
  char* read_location = reinterpret_cast<char*>(input_buffer_.get());

  // If there are unread bytes in the input stream we move them to the head
  // of the stream to maximize the space available to read new data into.
  // TODO(srbs): A circular buffer would be useful here.
  if (avail_in_ > 0) {
    size_t read_bytes = next_in_ - input_buffer_.get();
    // Remove `read_bytes` from the head of the input stream.
    // Move unread bytes to the head of the input stream.
    if (read_bytes > 0) {
      memmove(input_buffer_.get(), next_in_, avail_in_);
    }

    bytes_to_read -= avail_in_;
    read_location += avail_in_;
  }
  StringPiece data;
  // Try to read enough data to fill up input_buffer_.
  Status s = file_->Read(file_pos_, bytes_to_read, &data, read_location);
  if (data.data() != read_location) {
    memmove(read_location, data.data(), data.size());
  }

  // Since we moved unread data to the head of the input stream we can point
  // next_in to the head of the input stream.
  next_in_ = input_buffer_.get();

  // Note: data.size() could be different from bytes_to_read.
  avail_in_ += data.size();
  file_pos_ += data.size();

  if (!s.ok() && !errors::IsOutOfRange(s)) {
    return s;
  }

  // We throw OutOfRange error iff no new data has been read from file.
  // Since we never check how much data is remaining in the file, it is
  // possible that on the last read there isn't enough data in the file to
  // fill up the buffer in which case file_->ReadNBytes would return an
  // OutOfRange error.
  if (data.empty()) {
    return errors::OutOfRange("EOF reached");
  }
  if (errors::IsOutOfRange(s)) {
    return OkStatus();
  }

  return s;
}

}  // namespace io
}  // namespace tensorflow

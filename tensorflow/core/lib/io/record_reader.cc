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

#include "tensorflow/core/lib/io/record_reader.h"

#include <limits.h>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/platform/env.h"

namespace {
using tensorflow::io::InputStreamInterface;
using tensorflow::RandomAccessFile;
using tensorflow::Status;
using tensorflow::int64;
using tensorflow::string;
using tensorflow::StringPiece;
using tensorflow::io::InputBuffer;

// Wraps a RandomAccessFile in an InputStreamInterface. A given instance of
// BufferedRandomAccessInputStream is NOT safe for concurrent use by multiple
// threads.
class BufferedRandomAccessInputStream : public InputStreamInterface {
 public:
  // Does not take ownership of 'file'
  // 'file' must outlive *this.
  BufferedRandomAccessInputStream(RandomAccessFile* file, int io_bufer_size);

  Status ReadNBytes(int64 bytes_to_read, string* result) override;

  int64 Tell() const override;

  Status Seek(int64 position) { return input_buffer->Seek(position); }

  Status Reset() override { return Seek(0); }

 private:
  RandomAccessFile* file_;  // Not owned.
  int io_bufer_size_;
  bool returned_partial_ = false;
  std::unique_ptr<InputBuffer> input_buffer;
};

BufferedRandomAccessInputStream::BufferedRandomAccessInputStream(
    RandomAccessFile* file, int io_bufer_size)
    : file_(file), io_bufer_size_(io_bufer_size) {
  input_buffer.reset(new InputBuffer(file_, io_bufer_size_));
}

Status BufferedRandomAccessInputStream::ReadNBytes(int64 bytes_to_read,
                                                   string* result) {
  if (bytes_to_read < 0) {
    return tensorflow::errors::InvalidArgument(
        "Cannot read negative number of bytes");
  }
  if (returned_partial_) {
    return tensorflow::errors::OutOfRange("reached end of file");
  }
  result->clear();
  result->resize(bytes_to_read);
  char* result_buffer = &(*result)[0];
  Status s = input_buffer->ReadNBytes(bytes_to_read, result);
  if (!s.ok()) {
    if (tensorflow::errors::IsOutOfRange(s)) returned_partial_ = true;
    return s;
  }

  // If the amount of data we read is less than what we wanted, we return an
  // out of range error. We need to catch this explicitly since file_->Read()
  // would not do so if at least 1 byte is read (b/30839063).
  if (result->size() < bytes_to_read) {
    returned_partial_ = true;
    return tensorflow::errors::OutOfRange("reached end of file");
  }
  return Status::OK();
}

int64 BufferedRandomAccessInputStream::Tell() const {
  LOG(FATAL) << "not implemented";
}
int kIoBufferSize = 2 << 20;
}
namespace tensorflow {
namespace io {

RecordReaderOptions RecordReaderOptions::CreateRecordReaderOptions(
    const string& compression_type) {
  RecordReaderOptions options;
  if (compression_type == "ZLIB") {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::DEFAULT();
#endif  // IS_SLIM_BUILD
  } else if (compression_type == compression::kGzip) {
    options.compression_type = io::RecordReaderOptions::ZLIB_COMPRESSION;
#if defined(IS_SLIM_BUILD)
    LOG(ERROR) << "Compression is not supported but compression_type is set."
               << " No compression will be used.";
#else
    options.zlib_options = io::ZlibCompressionOptions::GZIP();
#endif  // IS_SLIM_BUILD
  } else if (compression_type != compression::kNone) {
    LOG(ERROR) << "Unsupported compression_type:" << compression_type
               << ". No comprression will be used.";
  }
  return options;
}

RecordReader::RecordReader(RandomAccessFile* file,
                           const RecordReaderOptions& options)
    : options_(options) {
  if (options.compression_type == RecordReaderOptions::ZLIB_COMPRESSION) {
// We don't have zlib available on all embedded platforms, so fail.
#if defined(IS_SLIM_BUILD)
    LOG(FATAL) << "Zlib compression is unsupported on mobile platforms.";
#else   // IS_SLIM_BUILD
    underlaying_input_stream_ = new RandomAccessInputStream(file);
    input_stream_ = new ZlibInputStream(
        underlaying_input_stream_, options.zlib_options.input_buffer_size,
        options.zlib_options.output_buffer_size, options.zlib_options);
#endif  // IS_SLIM_BUILD
  } else if (options.compression_type == RecordReaderOptions::NONE) {
    // Nothing to do.
    input_stream_ = new BufferedRandomAccessInputStream(file, kIoBufferSize);
    underlaying_input_stream_ = nullptr;
  } else {
    LOG(FATAL) << "Unspecified compression type :" << options.compression_type;
  }
}

RecordReader::~RecordReader() {
  delete input_stream_;
  delete underlaying_input_stream_;
}

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
// May use *storage as backing store.
Status RecordReader::ReadChecksummed(uint64 offset, size_t n,
                                     StringPiece* result, string* storage) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  const size_t expected = n + sizeof(uint32);
  storage->resize(expected);

  // If we have a zlib compressed buffer, we assume that the
  // file is being read sequentially, and we use the underlying
  // implementation to read the data.
  //
  // No checks are done to validate that the file is being read
  // sequentially.  At some point the zlib input buffer may support
  // seeking, possibly inefficiently.
  TF_RETURN_IF_ERROR(input_stream_->ReadNBytes(expected, storage));

  if (storage->size() != expected) {
    if (storage->size() == 0) {
      return errors::OutOfRange("eof");
    } else {
      return errors::DataLoss("truncated record at ", offset);
    }
  }

  uint32 masked_crc = core::DecodeFixed32(storage->data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(storage->data(), n)) {
    return errors::DataLoss("corrupted record at ", offset);
  }
  *result = StringPiece(storage->data(), n);

  return Status::OK();
}

Status RecordReader::ReadRecord(uint64* offset, string* record) {
  static const size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static const size_t kFooterSize = sizeof(uint32);

  // Read header data.
  StringPiece lbuf;
  Status s = ReadChecksummed(*offset, sizeof(uint64), &lbuf, record);
  if (!s.ok()) {
    return s;
  }
  const uint64 length = core::DecodeFixed64(lbuf.data());

  // Read data
  StringPiece data;
  s = ReadChecksummed(*offset + kHeaderSize, length, &data, record);
  if (!s.ok()) {
    if (errors::IsOutOfRange(s)) {
      s = errors::DataLoss("truncated record at ", *offset);
    }
    return s;
  }

  if (record->data() != data.data()) {
    // RandomAccessFile placed the data in some other location.
    memmove(&(*record)[0], data.data(), data.size());
  }

  record->resize(data.size());

  *offset += kHeaderSize + length + kFooterSize;
  return Status::OK();
}

}  // namespace io
}  // namespace tensorflow

#include "tensorflow/core/lib/io/record_reader.h"

#include <limits.h>
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {
namespace io {

RecordReader::RecordReader(RandomAccessFile* file) : src_(file) {}

RecordReader::~RecordReader() {}

// Read n+4 bytes from file, verify that checksum of first n bytes is
// stored in the last 4 bytes and store the first n bytes in *result.
// May use *storage as backing store.
static Status ReadChecksummed(RandomAccessFile* file, uint64 offset, size_t n,
                              StringPiece* result, string* storage) {
  if (n >= SIZE_MAX - sizeof(uint32)) {
    return errors::DataLoss("record size too large");
  }

  const size_t expected = n + sizeof(uint32);
  storage->resize(expected);
  StringPiece data;
  Status s = file->Read(offset, expected, &data, &(*storage)[0]);
  if (!s.ok()) {
    return s;
  }
  if (data.size() != expected) {
    if (data.size() == 0) {
      return errors::OutOfRange("eof");
    } else {
      return errors::DataLoss("truncated record at ", offset);
    }
  }
  uint32 masked_crc = core::DecodeFixed32(data.data() + n);
  if (crc32c::Unmask(masked_crc) != crc32c::Value(data.data(), n)) {
    return errors::DataLoss("corrupted record at ", offset);
  }
  *result = StringPiece(data.data(), n);
  return Status::OK();
}

Status RecordReader::ReadRecord(uint64* offset, string* record) {
  static const size_t kHeaderSize = sizeof(uint64) + sizeof(uint32);
  static const size_t kFooterSize = sizeof(uint32);

  // Read length
  StringPiece lbuf;
  Status s = ReadChecksummed(src_, *offset, sizeof(uint64), &lbuf, record);
  if (!s.ok()) {
    return s;
  }
  const uint64 length = core::DecodeFixed64(lbuf.data());

  // Read data
  StringPiece data;
  s = ReadChecksummed(src_, *offset + kHeaderSize, length, &data, record);
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

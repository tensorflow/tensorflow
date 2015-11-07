#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace io {

InputBuffer::InputBuffer(RandomAccessFile* file, size_t buffer_bytes)
    : file_(file),
      file_pos_(0),
      size_(buffer_bytes),
      buf_(new char[size_]),
      pos_(buf_),
      limit_(buf_) {}

InputBuffer::~InputBuffer() {
  delete file_;
  delete[] buf_;
}

Status InputBuffer::FillBuffer() {
  StringPiece data;
  Status s = file_->Read(file_pos_, size_, &data, buf_);
  if (data.data() != buf_) {
    memmove(buf_, data.data(), data.size());
  }
  pos_ = buf_;
  limit_ = pos_ + data.size();
  file_pos_ += data.size();
  return s;
}

Status InputBuffer::ReadLine(string* result) {
  result->clear();
  int i;
  Status s;
  for (i = 0;; i++) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    char c = *pos_++;
    if (c == '\n') {
      // We don't append the '\n' to *result
      return Status::OK();
    }
    *result += c;
  }
  if (errors::IsOutOfRange(s) && !result->empty()) {
    return Status::OK();
  }
  return s;
}

Status InputBuffer::ReadNBytes(int64 bytes_to_read, string* result) {
  result->clear();
  if (bytes_to_read < 0) {
    return errors::InvalidArgument("Can't read a negative number of bytes: ",
                                   bytes_to_read);
  }
  result->reserve(bytes_to_read);
  Status s;
  while (result->size() < static_cast<size_t>(bytes_to_read)) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    const int64 bytes_to_copy =
        std::min<int64>(limit_ - pos_, bytes_to_read - result->size());
    result->insert(result->size(), pos_, bytes_to_copy);
    pos_ += bytes_to_copy;
  }
  if (errors::IsOutOfRange(s) &&
      (result->size() == static_cast<size_t>(bytes_to_read))) {
    return Status::OK();
  }
  return s;
}

Status InputBuffer::SkipNBytes(int64 bytes_to_skip) {
  if (bytes_to_skip < 0) {
    return errors::InvalidArgument("Can only skip forward, not ",
                                   bytes_to_skip);
  }
  int64 bytes_skipped = 0;
  Status s;
  while (bytes_skipped < bytes_to_skip) {
    if (pos_ == limit_) {
      // Get more data into buffer
      s = FillBuffer();
      if (limit_ == buf_) {
        break;
      }
    }
    const int64 bytes_to_advance =
        std::min<int64>(limit_ - pos_, bytes_to_skip - bytes_skipped);
    bytes_skipped += bytes_to_advance;
    pos_ += bytes_to_advance;
  }
  if (errors::IsOutOfRange(s) && bytes_skipped == bytes_to_skip) {
    return Status::OK();
  }
  return s;
}

}  // namespace io
}  // namespace tensorflow

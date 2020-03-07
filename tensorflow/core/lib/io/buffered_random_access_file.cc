#include "tensorflow/core/lib/io/buffered_random_access_file.h"

namespace tensorflow {
namespace io {

BufferedRandomAccessFile::BufferedRandomAccessFile(
    std::unique_ptr<RandomAccessFile>&& file, size_t buf_size)
    : buf_size_(buf_size),
      buf_(new char[buf_size_]),
      pos_(0),
      limit_(0),
      file_pos_(0),
      file_(std::move(file)) {}

BufferedRandomAccessFile::~BufferedRandomAccessFile() {
  mutex_lock lock(mu_);
  delete[] buf_;
  buf_ = nullptr;
}

Status BufferedRandomAccessFile::Read(uint64 offset, size_t n,
                                      StringPiece* result,
                                      char* scratch) const {
  mutex_lock lock(mu_);
  Seek(offset);

  Status status;
  size_t total_read = 0;
  while (total_read < n) {
    if (pos_ == limit_) {
      if (n - total_read >= buf_size_) {
        // Unnecessary to read into buffer and then copy to scratch when buffer
        // is empty and byte to read is large enough.
        StringPiece tmp_result;
        status = file_->Read(offset + total_read, n - total_read,
                                   &tmp_result, scratch + total_read);
        // In case the data read is not filled from the start of the scratch
        // buffer (`scratch + total_read` here).
        if (tmp_result.length() > 0 &&
            tmp_result.data() != scratch + total_read) {
          memmove(scratch + total_read, tmp_result.data(), tmp_result.size());
        }
        if (status.ok() ||
            (errors::IsOutOfRange(status) && tmp_result.length() > 0)) {
          total_read += tmp_result.length();
          file_pos_ += tmp_result.length();
          continue;
        } else {
          break;
        }
      }

      status = FillBuffer();
      if (pos_ == limit_) {
        // buffer is still empty after filling, error or EOF.
        break;
      }
    }
    const size_t bytes_to_copy =
        std::min<size_t>(limit_ - pos_, n - total_read);
    memcpy(scratch + total_read, buf_ + pos_, bytes_to_copy);
    total_read += bytes_to_copy;
    pos_ += bytes_to_copy;
    DCHECK(pos_ < limit_);
  }
  *result = StringPiece(scratch, total_read);
  if (errors::IsOutOfRange(status) && total_read == n) {
    // fix the case that FillBuffer results in OutOfRangeError because of large
    // buf_size_ while actually n bytes are read.
    return Status::OK();
  }
  return status;
}

void BufferedRandomAccessFile::Seek(uint64 offset) const {
  CHECK(file_pos_ >= limit_)
      << "file_pos_ should be larger than limit_, but we get: " << file_pos_
      << " and " << limit_;
  const uint64 buf_head_pos_in_file = file_pos_ - static_cast<uint64>(limit_);

  // If buffer is empty, the buffer tail may NOT point to file_pos_
  // because of the skip-buffer-optimization in Read(). So only when buffer is
  // not empty and offset fits into current buffer range, seek is performed
  // within buffer range.
  if (limit_ != pos_ && buf_head_pos_in_file <= offset && offset < file_pos_) {
    // still within buffer range.
    pos_ = offset - buf_head_pos_in_file;
    DCHECK(0 <= pos_ && pos_ < limit_);
  } else {
    pos_ = 0;
    limit_ = 0;
    file_pos_ = offset;
  }
}

Status BufferedRandomAccessFile::FillBuffer() const {
  DCHECK(pos_ == limit_);
  StringPiece result;
  Status status = file_->Read(file_pos_, buf_size_, &result, buf_);
  if (!status.ok() && !errors::IsOutOfRange(status)) {
    return status;
  }

  if (result.length() > 0 && result.data() != buf_) {
    memmove(buf_, result.data(), result.size());
  }

  pos_ = 0;
  limit_ = result.length();
  file_pos_ += result.length();
  return status;
}

}  // namespace io
}  // namespace tensorflow
#ifndef TENSORFLOW_CORE_LIB_IO_BUFFERED_RANDOM_ACCESS_FILE_H_
#define TENSORFLOW_CORE_LIB_IO_BUFFERED_RANDOM_ACCESS_FILE_H_

#include <memory>

#include "tensorflow/core/platform/file_system.h"

#include "tensorflow/core/platform/mutex.h"
namespace tensorflow {
namespace io {

/// \brief A RandomAccessFile with an internal buffer.
class BufferedRandomAccessFile : public RandomAccessFile {
 public:
  /// \brief Create a BufferedRandomAccessFile with given buf_size and an owned
  /// RandomAccessFile.
  BufferedRandomAccessFile(std::unique_ptr<RandomAccessFile>&& other_file,
                           size_t buf_size);

  /// \brief Create a BufferedRandomAccessFile with given buf_size and a
  /// RandomAccessFile which may be owned or not.
  BufferedRandomAccessFile(RandomAccessFile* other_file, size_t buf_size,
                           bool own_file = false);

  ~BufferedRandomAccessFile() override;

  /// \brief Reads up to `n` bytes from the file starting at `offset`.
  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override;

 private:
  /// \brief Fill the internal buffer when it's exhausted.
  Status FillBuffer() const;

  /// \brief Seek to `offset` in file before reading. May invalidate buffer.
  void Seek(uint64 offset) const;

 private:
  mutable mutex mu_;
  const size_t buf_size_;                 // Size of buffer.
  char* buf_ GUARDED_BY(mu_);             // The buffer array.
  mutable size_t pos_ GUARDED_BY(mu_);    // Current reading position in buffer.
  mutable size_t limit_ GUARDED_BY(mu_);  // The end of reading range in buffer,
                                          // not included.  When limit_ == pos_,
                                          // the buffer becomes invalid.
  mutable uint64 file_pos_ GUARDED_BY(mu_);  // Limit_'s offset in other_file_.
  bool own_file_ GUARDED_BY(mu_);            // Whether other_file_ is owned.
  tensorflow::RandomAccessFile* other_file_
      GUARDED_BY(mu_);  // The wrapped file.

  TF_DISALLOW_COPY_AND_ASSIGN(BufferedRandomAccessFile);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_IO_BUFFERED_RANDOM_ACCESS_FILE_H_
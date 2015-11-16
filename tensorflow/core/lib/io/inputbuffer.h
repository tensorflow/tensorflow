#ifndef TENSORFLOW_LIB_IO_INPUTBUFFER_H_
#define TENSORFLOW_LIB_IO_INPUTBUFFER_H_

#include <string>
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
namespace io {

// An InputBuffer provides a buffer on top of a RandomAccessFile.
// A given instance of an InputBuffer is NOT safe for concurrent use
// by multiple threads
class InputBuffer {
 public:
  // Create an InputBuffer for "file" with a buffer size of
  // "buffer_bytes" bytes.  Takes ownership of "file" and will
  // delete it when the InputBuffer is destroyed.
  InputBuffer(RandomAccessFile* file, size_t buffer_bytes);
  ~InputBuffer();

  // Read one text line of data into "*result" until end-of-file or a
  // \n is read.  (The \n is not included in the result.)  Overwrites
  // any existing data in *result.
  //
  // If successful, returns OK.  If we are already at the end of the
  // file, we return an OUT_OF_RANGE error.  Otherwise, we return
  // some other non-OK status.
  Status ReadLine(string* result);

  // Reads bytes_to_read bytes into *result, overwriting *result.
  //
  // If successful, returns OK.  If we there are not enough bytes to
  // read before the end of the file, we return an OUT_OF_RANGE error.
  // Otherwise, we return some other non-OK status.
  Status ReadNBytes(int64 bytes_to_read, string* result);

  // Like ReadNBytes() without returning the bytes read.
  Status SkipNBytes(int64 bytes_to_skip);

  // Returns the position in the file.
  int64 Tell() const { return file_pos_ - (limit_ - pos_); }

 private:
  Status FillBuffer();

  RandomAccessFile* file_;  // Owned
  int64 file_pos_;          // Next position to read from in "file_"
  size_t size_;             // Size of "buf_"
  char* buf_;               // The buffer itself
  // [pos_,limit_) hold the "limit_ - pos_" bytes just before "file_pos_"
  char* pos_;    // Current position in "buf"
  char* limit_;  // Just past end of valid data in "buf"

  TF_DISALLOW_COPY_AND_ASSIGN(InputBuffer);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_INPUTBUFFER_H_

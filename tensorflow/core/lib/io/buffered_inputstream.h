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

#ifndef TENSORFLOW_LIB_IO_BUFFERED_INPUTSTREAM_H_
#define TENSORFLOW_LIB_IO_BUFFERED_INPUTSTREAM_H_

#include "tensorflow/core/lib/io/inputstream_interface.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {
namespace io {

// Provides a buffer on top of an InputStreamInterface. A single instance of
// BufferedInputStream is NOT safe for concurrent use by multiple threads.
class BufferedInputStream : public InputStreamInterface {
 public:
  // Does not take ownership of input_stream unless owns_input_stream is set
  // to true. input_stream must outlive *this then.
  // TODO(rohanj): Remove owns_input_stream once the constructor below is
  // removed.
  BufferedInputStream(InputStreamInterface* input_stream, size_t buffer_bytes,
                      bool owns_input_stream = false);

  // For backwards compatibility, expose an interface that is similar to what
  // InputBuffer exposes. Does not take ownership of file. file must outlive
  // *this. This will be removed once we migrate all uses of this class to the
  // constructor above.
  BufferedInputStream(RandomAccessFile* file, size_t buffer_bytes);

  ~BufferedInputStream() override;

  Status ReadNBytes(int64 bytes_to_read, string* result) override;

  Status SkipNBytes(int64 bytes_to_skip) override;

  int64 Tell() const override;

  // Seek to this offset within the file.
  //
  // If we seek to somewhere within our pre-buffered data, we will re-use what
  // data we can.  Otherwise, Seek() throws out the current buffer and the next
  // read will trigger an underlying read.
  //
  // Note: When seeking backwards in a stream, this implementation uses
  // Reset() + SkipNBytes(), so its performance will be dependent
  // largely on the performance of SkipNBytes().
  Status Seek(int64 position);

  // Read one text line of data into "*result" until end-of-file or a
  // \n is read.  (The \n is not included in the result.)  Overwrites
  // any existing data in *result.
  //
  // If successful, returns OK.  If we are already at the end of the
  // file, we return an OUT_OF_RANGE error.  Otherwise, we return
  // some other non-OK status.
  Status ReadLine(string* result);

  // Returns one text line of data until end-of-file or a '\n' is read. The '\n'
  // is included in the result.
  // This method is a substitute for ReadLine() when called from Python which is
  // the expectation in the python File::readline() API.
  // Also, '\0's are treated like any other character within the line and given
  // no special treatment.
  string ReadLineAsString();

  // Reads the entire contents of the file into *result.
  //
  // Note: the amount of memory used by this function call is unbounded, so only
  // use in ops that expect that behavior.
  Status ReadAll(string* result);

  Status Reset() override;

 private:
  Status FillBuffer();
  Status ReadLineHelper(string* result, bool include_eol);

  InputStreamInterface* input_stream_;  // not owned.
  size_t size_;                         // buffer size.
  string buf_;                          // the buffer itself.
  // buf_[pos_, limit_) holds the valid "read ahead" data in the file.
  size_t pos_ = 0;    // current position in buf_.
  size_t limit_ = 0;  // just past the end of valid data in buf_.
  bool owns_input_stream_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(BufferedInputStream);
};

}  // namespace io
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_LIB_IO_BUFFERED_INPUTSTREAM_H_

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

#ifndef XLA_TSL_LIB_IO_BUFFERED_INPUTSTREAM_H_
#define XLA_TSL_LIB_IO_BUFFERED_INPUTSTREAM_H_

#include <string>

#include "xla/tsl/lib/io/inputstream_interface.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {
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

  absl::Status ReadNBytes(int64_t bytes_to_read, tstring* result) override;

  absl::Status SkipNBytes(int64_t bytes_to_skip) override;

  int64_t Tell() const override;

  // Seek to this offset within the file.
  //
  // If we seek to somewhere within our pre-buffered data, we will re-use what
  // data we can.  Otherwise, Seek() throws out the current buffer and the next
  // read will trigger an underlying read.
  //
  // Note: When seeking backwards in a stream, this implementation uses
  // Reset() + SkipNBytes(), so its performance will be dependent
  // largely on the performance of SkipNBytes().
  absl::Status Seek(int64_t position);

  // Read one text line of data into "*result" until end-of-file or a
  // \n is read.  (The \n is not included in the result.)  Overwrites
  // any existing data in *result.
  //
  // If successful, returns OK.  If we are already at the end of the
  // file, we return an OUT_OF_RANGE error.  Otherwise, we return
  // some other non-OK status.
  absl::Status ReadLine(std::string* result);
  absl::Status ReadLine(tstring* result);

  // Returns one text line of data until end-of-file or a '\n' is read. The '\n'
  // is included in the result.
  // This method is a substitute for ReadLine() when called from Python which is
  // the expectation in the python File::readline() API.
  // Also, '\0's are treated like any other character within the line and given
  // no special treatment.
  std::string ReadLineAsString();

  // Skip one text line of data.
  //
  // If successful, returns OK.  If we are already at the end of the
  // file, we return an OUT_OF_RANGE error.  Otherwise, we return
  // some other non-OK status.
  absl::Status SkipLine();

  // Reads the entire contents of the file into *result.
  //
  // Note: the amount of memory used by this function call is unbounded, so only
  // use in ops that expect that behavior.
  template <typename T>
  absl::Status ReadAll(T* result);

  absl::Status Reset() override;

 private:
  absl::Status FillBuffer();
  template <typename StringType>
  absl::Status ReadLineHelper(StringType* result, bool include_eol);

  InputStreamInterface* input_stream_;  // not owned.
  size_t size_;                         // buffer size.
  tstring buf_;                         // the buffer itself.
  // buf_[pos_, limit_) holds the valid "read ahead" data in the file.
  size_t pos_ = 0;    // current position in buf_.
  size_t limit_ = 0;  // just past the end of valid data in buf_.
  bool owns_input_stream_ = false;
  // When EoF is reached, file_status_ contains the status to skip unnecessary
  // buffer allocations.
  absl::Status file_status_ = absl::OkStatus();

  BufferedInputStream(const BufferedInputStream&) = delete;
  void operator=(const BufferedInputStream&) = delete;
};

// Explicit instantiations defined in buffered_inputstream.cc.
#ifndef SWIG
extern template Status BufferedInputStream::ReadAll<std::string>(
    std::string* result);
extern template Status BufferedInputStream::ReadAll<tstring>(tstring* result);
#endif  // SWIG

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_BUFFERED_INPUTSTREAM_H_

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

#ifndef TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_READER_H_
#define TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_READER_H_

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

class RecordReader;

// A wrapper around io::RecordReader that is more easily SWIG wrapped for
// Python.  An instance of this class is not safe for concurrent access
// by multiple threads.
class PyRecordReader {
 public:
  // TODO(vrv): make this take a shared proto to configure
  // the compression options.
  static PyRecordReader* New(const string& filename, uint64 start_offset,
                             const string& compression_type_string,
                             TF_Status* out_status);

  ~PyRecordReader();

  // Attempt to get the next record at "current_offset()". Populates status
  // with OK on success, OUT_OF_RANGE for end of file, DATA_LOSS for some
  // kinds of truncated reads, or another code for other errors
  // (e.g., filesystem errors).
  void GetNext(TF_Status* status);

  // Return the current record contents.  Only valid after the preceding call
  // to GetNext() returned true
  string record() const { return record_; }
  // Return the current offset in the file.
  uint64 offset() const { return offset_; }

  // Close the underlying file and release its resources.
  void Close();

 private:
  PyRecordReader();

  uint64 offset_;
  RandomAccessFile* file_;    // Owned
  io::RecordReader* reader_;  // Owned
  string record_;
  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordReader);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_READER_H_

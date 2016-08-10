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

#ifndef TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_
#define TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class WritableFile;

namespace io {

class RecordWriter;

// A wrapper around io::RecordWriter that is more easily SWIG wrapped for
// Python.  An instance of this class is not safe for concurrent access
// by multiple threads.
class PyRecordWriter {
 public:
  // TODO(vrv): make this take a shared proto to configure
  // the compression options.
  static PyRecordWriter* New(const string& filename,
                             const string& compression_type_string);
  ~PyRecordWriter();

  bool WriteRecord(tensorflow::StringPiece record);
  void Close();

 private:
  PyRecordWriter();

  WritableFile* file_;        // Owned
  io::RecordWriter* writer_;  // Owned
  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordWriter);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_

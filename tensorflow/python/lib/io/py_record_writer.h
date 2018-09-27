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

#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_writer.h"
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
  static PyRecordWriter* New(const string& filename,
                             const io::RecordWriterOptions& compression_options,
                             TF_Status* out_status);
  ~PyRecordWriter();

  void WriteRecord(tensorflow::StringPiece record, TF_Status* out_status);
  void Flush(TF_Status* out_status);
  void Close(TF_Status* out_status);

 private:
  PyRecordWriter();

  std::unique_ptr<io::RecordWriter> writer_;
  std::unique_ptr<WritableFile> file_;
  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordWriter);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_

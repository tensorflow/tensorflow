#ifndef THIRD_PARTY_TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_
#define THIRD_PARTY_TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class WritableFile;

namespace io {

class RecordWriter;

// A wrapper around io::RecordWriter that is more easily SWIG wrapped for
// Python.  An instance of this class is not safe for concurrent access
// by multiple threads.
class PyRecordWriter {
 public:
  static PyRecordWriter* New(const string& filename);
  ~PyRecordWriter();

  bool WriteRecord(::tensorflow::StringPiece record);
  void Close();

 private:
  PyRecordWriter();

  WritableFile* file_;        // Owned
  io::RecordWriter* writer_;  // Owned
  TF_DISALLOW_COPY_AND_ASSIGN(PyRecordWriter);
};

}  // namespace io
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_WRITER_H_

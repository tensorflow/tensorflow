#ifndef TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_READER_H_
#define TENSORFLOW_PYTHON_LIB_IO_PY_RECORD_READER_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

class RecordReader;

// A wrapper around io::RecordReader that is more easily SWIG wrapped for
// Python.  An instance of this class is not safe for concurrent access
// by multiple threads.
class PyRecordReader {
 public:
  static PyRecordReader* New(const string& filename, uint64 start_offset);
  ~PyRecordReader();

  // Attempt to get the next record at "current_offset()".  If
  // successful, returns true, and the record contents can be retrieve
  // with "this->record()".  Otherwise, returns false.
  bool GetNext();
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

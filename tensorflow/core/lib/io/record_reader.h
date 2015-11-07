#ifndef TENSORFLOW_LIB_IO_RECORD_READER_H_
#define TENSORFLOW_LIB_IO_RECORD_READER_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

class RecordReader {
 public:
  // Create a reader that will return log records from "*file".
  // "*file" must remain live while this Reader is in use.
  explicit RecordReader(RandomAccessFile* file);

  ~RecordReader();

  // Read the record at "*offset" into *record and update *offset to
  // point to the offset of the next record.  Returns OK on success,
  // OUT_OF_RANGE for end of file, or something else for an error.
  Status ReadRecord(uint64* offset, string* record);

 private:
  RandomAccessFile* src_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecordReader);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_RECORD_READER_H_

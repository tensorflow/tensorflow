#ifndef TENSORFLOW_LIB_IO_RECORD_WRITER_H_
#define TENSORFLOW_LIB_IO_RECORD_WRITER_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class WritableFile;

namespace io {

class RecordWriter {
 public:
  // Create a writer that will append data to "*dest".
  // "*dest" must be initially empty.
  // "*dest" must remain live while this Writer is in use.
  explicit RecordWriter(WritableFile* dest);

  ~RecordWriter();

  Status WriteRecord(StringPiece slice);

 private:
  WritableFile* const dest_;

  TF_DISALLOW_COPY_AND_ASSIGN(RecordWriter);
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_RECORD_WRITER_H_

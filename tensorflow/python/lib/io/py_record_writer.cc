#include "tensorflow/python/lib/io/py_record_writer.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {
namespace io {

PyRecordWriter::PyRecordWriter() {}

PyRecordWriter* PyRecordWriter::New(const string& filename) {
  WritableFile* file;
  Status s = Env::Default()->NewWritableFile(filename, &file);
  if (!s.ok()) {
    return nullptr;
  }
  PyRecordWriter* writer = new PyRecordWriter;
  writer->file_ = file;
  writer->writer_ = new RecordWriter(writer->file_);
  return writer;
}

PyRecordWriter::~PyRecordWriter() {
  delete writer_;
  delete file_;
}

bool PyRecordWriter::WriteRecord(::tensorflow::StringPiece record) {
  if (writer_ == nullptr) return false;
  Status s = writer_->WriteRecord(record);
  return s.ok();
}

void PyRecordWriter::Close() {
  delete writer_;
  delete file_;
  writer_ = nullptr;
  file_ = nullptr;
}

}  // namespace io
}  // namespace tensorflow

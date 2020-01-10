#include "tensorflow/python/lib/io/py_record_reader.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/env.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

PyRecordReader::PyRecordReader() {}

PyRecordReader* PyRecordReader::New(const string& filename,
                                    uint64 start_offset) {
  RandomAccessFile* file;
  Status s = Env::Default()->NewRandomAccessFile(filename, &file);
  if (!s.ok()) {
    return nullptr;
  }
  PyRecordReader* reader = new PyRecordReader;
  reader->offset_ = start_offset;
  reader->file_ = file;
  reader->reader_ = new RecordReader(reader->file_);
  return reader;
}

PyRecordReader::~PyRecordReader() {
  delete reader_;
  delete file_;
}

bool PyRecordReader::GetNext() {
  if (reader_ == nullptr) return false;
  Status s = reader_->ReadRecord(&offset_, &record_);
  return s.ok();
}

void PyRecordReader::Close() {
  delete reader_;
  delete file_;
  file_ = nullptr;
  reader_ = nullptr;
}

}  // namespace io
}  // namespace tensorflow

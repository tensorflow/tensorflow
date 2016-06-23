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

#include "tensorflow/python/lib/io/py_record_writer.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

PyRecordWriter::PyRecordWriter() {}

PyRecordWriter* PyRecordWriter::New(const string& filename,
                                    const string& compression_type_string) {
  std::unique_ptr<WritableFile> file;
  Status s = Env::Default()->NewWritableFile(filename, &file);
  if (!s.ok()) {
    return nullptr;
  }
  PyRecordWriter* writer = new PyRecordWriter;
  writer->file_ = file.release();

  RecordWriterOptions options;
  if (compression_type_string == "ZLIB") {
    options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  }
  writer->writer_ = new RecordWriter(writer->file_, options);
  return writer;
}

PyRecordWriter::~PyRecordWriter() {
  delete writer_;
  delete file_;
}

bool PyRecordWriter::WriteRecord(tensorflow::StringPiece record) {
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

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

#include "tensorflow/python/lib/io/py_record_reader.h"

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;

namespace io {

PyRecordReader::PyRecordReader() {}

// NOTE(sethtroisi): At this time PyRecordReader doesn't benefit from taking
// RecordReaderOptions, if this changes the API can be updated at that time.
PyRecordReader* PyRecordReader::New(const string& filename, uint64 start_offset,
                                    const string& compression_type_string,
                                    TF_Status* out_status) {
  std::unique_ptr<RandomAccessFile> file;
  Status s = Env::Default()->NewRandomAccessFile(filename, &file);
  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return nullptr;
  }
  uint64 file_size;
  // for some cloud file system (gcs). For empty files, we can still get error
  // messages form read, we use file size to hijack the actual reader.
  Status s1 = Env::Default()->GetFileSize(filename, &file_size);
  if (!s1.ok()) {
    Set_TF_Status_from_Status(out_status, s1);
    return nullptr;
  } else {
    if (file_size == 0) {
      Set_TF_Status_from_Status(out_status, errors::OutOfRange("eof"));
      return nullptr;
    }
  }
  PyRecordReader* reader = new PyRecordReader;
  reader->offset_ = start_offset;
  reader->file_ = file.release();

  static const uint64 kReaderBufferSize = 16 * 1024 * 1024;
  RecordReaderOptions options =
      RecordReaderOptions::CreateRecordReaderOptions(compression_type_string);
  options.buffer_size = kReaderBufferSize;
  reader->reader_ = new RecordReader(reader->file_, options);
  return reader;
}

PyRecordReader::~PyRecordReader() {
  delete reader_;
  delete file_;
}

void PyRecordReader::GetNext(TF_Status* status) {
  if (reader_ == nullptr) {
    Set_TF_Status_from_Status(status,
                              errors::FailedPrecondition("Reader is closed."));
    return;
  }
  Status s = reader_->ReadRecord(&offset_, &record_);
  Set_TF_Status_from_Status(status, s);
}

void PyRecordReader::Close() {
  delete reader_;
  delete file_;
  file_ = nullptr;
  reader_ = nullptr;
}

}  // namespace io
}  // namespace tensorflow

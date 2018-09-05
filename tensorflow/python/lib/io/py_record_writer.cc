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

#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/zlib_compression_options.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace io {

PyRecordWriter::PyRecordWriter() {}

PyRecordWriter* PyRecordWriter::New(const string& filename,
                                    const string& compression_type_string,
                                    TF_Status* out_status) {
  std::unique_ptr<WritableFile> file;
  Status s = Env::Default()->NewWritableFile(filename, &file);
  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return nullptr;
  }
  PyRecordWriter* writer = new PyRecordWriter;
  writer->file_ = std::move(file);

  RecordWriterOptions options =
      RecordWriterOptions::CreateRecordWriterOptions(compression_type_string);

  writer->writer_.reset(new RecordWriter(writer->file_.get(), options));
  return writer;
}

PyRecordWriter::~PyRecordWriter() {
  // Writer depends on file during close for zlib flush, so destruct first.
  writer_.reset();
  file_.reset();
}

void PyRecordWriter::WriteRecord(tensorflow::StringPiece record,
                                 TF_Status* out_status) {
  if (writer_ == nullptr) {
    TF_SetStatus(out_status, TF_FAILED_PRECONDITION,
                 "Writer not initialized or previously closed");
    return;
  }
  Status s = writer_->WriteRecord(record);
  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
  }
}

void PyRecordWriter::Flush(TF_Status* out_status) {
  if (writer_ == nullptr) {
    TF_SetStatus(out_status, TF_FAILED_PRECONDITION,
                 "Writer not initialized or previously closed");
    return;
  }
  Status s = writer_->Flush();
  if (!s.ok()) {
    Set_TF_Status_from_Status(out_status, s);
    return;
  }
}

void PyRecordWriter::Close(TF_Status* out_status) {
  if (writer_ != nullptr) {
    Status s = writer_->Close();
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    writer_.reset(nullptr);
  }
  if (file_ != nullptr) {
    Status s = file_->Close();
    if (!s.ok()) {
      Set_TF_Status_from_Status(out_status, s);
      return;
    }
    file_.reset(nullptr);
  }
}

}  // namespace io
}  // namespace tensorflow

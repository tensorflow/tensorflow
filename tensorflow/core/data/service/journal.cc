/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/data/service/journal.h"

#include "absl/memory/memory.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace data {

namespace {
constexpr StringPiece kJournal = "journal";
}  // namespace

std::string DataServiceJournalFile(StringPiece journal_dir) {
  return io::JoinPath(journal_dir, kJournal);
}

FileJournalWriter::FileJournalWriter(Env* env, StringPiece journal_dir)
    : env_(env), journal_dir_(journal_dir) {}

Status FileJournalWriter::EnsureInitialized() {
  if (writer_) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(journal_dir_));
  TF_RETURN_IF_ERROR(
      env_->NewAppendableFile(DataServiceJournalFile(journal_dir_), &file_));
  writer_ = absl::make_unique<io::RecordWriter>(file_.get());
  return Status::OK();
}

Status FileJournalWriter::Write(Update update) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  std::string s = update.SerializeAsString();
  if (s.empty()) {
    return errors::Internal("Failed to serialize update ", update.DebugString(),
                            " to string");
  }
  TF_RETURN_IF_ERROR(writer_->WriteRecord(s));
  TF_RETURN_IF_ERROR(writer_->Flush());
  TF_RETURN_IF_ERROR(file_->Sync());
  return Status::OK();
}

FileJournalReader::FileJournalReader(Env* env, StringPiece journal_dir)
    : env_(env), journal_dir_(journal_dir) {}

Status FileJournalReader::EnsureInitialized() {
  if (reader_) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      env_->NewRandomAccessFile(DataServiceJournalFile(journal_dir_), &file_));
  reader_ = absl::make_unique<io::RecordReader>(file_.get());
  return Status::OK();
}

Status FileJournalReader::Read(Update* update, bool* end_of_journal) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  tstring record;
  Status s = reader_->ReadRecord(&offset_, &record);
  if (errors::IsOutOfRange(s)) {
    *end_of_journal = true;
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(s);
  if (!update->ParseFromString(record)) {
    return errors::DataLoss("Failed to parse journal record.");
  }
  *end_of_journal = false;
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow

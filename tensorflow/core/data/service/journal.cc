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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace data {

namespace {
constexpr absl::string_view kJournal = "journal";

absl::Status ParseSequenceNumber(const std::string& journal_file,
                                 int64_t* sequence_number) {
  if (!RE2::FullMatch(journal_file, ".*_(\\d+)", sequence_number)) {
    return errors::InvalidArgument("Failed to parse journal file name: ",
                                   journal_file);
  }
  return absl::OkStatus();
}
}  // namespace

std::string DataServiceJournalFile(const std::string& journal_dir,
                                   int64_t sequence_number) {
  return io::JoinPath(journal_dir,
                      absl::StrCat(kJournal, "_", sequence_number));
}

FileJournalWriter::FileJournalWriter(Env* env, const std::string& journal_dir)
    : env_(env), journal_dir_(journal_dir) {}

absl::Status FileJournalWriter::EnsureInitialized() {
  if (writer_) {
    return absl::OkStatus();
  }
  std::vector<std::string> journal_files;
  TF_RETURN_IF_ERROR(env_->RecursivelyCreateDir(journal_dir_));
  TF_RETURN_IF_ERROR(env_->GetChildren(journal_dir_, &journal_files));
  int64_t latest_sequence_number = -1;
  for (const auto& file : journal_files) {
    int64_t sequence_number;
    TF_RETURN_IF_ERROR(ParseSequenceNumber(file, &sequence_number));
    latest_sequence_number = std::max(latest_sequence_number, sequence_number);
  }
  std::string journal_file =
      DataServiceJournalFile(journal_dir_, latest_sequence_number + 1);
  TF_RETURN_IF_ERROR(env_->NewAppendableFile(journal_file, &file_));
  writer_ = std::make_unique<io::RecordWriter>(file_.get());
  VLOG(1) << "Created journal writer to write to " << journal_file;
  return absl::OkStatus();
}

absl::Status FileJournalWriter::Write(const Update& update) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  std::string s = update.SerializeAsString();
  if (s.empty()) {
    return errors::Internal("Failed to serialize update ", update.DebugString(),
                            " to string");
  }
  TF_RETURN_IF_ERROR(writer_->WriteRecord(s));
  TF_RETURN_IF_ERROR(writer_->Flush());
  TF_RETURN_IF_ERROR(file_->Sync());
  if (VLOG_IS_ON(4)) {
    VLOG(4) << "Wrote journal entry: " << update.DebugString();
  }
  return absl::OkStatus();
}

FileJournalReader::FileJournalReader(Env* env, absl::string_view journal_dir)
    : env_(env), journal_dir_(journal_dir) {}

absl::Status FileJournalReader::EnsureInitialized() {
  if (reader_) {
    return absl::OkStatus();
  }
  return UpdateFile(DataServiceJournalFile(journal_dir_, 0));
}

absl::Status FileJournalReader::Read(Update& update, bool& end_of_journal) {
  TF_RETURN_IF_ERROR(EnsureInitialized());
  while (true) {
    tstring record;
    absl::Status s = reader_->ReadRecord(&record);
    if (absl::IsOutOfRange(s)) {
      sequence_number_++;
      std::string next_journal_file =
          DataServiceJournalFile(journal_dir_, sequence_number_);
      if (absl::IsNotFound(env_->FileExists(next_journal_file))) {
        VLOG(3) << "Next journal file " << next_journal_file
                << " does not exist. End of journal reached.";
        end_of_journal = true;
        return absl::OkStatus();
      }
      TF_RETURN_IF_ERROR(UpdateFile(next_journal_file));
      continue;
    }
    TF_RETURN_IF_ERROR(s);
    if (!update.ParseFromString(record)) {
      return errors::DataLoss("Failed to parse journal record.");
    }
    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Read journal entry: " << update.DebugString();
    }
    end_of_journal = false;
    return absl::OkStatus();
  }
}

absl::Status FileJournalReader::UpdateFile(const std::string& filename) {
  VLOG(1) << "Reading from journal file " << filename;
  TF_RETURN_IF_ERROR(env_->NewRandomAccessFile(filename, &file_));
  io::RecordReaderOptions opts;
  opts.buffer_size = 2 << 20;  // 2MB
  reader_ = std::make_unique<io::SequentialRecordReader>(file_.get(), opts);
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow

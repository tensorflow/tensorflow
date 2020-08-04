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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_JOURNAL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_JOURNAL_H_

#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace data {

// Returns the location of the journal file within the journal directory.
std::string DataServiceJournalFile(StringPiece journal_dir);

// Interface for writing to a journal.
class JournalWriter {
 public:
  virtual ~JournalWriter() = default;
  // Writes and syncs an update to the journal.
  virtual Status Write(Update update) = 0;
};

// FileJournalWriter is not thread-safe, requiring external synchronization when
// used by multiple threads.
class FileJournalWriter : public JournalWriter {
 public:
  // Creates a journal writer to write to the given journal directory.
  // If there is already journal data there, the journal writer will append to
  // the existing journal.
  explicit FileJournalWriter(Env* env, StringPiece journal_dir);
  FileJournalWriter(const FileJournalWriter&) = delete;
  FileJournalWriter& operator=(const FileJournalWriter&) = delete;

  Status Write(Update update) override;

 private:
  // Initializes the writer if it is not yet initialized.
  Status EnsureInitialized();

  Env* env_;
  const std::string journal_dir_;
  std::unique_ptr<WritableFile> file_;
  std::unique_ptr<io::RecordWriter> writer_;
};

// NoopJournalWriter implements the JournalWriter interface, but doesn't
// actually write journal entries anywhere.
class NoopJournalWriter : public JournalWriter {
 public:
  // Creates a journal writer which does nothing.
  explicit NoopJournalWriter();
  NoopJournalWriter(const NoopJournalWriter&) = delete;
  NoopJournalWriter& operator=(const NoopJournalWriter&) = delete;

  Status Write(Update update) override;
};

// Interface for reading from a journal.
class JournalReader {
 public:
  virtual ~JournalReader() = default;
  // Reads the next update from the journal. Sets `*end_of_journal=true` if
  // there are no more updates left in the journal.
  virtual Status Read(Update* update, bool* end_of_journal) = 0;
};

// JournalReader is not thread-safe, requiring external synchronization when
// used by multiple threads.
class FileJournalReader : public JournalReader {
 public:
  explicit FileJournalReader(Env* env, StringPiece journal_dir);
  FileJournalReader(const FileJournalReader&) = delete;
  FileJournalReader& operator=(const FileJournalReader&) = delete;

  Status Read(Update* update, bool* end_of_journal) override;

 private:
  // Initializes the reader if it is not yet initialized.
  Status EnsureInitialized();

  Env* env_;
  const std::string journal_dir_;
  // Current offset into `file_`.
  uint64 offset_ = 0;
  std::unique_ptr<RandomAccessFile> file_;
  std::unique_ptr<io::RecordReader> reader_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_JOURNAL_H_

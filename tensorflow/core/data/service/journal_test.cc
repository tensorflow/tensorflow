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

#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/data_service.pb.h"

namespace tensorflow {
namespace data {

namespace {
using ::testing::HasSubstr;

bool NewJournalDir(std::string& journal_dir) {
  std::string filename = testing::TmpDir();
  if (!Env::Default()->CreateUniqueFileName(&filename, "journal_dir")) {
    return false;
  }
  journal_dir = filename;
  return true;
}

Update MakeCreateIterationUpdate() {
  Update update;
  CreateIterationUpdate* create_iteration = update.mutable_create_iteration();
  create_iteration->set_job_id(3);
  create_iteration->set_iteration_id(8);
  create_iteration->set_repetition(5);
  return update;
}

Update MakeFinishTaskUpdate() {
  Update update;
  FinishTaskUpdate* finish_task = update.mutable_finish_task();
  finish_task->set_task_id(8);
  return update;
}

Update MakeRegisterDatasetUpdate() {
  Update update;
  RegisterDatasetUpdate* register_dataset = update.mutable_register_dataset();
  register_dataset->set_dataset_id("dataset_id");
  register_dataset->set_fingerprint(3);
  return update;
}

Status CheckJournalContent(StringPiece journal_dir,
                           const std::vector<Update>& expected) {
  FileJournalReader reader(Env::Default(), journal_dir);
  for (const auto& update : expected) {
    Update result;
    bool end_of_journal = true;
    TF_RETURN_IF_ERROR(reader.Read(result, end_of_journal));
    EXPECT_FALSE(end_of_journal);
    // We can't use the testing::EqualsProto matcher because it is not available
    // in OSS.
    EXPECT_EQ(result.SerializeAsString(), update.SerializeAsString());
  }
  Update result;
  bool end_of_journal = false;
  TF_RETURN_IF_ERROR(reader.Read(result, end_of_journal));
  EXPECT_TRUE(end_of_journal);
  return absl::OkStatus();
}
}  // namespace

TEST(Journal, RoundTripMultiple) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));
  std::vector<Update> updates = {MakeCreateIterationUpdate(),
                                 MakeRegisterDatasetUpdate(),
                                 MakeFinishTaskUpdate()};
  FileJournalWriter writer(Env::Default(), journal_dir);
  for (const auto& update : updates) {
    TF_EXPECT_OK(writer.Write(update));
  }

  TF_EXPECT_OK(CheckJournalContent(journal_dir, updates));
}

TEST(Journal, AppendExistingJournal) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));
  std::vector<Update> updates = {MakeCreateIterationUpdate(),
                                 MakeRegisterDatasetUpdate(),
                                 MakeFinishTaskUpdate()};
  for (const auto& update : updates) {
    FileJournalWriter writer(Env::Default(), journal_dir);
    TF_EXPECT_OK(writer.Write(update));
  }

  TF_EXPECT_OK(CheckJournalContent(journal_dir, updates));
}

TEST(Journal, MissingFile) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));
  FileJournalReader reader(Env::Default(), journal_dir);
  Update result;
  bool end_of_journal = true;
  Status s = reader.Read(result, end_of_journal);
  EXPECT_TRUE(absl::IsNotFound(s));
}

TEST(Journal, NonRecordData) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));

  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(journal_dir));
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewAppendableFile(
        DataServiceJournalFile(journal_dir, /*sequence_number=*/0), &file));
    TF_ASSERT_OK(file->Append("not record data"));
  }

  FileJournalReader reader(Env::Default(), journal_dir);
  Update result;
  bool end_of_journal = true;
  Status s = reader.Read(result, end_of_journal);
  EXPECT_THAT(s.message(), HasSubstr("corrupted record"));
  EXPECT_EQ(s.code(), error::DATA_LOSS);
}

TEST(Journal, InvalidRecordData) {
  std::string journal_dir;
  EXPECT_TRUE(NewJournalDir(journal_dir));

  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(journal_dir));
  {
    std::unique_ptr<WritableFile> file;
    TF_ASSERT_OK(Env::Default()->NewAppendableFile(
        DataServiceJournalFile(journal_dir, /*sequence_number=*/0), &file));
    auto writer = std::make_unique<io::RecordWriter>(file.get());
    TF_ASSERT_OK(writer->WriteRecord("not serialized proto"));
  }

  FileJournalReader reader(Env::Default(), journal_dir);
  Update result;
  bool end_of_journal = true;
  Status s = reader.Read(result, end_of_journal);
  EXPECT_THAT(s.message(), HasSubstr("Failed to parse journal record"));
  EXPECT_EQ(s.code(), error::DATA_LOSS);
}
}  // namespace data
}  // namespace tensorflow

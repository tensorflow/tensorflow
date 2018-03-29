/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <iostream>
#include <vector>

#include "tensorflow/contrib/tensorboard/db/schema.h"
#include "tensorflow/contrib/tensorboard/db/summary_db_writer.h"
#include "tensorflow/core/lib/db/sqlite.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

template <typename T>
string AddCommas(T n) {
  static_assert(std::is_integral<T>::value, "is_integral");
  string s = strings::StrCat(n);
  if (s.size() > 3) {
    int extra = s.size() / 3 - (s.size() % 3 == 0 ? 1 : 0);
    s.append(extra, 'X');
    int c = 0;
    for (int i = s.size() - 1; i > 0; --i) {
      s[i] = s[i - extra];
      if (++c % 3 == 0) {
        s[--i] = ',';
        --extra;
      }
    }
  }
  return s;
}

int main(int argc, char* argv[]) {
  string path;
  string events;
  string experiment_name;
  string run_name;
  string user_name;
  std::vector<Flag> flag_list = {
      Flag("db", &path, "Path of SQLite DB file"),
      Flag("events", &events, "TensorFlow record proto event log file"),
      Flag("experiment_name", &experiment_name, "The DB experiment_name value"),
      Flag("run_name", &run_name, "The DB run_name value"),
      Flag("user_name", &user_name, "The DB user_name value"),
  };
  string usage = Flags::Usage(argv[0], flag_list);
  bool parse_result = Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || path.empty()) {
    std::cerr << "The loader tool imports tf.Event record files, created by\n"
              << "SummaryFileWriter, into the sorts of SQLite database files\n"
              << "created by SummaryDbWriter.\n\n"
              << "In addition to the flags below, the environment variables\n"
              << "defined by core/lib/db/sqlite.cc can also be set.\n\n"
              << usage;
    return -1;
  }
  port::InitMain(argv[0], &argc, &argv);
  Env* env = Env::Default();

  LOG(INFO) << "Opening SQLite file: " << path;
  Sqlite* db;
  TF_CHECK_OK(Sqlite::Open(
      path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      &db));
  core::ScopedUnref unref_db(db);

  LOG(INFO) << "Initializing TensorBoard schema";
  TF_CHECK_OK(SetupTensorboardSqliteDb(db));

  LOG(INFO) << "Creating SummaryDbWriter";
  SummaryWriterInterface* db_writer;
  TF_CHECK_OK(CreateSummaryDbWriter(db, experiment_name, run_name, user_name,
                                    env, &db_writer));
  core::ScopedUnref unref(db_writer);

  LOG(INFO) << "Loading TF event log: " << events;
  std::unique_ptr<RandomAccessFile> file;
  TF_CHECK_OK(env->NewRandomAccessFile(events, &file));
  io::RecordReader reader(file.get());

  uint64 start = env->NowMicros();
  uint64 records = 0;
  uint64 offset = 0;
  string record;
  while (true) {
    std::unique_ptr<Event> event = std::unique_ptr<Event>(new Event);
    Status s = reader.ReadRecord(&offset, &record);
    if (s.code() == error::OUT_OF_RANGE) break;
    TF_CHECK_OK(s);
    if (!ParseProtoUnlimited(event.get(), record)) {
      LOG(FATAL) << "Corrupt tf.Event record"
                 << " offset=" << (offset - record.size())
                 << " size=" << static_cast<int>(record.size());
    }
    TF_CHECK_OK(db_writer->WriteEvent(std::move(event)));
    ++records;
  }
  uint64 elapsed = env->NowMicros() - start;
  LOG(INFO) << "Loaded " << AddCommas(offset) << " bytes with "
            << AddCommas(records) << " records at "
            << (elapsed == 0 ? offset : static_cast<uint64>(
                                            offset / (elapsed / 1000000.0)))
            << " bps";
  return 0;
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char* argv[]) { return tensorflow::main(argc, argv); }

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/debug_events_writer.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"

namespace tensorflow {
namespace tfdbg {

// shorthand
Env* env() { return Env::Default(); }

class DebugEventsWriterTest : public ::testing::Test {
 public:
  static string GetDebugEventFileName(DebugEventsWriter* writer,
                                      DebugEventFileType type) {
    return writer->FileName(type);
  }

  static void ReadDebugEventProtos(DebugEventsWriter* writer,
                                   DebugEventFileType type,
                                   std::vector<DebugEvent>* protos) {
    protos->clear();
    const string filename = writer->FileName(type);
    std::unique_ptr<RandomAccessFile> debug_events_file;
    TF_CHECK_OK(env()->NewRandomAccessFile(filename, &debug_events_file));
    io::RecordReader* reader = new io::RecordReader(debug_events_file.get());

    uint64 offset = 0;
    DebugEvent actual;
    while (ReadDebugEventProto(reader, &offset, &actual)) {
      protos->push_back(actual);
    }

    delete reader;
  }

  static bool ReadDebugEventProto(io::RecordReader* reader, uint64* offset,
                                  DebugEvent* proto) {
    tstring record;
    Status s = reader->ReadRecord(offset, &record);
    if (!s.ok()) {
      return false;
    }
    return ParseProtoUnlimited(proto, record);
  }

  void SetUp() override {
    dump_root_ = io::JoinPath(
        testing::TmpDir(),
        strings::Printf("%010lld", static_cast<long long>(env()->NowMicros())));
  }

  void TearDown() override {
    if (env()->IsDirectory(dump_root_).ok()) {
      int64 undeleted_files = 0;
      int64 undeleted_dirs = 0;
      TF_ASSERT_OK(env()->DeleteRecursively(dump_root_, &undeleted_files,
                                            &undeleted_dirs));
      ASSERT_EQ(0, undeleted_files);
      ASSERT_EQ(0, undeleted_dirs);
    }
  }

  string dump_root_;
};

TEST_F(DebugEventsWriterTest, GetDebugEventsWriterSameRootGivesSameObject) {
  // Test the per-dump_root_ singleton pattern.
  DebugEventsWriter* writer_1 =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  DebugEventsWriter* writer_2 =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  EXPECT_EQ(writer_1, writer_2);
}

TEST_F(DebugEventsWriterTest, ConcurrentGetDebugEventsWriterSameDumpRoot) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 4);

  std::vector<DebugEventsWriter*> writers;
  mutex mu;
  auto fn = [this, &writers, &mu]() {
    DebugEventsWriter* writer =
        DebugEventsWriter::GetDebugEventsWriter(dump_root_);
    {
      mutex_lock l(mu);
      writers.push_back(writer);
    }
  };
  for (size_t i = 0; i < 4; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  EXPECT_EQ(writers.size(), 4);
  EXPECT_EQ(writers[0], writers[1]);
  EXPECT_EQ(writers[1], writers[2]);
  EXPECT_EQ(writers[2], writers[3]);
}

TEST_F(DebugEventsWriterTest, ConcurrentGetDebugEventsWriterDiffDumpRoots) {
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 3);

  std::atomic_int_fast64_t counter(0);
  std::vector<DebugEventsWriter*> writers;
  mutex mu;
  auto fn = [this, &counter, &writers, &mu]() {
    const string new_dump_root =
        io::JoinPath(dump_root_, strings::Printf("%ld", counter.fetch_add(1)));
    DebugEventsWriter* writer =
        DebugEventsWriter::GetDebugEventsWriter(new_dump_root);
    {
      mutex_lock l(mu);
      writers.push_back(writer);
    }
  };
  for (size_t i = 0; i < 3; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  EXPECT_EQ(writers.size(), 3);
  EXPECT_NE(writers[0], writers[1]);
  EXPECT_NE(writers[0], writers[2]);
  EXPECT_NE(writers[1], writers[2]);
}

TEST_F(DebugEventsWriterTest, GetDebugEventsWriterDifferentRoots) {
  // Test the DebugEventsWriters for different directories are different.
  DebugEventsWriter* writer_1 =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  const string dump_root_2 = io::JoinPath(dump_root_, "subdirectory");
  DebugEventsWriter* writer_2 =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_2);
  EXPECT_NE(writer_1, writer_2);
}

TEST_F(DebugEventsWriterTest, GetAndInitDebugEventsWriter) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());
  TF_ASSERT_OK(writer->Close());

  // Verify the metadata file's content.
  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  // Check the content of the file version string.
  const string file_version = actuals[0].debug_metadata().file_version();
  EXPECT_EQ(file_version.find(DebugEventsWriter::kVersionPrefix), 0);
  EXPECT_GT(file_version.size(), strlen(DebugEventsWriter::kVersionPrefix));

  // Verify that the .source_files file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  // Verify that the .stack_frames file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
}

TEST_F(DebugEventsWriterTest, CallingCloseWithoutInitIsOkay) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, CallingCloseTwiceIsOkay) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Close());
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, ConcurrentInitCalls) {
  // Test that concurrent calls to Init() works correctly.
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 4);
  auto fn = [&writer]() { TF_ASSERT_OK(writer->Init()); };
  for (size_t i = 0; i < 3; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  // Verify the metadata file's content.
  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  // Check the content of the file version string.
  const string file_version = actuals[0].debug_metadata().file_version();
  EXPECT_EQ(file_version.find(DebugEventsWriter::kVersionPrefix), 0);
  EXPECT_GT(file_version.size(), strlen(DebugEventsWriter::kVersionPrefix));

  // Verify that the .source_files file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  // Verify that the .stack_frames file has been created and is empty.
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
}

TEST_F(DebugEventsWriterTest, InitTwiceDoesNotCreateNewMetadataFile) {
  // Test that Init() is idempotent.
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  EXPECT_GE(actuals[0].debug_metadata().file_version().size(), 0);

  string metadata_path_1 =
      GetDebugEventFileName(writer, DebugEventFileType::METADATA);
  TF_ASSERT_OK(writer->Init());
  EXPECT_EQ(GetDebugEventFileName(writer, DebugEventFileType::METADATA),
            metadata_path_1);
  TF_ASSERT_OK(writer->Close());

  // Verify the metadata file's content.
  ReadDebugEventProtos(writer, DebugEventFileType::METADATA, &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_GT(actuals[0].debug_metadata().tensorflow_version().length(), 0);
  EXPECT_GE(actuals[0].debug_metadata().file_version().size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteSourceFile) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  SourceFile* source_file_1 = new SourceFile();
  source_file_1->set_file_path("/home/tf_programs/main.py");
  source_file_1->set_host_name("localhost.localdomain");
  source_file_1->add_lines("import tensorflow as tf");
  source_file_1->add_lines("");
  source_file_1->add_lines("print(tf.constant([42.0]))");
  source_file_1->add_lines("");
  TF_ASSERT_OK(writer->WriteSourceFile(source_file_1));

  SourceFile* source_file_2 = new SourceFile();
  source_file_2->set_file_path("/home/tf_programs/train.py");
  source_file_2->set_host_name("localhost.localdomain");
  source_file_2->add_lines("import tensorflow.keras as keras");
  source_file_2->add_lines("");
  source_file_2->add_lines("model = keras.Sequential()");
  TF_ASSERT_OK(writer->WriteSourceFile(source_file_2));

  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 2);
  EXPECT_GT(actuals[0].wall_time(), 0);
  EXPECT_GT(actuals[1].wall_time(), actuals[0].wall_time());

  SourceFile actual_source_file_1 = actuals[0].source_file();
  EXPECT_EQ(actual_source_file_1.file_path(), "/home/tf_programs/main.py");
  EXPECT_EQ(actual_source_file_1.host_name(), "localhost.localdomain");
  EXPECT_EQ(actual_source_file_1.lines().size(), 4);
  EXPECT_EQ(actual_source_file_1.lines()[0], "import tensorflow as tf");
  EXPECT_EQ(actual_source_file_1.lines()[1], "");
  EXPECT_EQ(actual_source_file_1.lines()[2], "print(tf.constant([42.0]))");
  EXPECT_EQ(actual_source_file_1.lines()[3], "");

  SourceFile actual_source_file_2 = actuals[1].source_file();
  EXPECT_EQ(actual_source_file_2.file_path(), "/home/tf_programs/train.py");
  EXPECT_EQ(actual_source_file_2.host_name(), "localhost.localdomain");
  EXPECT_EQ(actual_source_file_2.lines().size(), 3);
  EXPECT_EQ(actual_source_file_2.lines()[0],
            "import tensorflow.keras as keras");
  EXPECT_EQ(actual_source_file_2.lines()[1], "");
  EXPECT_EQ(actual_source_file_2.lines()[2], "model = keras.Sequential()");

  // Verify no cross talk in the other non-execution debug-event files.
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteStackFramesFile) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  StackFrameWithId* stack_frame_1 = new StackFrameWithId();
  stack_frame_1->set_id("deadbeaf");
  GraphDebugInfo::FileLineCol* file_line_col =
      stack_frame_1->mutable_file_line_col();
  file_line_col->set_file_index(12);
  file_line_col->set_line(20);
  file_line_col->set_col(2);
  file_line_col->set_func("my_func");
  file_line_col->set_code("  x = y + z");

  StackFrameWithId* stack_frame_2 = new StackFrameWithId();
  stack_frame_2->set_id("eeeeeeec");
  file_line_col = stack_frame_2->mutable_file_line_col();
  file_line_col->set_file_index(12);
  file_line_col->set_line(21);
  file_line_col->set_col(4);
  file_line_col->set_func("my_func");
  file_line_col->set_code("  x = x ** 2.0");

  TF_ASSERT_OK(writer->WriteStackFrameWithId(stack_frame_1));
  TF_ASSERT_OK(writer->WriteStackFrameWithId(stack_frame_2));
  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 2);
  EXPECT_GT(actuals[0].wall_time(), 0);
  EXPECT_GT(actuals[1].wall_time(), actuals[0].wall_time());

  StackFrameWithId actual_stack_frame_1 = actuals[0].stack_frame_with_id();
  EXPECT_EQ(actual_stack_frame_1.id(), "deadbeaf");
  GraphDebugInfo::FileLineCol file_line_col_1 =
      actual_stack_frame_1.file_line_col();
  EXPECT_EQ(file_line_col_1.file_index(), 12);
  EXPECT_EQ(file_line_col_1.line(), 20);
  EXPECT_EQ(file_line_col_1.col(), 2);
  EXPECT_EQ(file_line_col_1.func(), "my_func");
  EXPECT_EQ(file_line_col_1.code(), "  x = y + z");

  StackFrameWithId actual_stack_frame_2 = actuals[1].stack_frame_with_id();
  EXPECT_EQ(actual_stack_frame_2.id(), "eeeeeeec");
  GraphDebugInfo::FileLineCol file_line_col_2 =
      actual_stack_frame_2.file_line_col();
  EXPECT_EQ(file_line_col_2.file_index(), 12);
  EXPECT_EQ(file_line_col_2.line(), 21);
  EXPECT_EQ(file_line_col_2.col(), 4);
  EXPECT_EQ(file_line_col_2.func(), "my_func");
  EXPECT_EQ(file_line_col_2.code(), "  x = x ** 2.0");

  // Verify no cross talk in the other non-execution debug-event files.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteGraphOpCreationAndDebuggedGraph) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  GraphOpCreation* graph_op_creation = new GraphOpCreation();
  graph_op_creation->set_op_type("MatMul");
  graph_op_creation->set_op_name("Dense_1/MatMul");
  TF_ASSERT_OK(writer->WriteGraphOpCreation(graph_op_creation));

  DebuggedGraph* debugged_graph = new DebuggedGraph();
  debugged_graph->set_graph_id("deadbeaf");
  debugged_graph->set_graph_name("my_func_graph");
  TF_ASSERT_OK(writer->WriteDebuggedGraph(debugged_graph));

  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 2);
  EXPECT_GT(actuals[0].wall_time(), 0);
  EXPECT_GT(actuals[1].wall_time(), actuals[0].wall_time());

  GraphOpCreation actual_op_creation = actuals[0].graph_op_creation();
  EXPECT_EQ(actual_op_creation.op_type(), "MatMul");
  EXPECT_EQ(actual_op_creation.op_name(), "Dense_1/MatMul");

  DebuggedGraph actual_debugged_graph = actuals[1].debugged_graph();
  EXPECT_EQ(actual_debugged_graph.graph_id(), "deadbeaf");
  EXPECT_EQ(actual_debugged_graph.graph_name(), "my_func_graph");

  // Verify no cross talk in the other non-execution debug-event files.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, ConcurrentWriteCallsToTheSameFile) {
  const size_t kConcurrentWrites = 100;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
    const string file_path = strings::Printf(
        "/home/tf_programs/program_%.3ld.py", counter.fetch_add(1));
    SourceFile* source_file = new SourceFile();
    source_file->set_file_path(file_path);
    source_file->set_host_name("localhost.localdomain");
    TF_ASSERT_OK(writer->WriteSourceFile(source_file));
  };
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites);
  std::vector<string> file_paths;
  std::vector<string> host_names;
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    file_paths.push_back(actuals[i].source_file().file_path());
    host_names.push_back(actuals[i].source_file().host_name());
  }
  std::sort(file_paths.begin(), file_paths.end());
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    EXPECT_EQ(file_paths[i],
              strings::Printf("/home/tf_programs/program_%.3ld.py", i));
    EXPECT_EQ(host_names[i], "localhost.localdomain");
  }
}

TEST_F(DebugEventsWriterTest, ConcurrentWriteAndFlushCallsToTheSameFile) {
  const size_t kConcurrentWrites = 100;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
    const string file_path = strings::Printf(
        "/home/tf_programs/program_%.3ld.py", counter.fetch_add(1));
    SourceFile* source_file = new SourceFile();
    source_file->set_file_path(file_path);
    source_file->set_host_name("localhost.localdomain");
    TF_ASSERT_OK(writer->WriteSourceFile(source_file));
    TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  };
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites);
  std::vector<string> file_paths;
  std::vector<string> host_names;
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    file_paths.push_back(actuals[i].source_file().file_path());
    host_names.push_back(actuals[i].source_file().host_name());
  }
  std::sort(file_paths.begin(), file_paths.end());
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    EXPECT_EQ(file_paths[i],
              strings::Printf("/home/tf_programs/program_%.3ld.py", i));
    EXPECT_EQ(host_names[i], "localhost.localdomain");
  }
}

TEST_F(DebugEventsWriterTest, ConcurrentWriteCallsToTheDifferentFiles) {
  const int32 kConcurrentWrites = 30;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 10);
  std::atomic_int_fast32_t counter(0);
  auto fn = [&writer, &counter]() {
    const int32 index = counter.fetch_add(1);
    if (index % 3 == 0) {
      SourceFile* source_file = new SourceFile();
      source_file->set_file_path(
          strings::Printf("/home/tf_programs/program_%.2d.py", index));
      source_file->set_host_name("localhost.localdomain");
      TF_ASSERT_OK(writer->WriteSourceFile(source_file));
    } else if (index % 3 == 1) {
      StackFrameWithId* stack_frame = new StackFrameWithId();
      stack_frame->set_id(strings::Printf("e%.2d", index));
      TF_ASSERT_OK(writer->WriteStackFrameWithId(stack_frame));
    } else {
      GraphOpCreation* op_creation = new GraphOpCreation();
      op_creation->set_op_type("Log");
      op_creation->set_op_name(strings::Printf("Log_%.2d", index));
      TF_ASSERT_OK(writer->WriteGraphOpCreation(op_creation));
    }
  };
  for (size_t i = 0; i < kConcurrentWrites; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;

  TF_ASSERT_OK(writer->Close());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites / 3);
  std::vector<string> file_paths;
  std::vector<string> host_names;
  for (int32 i = 0; i < kConcurrentWrites / 3; ++i) {
    file_paths.push_back(actuals[i].source_file().file_path());
    host_names.push_back(actuals[i].source_file().host_name());
  }
  std::sort(file_paths.begin(), file_paths.end());
  for (int32 i = 0; i < kConcurrentWrites / 3; ++i) {
    EXPECT_EQ(file_paths[i],
              strings::Printf("/home/tf_programs/program_%.2d.py", i * 3));
    EXPECT_EQ(host_names[i], "localhost.localdomain");
  }

  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites / 3);
  std::vector<string> stack_frame_ids;
  for (int32 i = 0; i < kConcurrentWrites / 3; ++i) {
    stack_frame_ids.push_back(actuals[i].stack_frame_with_id().id());
  }
  std::sort(stack_frame_ids.begin(), stack_frame_ids.end());
  for (int32 i = 0; i < kConcurrentWrites / 3; ++i) {
    EXPECT_EQ(stack_frame_ids[i], strings::Printf("e%.2d", i * 3 + 1));
  }

  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), kConcurrentWrites / 3);
  std::vector<string> op_types;
  std::vector<string> op_names;
  for (int32 i = 0; i < kConcurrentWrites / 3; ++i) {
    op_types.push_back(actuals[i].graph_op_creation().op_type());
    op_names.push_back(actuals[i].graph_op_creation().op_name());
  }
  std::sort(op_names.begin(), op_names.end());
  for (int32 i = 0; i < kConcurrentWrites / 3; ++i) {
    EXPECT_EQ(op_types[i], "Log");
    EXPECT_EQ(op_names[i], strings::Printf("Log_%.2d", i * 3 + 2));
  }
}

TEST_F(DebugEventsWriterTest, WriteExecutionWithCyclicBufferNoFlush) {
  // Verify that no writing to disk happens until the flushing method is called.
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    Execution* execution = new Execution();
    execution->set_op_type("Log");
    execution->add_input_tensor_ids(i);
    TF_ASSERT_OK(writer->WriteExecution(execution));
  }

  std::vector<DebugEvent> actuals;
  // Before FlushExecutionFiles() is called, the file should be empty.
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), 0);

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, WriteExecutionWithCyclicBufferFlush) {
  // Verify that writing to disk happens when the flushing method is called.
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    Execution* execution = new Execution();
    execution->set_op_type("Log");
    execution->add_input_tensor_ids(i);
    TF_ASSERT_OK(writer->WriteExecution(execution));
  }

  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  // Expect there to be only the last kCyclicBufferSize debug events,
  // and the order should be correct.
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), kCyclicBufferSize);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    EXPECT_EQ(actuals[i].execution().op_type(), "Log");
    EXPECT_EQ(actuals[i].execution().input_tensor_ids().size(), 1);
    EXPECT_EQ(actuals[i].execution().input_tensor_ids()[0],
              kCyclicBufferSize + i);
  }

  // Second, write more than the capacity of the circular buffer,
  // in a concurrent fashion.
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
    Execution* execution = new Execution();
    execution->set_op_type("Abs");
    execution->add_input_tensor_ids(counter.fetch_add(1));
    TF_ASSERT_OK(writer->WriteExecution(execution));
  };
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;
  TF_ASSERT_OK(writer->Close());

  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  // NOTE: This includes the files from the first stage above, because the
  // .execution file hasn't changed.
  EXPECT_EQ(actuals.size(), kCyclicBufferSize * 2);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    const size_t index = i + kCyclicBufferSize;
    EXPECT_EQ(actuals[index].execution().op_type(), "Abs");
    EXPECT_EQ(actuals[index].execution().input_tensor_ids().size(), 1);
    EXPECT_GE(actuals[index].execution().input_tensor_ids()[0], 0);
    EXPECT_LE(actuals[index].execution().input_tensor_ids()[0],
              kCyclicBufferSize * 2);
  }

  // Verify no cross-talk.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, WriteGrahExecutionTraceWithCyclicBufferNoFlush) {
  // Check no writing to disk happens before the flushing method is called.
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(strings::Printf("graph_%.2ld", i));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  }

  std::vector<DebugEvent> actuals;
  // Before FlushExecutionFiles() is called, the file should be empty.
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 0);

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, WriteGrahExecutionTraceWithoutPreviousInitCall) {
  const size_t kCyclicBufferSize = -1;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_, kCyclicBufferSize);
  // NOTE(cais): `writer->Init()` is not called here before
  // WriteGraphExecutionTrace() is called. This test checks that this is okay
  // and the `GraphExecutionTrace` gets written correctly even without `Init()`
  // being called first. This scenario can happen when a TF Graph with tfdbg
  // debug ops are executed on a remote TF server.

  GraphExecutionTrace* trace = new GraphExecutionTrace();
  trace->set_tfdbg_context_id(strings::Printf("graph_0"));
  TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), 1);
  EXPECT_EQ(actuals[0].graph_execution_trace().tfdbg_context_id(), "graph_0");

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

TEST_F(DebugEventsWriterTest, WriteGrahExecutionTraceWithCyclicBufferFlush) {
  const size_t kCyclicBufferSize = 10;
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  // First, try writing and flushing more debug events than the capacity
  // of the circular buffer, in a serial fashion.
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(strings::Printf("graph_%.2ld", i));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  }

  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  // Expect there to be only the last kCyclicBufferSize debug events,
  // and the order should be correct.
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), kCyclicBufferSize);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    EXPECT_EQ(actuals[i].graph_execution_trace().tfdbg_context_id(),
              strings::Printf("graph_%.2ld", i + kCyclicBufferSize));
  }

  // Second, write more than the capacity of the circular buffer,
  // in a concurrent fashion.
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  std::atomic_int_fast64_t counter(0);
  auto fn = [&writer, &counter]() {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(
        strings::Printf("new_graph_%.2ld", counter.fetch_add(1)));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  };
  for (size_t i = 0; i < kCyclicBufferSize * 2; ++i) {
    thread_pool->Schedule(fn);
  }
  delete thread_pool;
  TF_ASSERT_OK(writer->Close());

  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  // NOTE: This includes the files from the first stage above, because the
  // .graph_execution_traces file hasn't changed.
  EXPECT_EQ(actuals.size(), kCyclicBufferSize * 2);
  for (size_t i = 0; i < kCyclicBufferSize; ++i) {
    const size_t index = i + kCyclicBufferSize;
    EXPECT_EQ(actuals[index].graph_execution_trace().tfdbg_context_id().find(
                  "new_graph_"),
              0);
  }

  // Verify no cross-talk.
  ReadDebugEventProtos(writer, DebugEventFileType::SOURCE_FILES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::STACK_FRAMES, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  EXPECT_EQ(actuals.size(), 0);
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), 0);
}

TEST_F(DebugEventsWriterTest, RegisterDeviceAndGetIdTrace) {
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_);
  TF_ASSERT_OK(writer->Init());

  // Register and get some device IDs in a concurrent fashion.
  thread::ThreadPool* thread_pool =
      new thread::ThreadPool(Env::Default(), "test_pool", 8);
  int device_ids[8];
  for (int i = 0; i < 8; ++i) {
    thread_pool->Schedule([i, &writer, &device_ids]() {
      const string device_name = strings::Printf(
          "/job:localhost/replica:0/task:0/device:GPU:%d", i % 4);
      device_ids[i] = writer->RegisterDeviceAndGetId(device_name);
    });
  }
  delete thread_pool;
  TF_ASSERT_OK(writer->FlushNonExecutionFiles());
  TF_ASSERT_OK(writer->Close());

  // There should be only 4 unique device IDs, because there are only 4 unique
  // device names.
  EXPECT_EQ(device_ids[0], device_ids[4]);
  EXPECT_EQ(device_ids[1], device_ids[5]);
  EXPECT_EQ(device_ids[2], device_ids[6]);
  EXPECT_EQ(device_ids[3], device_ids[7]);
  // Assert that the four device IDs are all unique.
  EXPECT_EQ(absl::flat_hash_set<int>(device_ids, device_ids + 8).size(), 4);

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::GRAPHS, &actuals);
  // Due to the `% 4`, there are only 4 unique device names, even though there
  // are 8 threads each calling `RegisterDeviceAndGetId`.
  EXPECT_EQ(actuals.size(), 4);
  for (const DebugEvent& actual : actuals) {
    const string& device_name = actual.debugged_device().device_name();
    int device_index = -1;
    CHECK(absl::SimpleAtoi(device_name.substr(strlen(
                               "/job:localhost/replica:0/task:0/device:GPU:")),
                           &device_index));
    EXPECT_EQ(actual.debugged_device().device_id(), device_ids[device_index]);
  }
}

TEST_F(DebugEventsWriterTest, DisableCyclicBufferBehavior) {
  const size_t kCyclicBufferSize = 0;  // A value <= 0 disables cyclic behavior.
  DebugEventsWriter* writer =
      DebugEventsWriter::GetDebugEventsWriter(dump_root_, kCyclicBufferSize);
  TF_ASSERT_OK(writer->Init());

  const size_t kNumEvents = 20;

  for (size_t i = 0; i < kNumEvents; ++i) {
    Execution* execution = new Execution();
    execution->set_op_type("Log");
    execution->add_input_tensor_ids(i);
    TF_ASSERT_OK(writer->WriteExecution(execution));
  }
  TF_ASSERT_OK(writer->FlushExecutionFiles());

  std::vector<DebugEvent> actuals;
  ReadDebugEventProtos(writer, DebugEventFileType::EXECUTION, &actuals);
  EXPECT_EQ(actuals.size(), kNumEvents);
  for (size_t i = 0; i < kNumEvents; ++i) {
    EXPECT_EQ(actuals[i].execution().op_type(), "Log");
    EXPECT_EQ(actuals[i].execution().input_tensor_ids().size(), 1);
    EXPECT_EQ(actuals[i].execution().input_tensor_ids()[0], i);
  }

  for (size_t i = 0; i < kNumEvents; ++i) {
    GraphExecutionTrace* trace = new GraphExecutionTrace();
    trace->set_tfdbg_context_id(strings::Printf("graph_%.2ld", i));
    TF_ASSERT_OK(writer->WriteGraphExecutionTrace(trace));
  }
  TF_ASSERT_OK(writer->FlushExecutionFiles());

  ReadDebugEventProtos(writer, DebugEventFileType::GRAPH_EXECUTION_TRACES,
                       &actuals);
  EXPECT_EQ(actuals.size(), kNumEvents);
  for (size_t i = 0; i < kNumEvents; ++i) {
    EXPECT_EQ(actuals[i].graph_execution_trace().tfdbg_context_id(),
              strings::Printf("graph_%.2ld", i));
  }

  // Close the writer so the files can be safely deleted.
  TF_ASSERT_OK(writer->Close());
}

}  // namespace tfdbg
}  // namespace tensorflow

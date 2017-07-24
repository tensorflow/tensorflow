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

#include "tensorflow/core/util/events_writer.h"

#include <math.h>
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/event.pb.h"

namespace tensorflow {
namespace {

// shorthand
Env* env() { return Env::Default(); }

void WriteSimpleValue(EventsWriter* writer, double wall_time, int64 step,
                      const string& tag, float simple_value) {
  Event event;
  event.set_wall_time(wall_time);
  event.set_step(step);
  Summary::Value* summ_val = event.mutable_summary()->add_value();
  summ_val->set_tag(tag);
  summ_val->set_simple_value(simple_value);
  writer->WriteEvent(event);
}

void WriteFile(EventsWriter* writer) {
  WriteSimpleValue(writer, 1234, 34, "foo", 3.14159);
  WriteSimpleValue(writer, 2345, 35, "bar", -42);
}

static bool ReadEventProto(io::RecordReader* reader, uint64* offset,
                           Event* proto) {
  string record;
  Status s = reader->ReadRecord(offset, &record);
  if (!s.ok()) {
    return false;
  }
  return ParseProtoUnlimited(proto, record);
}

void VerifyFile(const string& filename) {
  CHECK(env()->FileExists(filename).ok());
  std::unique_ptr<RandomAccessFile> event_file;
  TF_CHECK_OK(env()->NewRandomAccessFile(filename, &event_file));
  io::RecordReader* reader = new io::RecordReader(event_file.get());

  uint64 offset = 0;

  Event actual;
  CHECK(ReadEventProto(reader, &offset, &actual));
  VLOG(1) << actual.ShortDebugString();
  // Wall time should be within 5s of now.

  double current_time = env()->NowMicros() / 1000000.0;
  EXPECT_LT(fabs(actual.wall_time() - current_time), 5);
  // Should have the current version number.
  EXPECT_EQ(actual.file_version(),
            strings::StrCat(EventsWriter::kVersionPrefix,
                            EventsWriter::kCurrentVersion));

  Event expected;
  CHECK(ReadEventProto(reader, &offset, &actual));
  VLOG(1) << actual.ShortDebugString();
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "wall_time: 1234 step: 34 "
      "summary { value { tag: 'foo' simple_value: 3.14159 } }",
      &expected));
  // TODO(keveman): Enable this check
  // EXPECT_THAT(expected, EqualsProto(actual));

  CHECK(ReadEventProto(reader, &offset, &actual));
  VLOG(1) << actual.ShortDebugString();
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "wall_time: 2345 step: 35 "
      "summary { value { tag: 'bar' simple_value: -42 } }",
      &expected));
  // TODO(keveman): Enable this check
  // EXPECT_THAT(expected, EqualsProto(actual));

  TF_CHECK_OK(env()->DeleteFile(filename));
  delete reader;
}

string GetDirName(const string& suffix) {
  return io::JoinPath(testing::TmpDir(), suffix);
}

TEST(EventWriter, WriteFlush) {
  string file_prefix = GetDirName("/writeflush_test");
  EventsWriter writer(file_prefix);
  WriteFile(&writer);
  EXPECT_TRUE(writer.Flush());
  string filename = writer.FileName();
  VerifyFile(filename);
}

TEST(EventWriter, WriteClose) {
  string file_prefix = GetDirName("/writeclose_test");
  EventsWriter writer(file_prefix);
  WriteFile(&writer);
  EXPECT_TRUE(writer.Close());
  string filename = writer.FileName();
  VerifyFile(filename);
}

TEST(EventWriter, WriteDelete) {
  string file_prefix = GetDirName("/writedelete_test");
  EventsWriter* writer = new EventsWriter(file_prefix);
  WriteFile(writer);
  string filename = writer->FileName();
  delete writer;
  VerifyFile(filename);
}

TEST(EventWriter, FailFlush) {
  string file_prefix = GetDirName("/failflush_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  WriteFile(&writer);
  TF_EXPECT_OK(env()->FileExists(filename));
  TF_ASSERT_OK(env()->DeleteFile(filename));
  EXPECT_EQ(errors::Code::NOT_FOUND, env()->FileExists(filename).code());
  EXPECT_FALSE(writer.Flush());
  EXPECT_EQ(errors::Code::NOT_FOUND, env()->FileExists(filename).code());
}

TEST(EventWriter, FailClose) {
  string file_prefix = GetDirName("/failclose_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  WriteFile(&writer);
  TF_EXPECT_OK(env()->FileExists(filename));
  TF_ASSERT_OK(env()->DeleteFile(filename));
  EXPECT_EQ(errors::Code::NOT_FOUND, env()->FileExists(filename).code());
  EXPECT_FALSE(writer.Close());
  EXPECT_EQ(errors::Code::NOT_FOUND, env()->FileExists(filename).code());
}

TEST(EventWriter, InitWriteClose) {
  string file_prefix = GetDirName("/initwriteclose_test");
  EventsWriter writer(file_prefix);
  EXPECT_TRUE(writer.Init());
  string filename0 = writer.FileName();
  TF_EXPECT_OK(env()->FileExists(filename0));
  WriteFile(&writer);
  EXPECT_TRUE(writer.Close());
  string filename1 = writer.FileName();
  EXPECT_EQ(filename0, filename1);
  VerifyFile(filename1);
}

TEST(EventWriter, NameWriteClose) {
  string file_prefix = GetDirName("/namewriteclose_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  TF_EXPECT_OK(env()->FileExists(filename));
  WriteFile(&writer);
  EXPECT_TRUE(writer.Close());
  VerifyFile(filename);
}

TEST(EventWriter, NameClose) {
  string file_prefix = GetDirName("/nameclose_test");
  EventsWriter writer(file_prefix);
  string filename = writer.FileName();
  EXPECT_TRUE(writer.Close());
  TF_EXPECT_OK(env()->FileExists(filename));
  TF_ASSERT_OK(env()->DeleteFile(filename));
}

TEST(EventWriter, FileDeletionBeforeWriting) {
  string file_prefix = GetDirName("/fdbw_test");
  EventsWriter writer(file_prefix);
  string filename0 = writer.FileName();
  TF_EXPECT_OK(env()->FileExists(filename0));
  env()->SleepForMicroseconds(
      2000000);  // To make sure timestamp part of filename will differ.
  TF_ASSERT_OK(env()->DeleteFile(filename0));
  EXPECT_TRUE(writer.Init());  // Init should reopen file.
  WriteFile(&writer);
  EXPECT_TRUE(writer.Flush());
  string filename1 = writer.FileName();
  EXPECT_NE(filename0, filename1);
  VerifyFile(filename1);
}

}  // namespace
}  // namespace tensorflow

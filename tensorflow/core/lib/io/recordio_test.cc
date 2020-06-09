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

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace io {
namespace {

// Construct a string of the specified length made out of the supplied
// partial string.
string BigString(const string& partial_string, size_t n) {
  string result;
  while (result.size() < n) {
    result.append(partial_string);
  }
  result.resize(n);
  return result;
}

// Construct a string from a number
string NumberString(int n) {
  char buf[50];
  snprintf(buf, sizeof(buf), "%d.", n);
  return string(buf);
}

// Return a skewed potentially long string
string RandomSkewedString(int i, random::SimplePhilox* rnd) {
  return BigString(NumberString(i), rnd->Skewed(17));
}

class StringDest : public WritableFile {
 public:
  explicit StringDest(string* contents) : contents_(contents) {}

  Status Close() override { return Status::OK(); }
  Status Flush() override { return Status::OK(); }
  Status Sync() override { return Status::OK(); }
  Status Append(StringPiece slice) override {
    contents_->append(slice.data(), slice.size());
    return Status::OK();
  }
#if defined(PLATFORM_GOOGLE)
  Status Append(const absl::Cord& data) override {
    contents_->append(std::string(data));
    return Status::OK();
  }
#endif
  Status Tell(int64* pos) override {
    *pos = contents_->size();
    return Status::OK();
  }

 private:
  string* contents_;
};

class StringSource : public RandomAccessFile {
 public:
  explicit StringSource(string* contents)
      : contents_(contents), force_error_(false) {}

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    if (force_error_) {
      force_error_ = false;
      return errors::DataLoss("read error");
    }

    if (offset >= contents_->size()) {
      return errors::OutOfRange("end of file");
    }

    if (contents_->size() < offset + n) {
      n = contents_->size() - offset;
    }
    *result = StringPiece(contents_->data() + offset, n);
    return Status::OK();
  }

  void force_error() { force_error_ = true; }

 private:
  string* contents_;
  mutable bool force_error_;
};

class RecordioTest : public ::testing::Test {
 private:
  string contents_;
  StringDest dest_;
  StringSource source_;
  bool reading_;
  uint64 readpos_;
  RecordWriter* writer_;
  RecordReader* reader_;

 public:
  RecordioTest()
      : dest_(&contents_),
        source_(&contents_),
        reading_(false),
        readpos_(0),
        writer_(new RecordWriter(&dest_)),
        reader_(new RecordReader(&source_)) {}

  ~RecordioTest() override {
    delete writer_;
    delete reader_;
  }

  void Write(const string& msg) {
    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    TF_ASSERT_OK(writer_->WriteRecord(StringPiece(msg)));
  }

#if defined(PLATFORM_GOOGLE)
  void Write(const absl::Cord& msg) {
    ASSERT_TRUE(!reading_) << "Write() after starting to read";
    TF_ASSERT_OK(writer_->WriteRecord(msg));
  }
#endif

  size_t WrittenBytes() const { return contents_.size(); }

  string Read() {
    if (!reading_) {
      reading_ = true;
    }
    tstring record;
    Status s = reader_->ReadRecord(&readpos_, &record);
    if (s.ok()) {
      return record;
    } else if (errors::IsOutOfRange(s)) {
      return "EOF";
    } else {
      return s.ToString();
    }
  }

  void IncrementByte(int offset, int delta) { contents_[offset] += delta; }

  void SetByte(int offset, char new_byte) { contents_[offset] = new_byte; }

  void ShrinkSize(int bytes) { contents_.resize(contents_.size() - bytes); }

  void FixChecksum(int header_offset, int len) {
    // Compute crc of type/len/data
    uint32_t crc = crc32c::Value(&contents_[header_offset + 6], 1 + len);
    crc = crc32c::Mask(crc);
    core::EncodeFixed32(&contents_[header_offset], crc);
  }

  void ForceError() { source_.force_error(); }

  void StartReadingAt(uint64_t initial_offset) { readpos_ = initial_offset; }

  void CheckOffsetPastEndReturnsNoRecords(uint64_t offset_past_end) {
    Write("foo");
    Write("bar");
    Write(BigString("x", 10000));
    reading_ = true;
    uint64 offset = WrittenBytes() + offset_past_end;
    tstring record;
    Status s = reader_->ReadRecord(&offset, &record);
    ASSERT_TRUE(errors::IsOutOfRange(s)) << s;
  }
};

TEST_F(RecordioTest, Empty) { ASSERT_EQ("EOF", Read()); }

TEST_F(RecordioTest, ReadWrite) {
  Write("foo");
  Write("bar");
  Write("");
  Write("xxxx");
  ASSERT_EQ("foo", Read());
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("", Read());
  ASSERT_EQ("xxxx", Read());
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());  // Make sure reads at eof work
}

#if defined(PLATFORM_GOOGLE)
TEST_F(RecordioTest, ReadWriteCords) {
  Write(absl::Cord("foo"));
  Write(absl::Cord("bar"));
  Write(absl::Cord(""));
  Write(absl::Cord("xxxx"));
  ASSERT_EQ("foo", Read());
  ASSERT_EQ("bar", Read());
  ASSERT_EQ("", Read());
  ASSERT_EQ("xxxx", Read());
  ASSERT_EQ("EOF", Read());
  ASSERT_EQ("EOF", Read());  // Make sure reads at eof work
}
#endif

TEST_F(RecordioTest, ManyRecords) {
  for (int i = 0; i < 100000; i++) {
    Write(NumberString(i));
  }
  for (int i = 0; i < 100000; i++) {
    ASSERT_EQ(NumberString(i), Read());
  }
  ASSERT_EQ("EOF", Read());
}

TEST_F(RecordioTest, RandomRead) {
  const int N = 500;
  {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    for (int i = 0; i < N; i++) {
      Write(RandomSkewedString(i, &rnd));
    }
  }
  {
    random::PhiloxRandom philox(301, 17);
    random::SimplePhilox rnd(&philox);
    for (int i = 0; i < N; i++) {
      ASSERT_EQ(RandomSkewedString(i, &rnd), Read());
    }
  }
  ASSERT_EQ("EOF", Read());
}

void TestNonSequentialReads(const RecordWriterOptions& writer_options,
                            const RecordReaderOptions& reader_options) {
  string contents;
  StringDest dst(&contents);
  RecordWriter writer(&dst, writer_options);
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK(writer.WriteRecord(NumberString(i))) << i;
  }
  TF_ASSERT_OK(writer.Close());

  StringSource file(&contents);
  RecordReader reader(&file, reader_options);

  tstring record;
  // First read sequentially to fill in the offsets table.
  uint64 offsets[10] = {0};
  uint64 offset = 0;
  for (int i = 0; i < 10; ++i) {
    offsets[i] = offset;
    TF_ASSERT_OK(reader.ReadRecord(&offset, &record)) << i;
  }

  // Read randomly: First go back to record #3 then forward to #8.
  offset = offsets[3];
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ("3.", record);
  EXPECT_EQ(offsets[4], offset);

  offset = offsets[8];
  TF_ASSERT_OK(reader.ReadRecord(&offset, &record));
  EXPECT_EQ("8.", record);
  EXPECT_EQ(offsets[9], offset);
}

TEST_F(RecordioTest, NonSequentialReads) {
  TestNonSequentialReads(RecordWriterOptions(), RecordReaderOptions());
}

TEST_F(RecordioTest, NonSequentialReadsWithReadBuffer) {
  RecordReaderOptions options;
  options.buffer_size = 1 << 10;
  TestNonSequentialReads(RecordWriterOptions(), options);
}

TEST_F(RecordioTest, NonSequentialReadsWithCompression) {
  TestNonSequentialReads(
      RecordWriterOptions::CreateRecordWriterOptions("ZLIB"),
      RecordReaderOptions::CreateRecordReaderOptions("ZLIB"));
}

// Tests of all the error paths in log_reader.cc follow:
void AssertHasSubstr(StringPiece s, StringPiece expected) {
  EXPECT_TRUE(absl::StrContains(s, expected))
      << s << " does not contain " << expected;
}

void TestReadError(const RecordWriterOptions& writer_options,
                   const RecordReaderOptions& reader_options) {
  const string wrote = BigString("well hello there!", 100);
  string contents;
  StringDest dst(&contents);
  TF_ASSERT_OK(RecordWriter(&dst, writer_options).WriteRecord(wrote));

  StringSource file(&contents);
  RecordReader reader(&file, reader_options);

  uint64 offset = 0;
  tstring read;
  file.force_error();
  Status status = reader.ReadRecord(&offset, &read);
  ASSERT_TRUE(errors::IsDataLoss(status));
  ASSERT_EQ(0, offset);

  // A failed Read() shouldn't update the offset, and thus a retry shouldn't
  // lose the record.
  status = reader.ReadRecord(&offset, &read);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_GT(offset, 0);
  EXPECT_EQ(wrote, read);
}

TEST_F(RecordioTest, ReadError) {
  TestReadError(RecordWriterOptions(), RecordReaderOptions());
}

TEST_F(RecordioTest, ReadErrorWithBuffering) {
  RecordReaderOptions options;
  options.buffer_size = 1 << 20;
  TestReadError(RecordWriterOptions(), options);
}

TEST_F(RecordioTest, ReadErrorWithCompression) {
  TestReadError(RecordWriterOptions::CreateRecordWriterOptions("ZLIB"),
                RecordReaderOptions::CreateRecordReaderOptions("ZLIB"));
}

TEST_F(RecordioTest, CorruptLength) {
  Write("foo");
  IncrementByte(6, 100);
  AssertHasSubstr(Read(), "Data loss");
}

TEST_F(RecordioTest, CorruptLengthCrc) {
  Write("foo");
  IncrementByte(10, 100);
  AssertHasSubstr(Read(), "Data loss");
}

TEST_F(RecordioTest, CorruptData) {
  Write("foo");
  IncrementByte(14, 10);
  AssertHasSubstr(Read(), "Data loss");
}

TEST_F(RecordioTest, CorruptDataCrc) {
  Write("foo");
  IncrementByte(WrittenBytes() - 1, 10);
  AssertHasSubstr(Read(), "Data loss");
}

TEST_F(RecordioTest, ReadEnd) { CheckOffsetPastEndReturnsNoRecords(0); }

TEST_F(RecordioTest, ReadPastEnd) { CheckOffsetPastEndReturnsNoRecords(5); }

}  // namespace
}  // namespace io
}  // namespace tensorflow

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

// Construct a string of the specified length made out of the supplied
// partial string.
static string BigString(const string& partial_string, size_t n) {
  string result;
  while (result.size() < n) {
    result.append(partial_string);
  }
  result.resize(n);
  return result;
}

// Construct a string from a number
static string NumberString(int n) {
  char buf[50];
  snprintf(buf, sizeof(buf), "%d.", n);
  return string(buf);
}

// Return a skewed potentially long string
static string RandomSkewedString(int i, random::SimplePhilox* rnd) {
  return BigString(NumberString(i), rnd->Skewed(17));
}

class RecordioTest : public ::testing::Test {
 private:
  class StringDest : public WritableFile {
   public:
    string contents_;

    Status Close() override { return Status::OK(); }
    Status Flush() override { return Status::OK(); }
    Status Sync() override { return Status::OK(); }
    Status Append(const StringPiece& slice) override {
      contents_.append(slice.data(), slice.size());
      return Status::OK();
    }
  };

  class StringSource : public RandomAccessFile {
   public:
    StringPiece contents_;
    mutable bool force_error_;
    mutable bool returned_partial_;
    StringSource() : force_error_(false), returned_partial_(false) {}

    Status Read(uint64 offset, size_t n, StringPiece* result,
                char* scratch) const override {
      EXPECT_FALSE(returned_partial_) << "must not Read() after eof/error";

      if (force_error_) {
        force_error_ = false;
        returned_partial_ = true;
        return errors::DataLoss("read error");
      }

      if (offset >= contents_.size()) {
        return errors::OutOfRange("end of file");
      }

      if (contents_.size() < offset + n) {
        n = contents_.size() - offset;
        returned_partial_ = true;
      }
      *result = StringPiece(contents_.data() + offset, n);
      return Status::OK();
    }
  };

  StringDest dest_;
  StringSource source_;
  bool reading_;
  uint64 readpos_;
  RecordWriter* writer_;
  RecordReader* reader_;

 public:
  RecordioTest()
      : reading_(false),
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

  size_t WrittenBytes() const { return dest_.contents_.size(); }

  string Read() {
    if (!reading_) {
      reading_ = true;
      source_.contents_ = StringPiece(dest_.contents_);
    }
    string record;
    Status s = reader_->ReadRecord(&readpos_, &record);
    if (s.ok()) {
      return record;
    } else if (errors::IsOutOfRange(s)) {
      return "EOF";
    } else {
      return s.ToString();
    }
  }

  void IncrementByte(int offset, int delta) {
    dest_.contents_[offset] += delta;
  }

  void SetByte(int offset, char new_byte) {
    dest_.contents_[offset] = new_byte;
  }

  void ShrinkSize(int bytes) {
    dest_.contents_.resize(dest_.contents_.size() - bytes);
  }

  void FixChecksum(int header_offset, int len) {
    // Compute crc of type/len/data
    uint32_t crc = crc32c::Value(&dest_.contents_[header_offset + 6], 1 + len);
    crc = crc32c::Mask(crc);
    core::EncodeFixed32(&dest_.contents_[header_offset], crc);
  }

  void ForceError() { source_.force_error_ = true; }

  void StartReadingAt(uint64_t initial_offset) { readpos_ = initial_offset; }

  void CheckOffsetPastEndReturnsNoRecords(uint64_t offset_past_end) {
    Write("foo");
    Write("bar");
    Write(BigString("x", 10000));
    reading_ = true;
    source_.contents_ = StringPiece(dest_.contents_);
    uint64 offset = WrittenBytes() + offset_past_end;
    string record;
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

// Tests of all the error paths in log_reader.cc follow:
static void AssertHasSubstr(StringPiece s, StringPiece expected) {
  EXPECT_TRUE(str_util::StrContains(s, expected))
      << s << " does not contain " << expected;
}

TEST_F(RecordioTest, ReadError) {
  Write("foo");
  ForceError();
  AssertHasSubstr(Read(), "Data loss");
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

}  // namespace io
}  // namespace tensorflow

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

#include "tensorflow/core/kernels/hadoop_sequence_file/reader.h"

#include <vector>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace io {

class SequenceFileReaderTest : public ::testing::Test {
 protected:
  class StringSource : public RandomAccessFile {
   public:
    explicit StringSource(const string& content) : content_(content) {}

    Status Read(uint64 offset, size_t n, StringPiece* result,
                char* scratch) const override {
      if (offset >= content_.size()) {
        return errors::OutOfRange("end of file");
      }
      if (content_.size() < offset + n) {
        LOG(ERROR) << content_.size() << " " << offset << " " << n;
        n = content_.size() - offset;
        returned_partial_ = true;
      }
      *result = StringPiece(content_.data() + offset, n);
      return Status::OK();
    }

   private:
    string content_;
    mutable bool returned_partial_ = false;
  };

  SequenceFileReaderTest() :
      reader_(new SequenceFileReader(SequenceFileReaderOptions::Defaults())) {}

  void VerifyVarInt(const char* s, int len, const int64 want) {
    const string content(s, len);
    StringSource source(content);
    reader_->stream_ = MakeUnique<BufferedInputStream>(&source, 3);
    const int64 got = reader_->ReadHadoopVarIntOrDie();
    EXPECT_EQ(want, got);
  }

  void VerifyBigEndianUint32(const char* s, const uint64 want) {
    const string content(s, 4);
    StringSource source(content);
    reader_->stream_ = MakeUnique<BufferedInputStream>(&source, 3);
    uint32 got;
    TF_EXPECT_OK(reader_->ReadBigEndianUint32(&got));
    EXPECT_EQ(want, got);
  }

  void VerifyBigEndianInt32(const char* s, const int64 want) {
    const string content(s, 4);
    StringSource source(content);
    reader_->stream_ = MakeUnique<BufferedInputStream>(&source, 3);
    int32 got;
    TF_EXPECT_OK(reader_->ReadBigEndianInt32(&got));
    EXPECT_EQ(want, got);
  }

  std::unique_ptr<SequenceFileReader> reader_;
};

TEST_F(SequenceFileReaderTest, HadoopVarInt) {
  VerifyVarInt("\x00", 1, 0);
  VerifyVarInt("\x01", 1, 1);
  VerifyVarInt("\xff", 1, -1);
  VerifyVarInt("\x64", 1, 100);
  VerifyVarInt("\x9c", 1, -100);
  VerifyVarInt("\x8f\xc8", 2, 200);
  VerifyVarInt("\x87\xc7", 2, -200);
  VerifyVarInt("\x8e\x1f\xff", 3, 8191);
  VerifyVarInt("\x86\x1f\xfe", 3, -8191);
  VerifyVarInt("\x8c\x7f\xff\xff\xff", 5, 2147483647);
  VerifyVarInt("\x84\x7f\xff\xff\xfe", 5, -2147483647);
  VerifyVarInt("\x8c\x6d\x7f\x77\x58", 5, 1837070168);
  VerifyVarInt("\x84\x6d\x7f\x77\x57", 5, -1837070168);
  VerifyVarInt("\x8c\xff\xff\xff\xfe", 5, 4294967294);
  VerifyVarInt("\x84\xff\xff\xff\xfd", 5, -4294967294);
  VerifyVarInt("\x88\x08\x00\x00\x00\x00\x00\x00\x00", 9, 576460752303423488);
  VerifyVarInt("\x80\x07\xff\xff\xff\xff\xff\xff\xff", 9, -576460752303423488);
}

TEST_F(SequenceFileReaderTest, BigEndian) {
  VerifyBigEndianUint32("\x00\x01\x10\x11", 0x00011011);
  VerifyBigEndianUint32("\xff\xfe\x10\x2e", 0xfffe102e);
  VerifyBigEndianInt32("\x00\x01\x10\x11", 0x00011011);
  VerifyBigEndianInt32("\xff\xfe\x10\x2e", int32(0xfffe102e));
}

TEST_F(SequenceFileReaderTest, ActualHadoopOutput) {
  string txt;
  TF_CHECK_OK(ReadFileToString(Env::Default(),
      "tensorflow/core/kernels/hadoop_sequence_file/testdata/simple.txt",
      &txt));
  std::vector<string> want = str_util::Split(txt, "\n");
  // Discard the empty line before EOF. This is because Java file reading will
  // ignore the new line just before EOF, while ReadFileToString will keep the
  // last new line.
  want.pop_back();
  string seq;
  TF_CHECK_OK(ReadFileToString(Env::Default(),
      "tensorflow/core/kernels/hadoop_sequence_file/testdata/simple.seq",
      &seq));
  StringSource source(seq);
  SequenceFileReader reader(&source, SequenceFileReaderOptions::Defaults());
  std::vector<string> got;
  while (true) {
    string s;
    const auto status = reader.ReadRecord(&s);
    if (!status.ok()) {
      ASSERT_TRUE(errors::IsOutOfRange(status));
      break;
    }
    got.emplace_back(s);
  }
  EXPECT_EQ(want, got);
}

}  // namespace io
}  // namespace tensorflow

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/io/inputstream_interface.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace io {
namespace {

class TestStringStream : public InputStreamInterface {
 public:
  explicit TestStringStream(const string& content) : content_(content) {}

  Status ReadNBytes(int64 bytes_to_read, string* result) override {
    result->clear();
    if (pos_ + bytes_to_read > content_.size()) {
      return errors::OutOfRange("limit reached");
    }
    *result = content_.substr(pos_, bytes_to_read);
    pos_ += bytes_to_read;
    return Status::OK();
  }

  int64 Tell() const override { return pos_; }

  Status Reset() override {
    pos_ = 0;
    return Status::OK();
  }

 private:
  string content_;
  int64 pos_ = 0;
};

TEST(InputStreamInterface, Basic) {
  TestStringStream ss("This is a test string");
  string res;
  TF_ASSERT_OK(ss.ReadNBytes(4, &res));
  EXPECT_EQ("This", res);
  TF_ASSERT_OK(ss.SkipNBytes(6));
  TF_ASSERT_OK(ss.ReadNBytes(11, &res));
  EXPECT_EQ("test string", res);
  // Skipping past end of the file causes OutOfRange error.
  EXPECT_TRUE(errors::IsOutOfRange(ss.SkipNBytes(1)));

  TF_ASSERT_OK(ss.Reset());
  TF_ASSERT_OK(ss.ReadNBytes(4, &res));
  EXPECT_EQ("This", res);
}

}  // anonymous namespace
}  // namespace io
}  // namespace tensorflow

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
#include "tensorflow/lite/testing/tf_driver.h"

#include <algorithm>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/escaping.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace testing {
namespace {

class TestDriver : public TfDriver {
 public:
  // No need for a full TfDriver. We just want to test the read/write methods.
  TestDriver() : TfDriver({}, {}, {}, {}) {}
  string WriteAndReadBack(tensorflow::DataType type,
                          const std::vector<int64_t>& shape,
                          const string& values) {
    tensorflow::Tensor t = {
        type,
        tensorflow::TensorShape{tensorflow::gtl::ArraySlice<int64_t>{
            reinterpret_cast<const int64_t*>(shape.data()), shape.size()}}};
    SetInput(values, &t);
    return ReadOutput(t);
  }
};

TEST(TfDriverTest, ReadingAndWritingValues) {
  TestDriver driver;
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_FLOAT, {1, 2, 2},
                                    "0.10,0.20,0.30,0.40"),
            "0.100000001,0.200000003,0.300000012,0.400000006");
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_INT32, {1, 2, 2},
                                    "10,40,100,-100"),
            "10,40,100,-100");
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_UINT8, {1, 2, 2},
                                    "48,49,121, 122"),
            "0,1,y,z");
}

TEST(TfDriverTest, ReadingAndWritingValuesStrings) {
  TestDriver driver;

  auto set_buffer = [](const std::vector<string>& values, string* buffer) {
    DynamicBuffer dynamic_buffer;
    for (const string& s : values) {
      dynamic_buffer.AddString(s.data(), s.size());
    }

    char* char_b = nullptr;
    int size = dynamic_buffer.WriteToBuffer(&char_b);
    *buffer = absl::BytesToHexString(absl::string_view(char_b, size));
    free(char_b);
  };

  string buffer;

  set_buffer({"", "", "", ""}, &buffer);
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_STRING, {1, 2, 2}, buffer),
            buffer);

  // Note that if we pass the empty string we get the "empty" buffer (where all
  // the strings are empty).
  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_STRING, {1, 2, 2}, ""),
            buffer);

  set_buffer({"AB", "ABC", "X", "YZ"}, &buffer);

  ASSERT_EQ(driver.WriteAndReadBack(tensorflow::DT_STRING, {1, 2, 2}, buffer),
            buffer);
}

TEST(TfDriverTest, SimpleTest) {
  std::unique_ptr<TfDriver> runner(
      new TfDriver({"a", "b", "c", "d"}, {"float", "float", "float", "float"},
                   {"1,8,8,3", "1,8,8,3", "1,8,8,3", "1,8,8,3"}, {"x", "y"}));

  runner->LoadModel("tensorflow/lite/testdata/multi_add.pb");
  EXPECT_TRUE(runner->IsValid()) << runner->GetErrorMessage();

  for (const auto& i : {"a", "b", "c", "d"}) {
    runner->ReshapeTensor(i, "1,2,2,1");
  }
  ASSERT_TRUE(runner->IsValid());
  runner->ResetTensor("c");
  runner->Invoke({{"a", "0.1,0.2,0.3,0.4"},
                  {"b", "0.001,0.002,0.003,0.004"},
                  {"d", "0.01,0.02,0.03,0.04"}});

  ASSERT_EQ(runner->ReadOutput("x"),
            "0.101000004,0.202000007,0.303000003,0.404000014");
  ASSERT_EQ(runner->ReadOutput("y"),
            "0.0109999999,0.0219999999,0.0329999998,0.0439999998");
}

}  // namespace
}  // namespace testing
}  // namespace tflite

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
// Unit test for TFLite Lookup op.

#include <iomanip>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class HashtableLookupOpModel : public SingleOpModel {
 public:
  HashtableLookupOpModel(std::initializer_list<int> lookup_shape,
                         std::initializer_list<int> key_shape,
                         std::initializer_list<int> value_shape,
                         TensorType type) {
    lookup_ = AddInput(TensorType_INT32);
    key_ = AddInput(TensorType_INT32);
    value_ = AddInput(type);
    output_ = AddOutput(type);
    hit_ = AddOutput(TensorType_UINT8);
    SetBuiltinOp(BuiltinOperator_HASHTABLE_LOOKUP, BuiltinOptions_NONE, 0);
    BuildInterpreter({lookup_shape, key_shape, value_shape});
  }

  void SetLookup(std::initializer_list<int> data) {
    PopulateTensor<int>(lookup_, data);
  }

  void SetHashtableKey(std::initializer_list<int> data) {
    PopulateTensor<int>(key_, data);
  }

  void SetHashtableValue(const std::vector<string>& content) {
    PopulateStringTensor(value_, content);
  }

  void SetHashtableValue(const std::function<float(int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    for (int i = 0; i < rows; i++) {
      tensor->data.f[i] = function(i);
    }
  }

  void SetHashtableValue(const std::function<float(int, int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    int features = tensor->dims->data[1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < features; j++) {
        tensor->data.f[i * features + j] = function(i, j);
      }
    }
  }

  std::vector<string> GetStringOutput() {
    TfLiteTensor* output = interpreter_->tensor(output_);
    int num = GetStringCount(output);
    std::vector<string> result(num);
    for (int i = 0; i < num; i++) {
      auto ref = GetString(output, i);
      result[i] = string(ref.str, ref.len);
    }
    return result;
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<uint8_t> GetHit() { return ExtractVector<uint8_t>(hit_); }

 private:
  int lookup_;
  int key_;
  int value_;
  int output_;
  int hit_;
};

// TODO(yichengfan): write more tests that exercise the details of the op,
// such as lookup errors and variable input shapes.
TEST(HashtableLookupOpTest, Test2DInput) {
  HashtableLookupOpModel m({4}, {3}, {3, 2}, TensorType_FLOAT32);

  m.SetLookup({1234, -292, -11, 0});
  m.SetHashtableKey({-11, 0, 1234});
  m.SetHashtableValue([](int i, int j) { return i + j / 10.0f; });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 2.0, 2.1,  // 2-nd item
                                 0, 0,      // Not found
                                 0.0, 0.1,  // 0-th item
                                 1.0, 1.1,  // 1-st item
                             })));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({
                              1,
                              0,
                              1,
                              1,
                          }));
}

TEST(HashtableLookupOpTest, Test1DInput) {
  HashtableLookupOpModel m({4}, {3}, {3}, TensorType_FLOAT32);

  m.SetLookup({1234, -292, -11, 0});
  m.SetHashtableKey({-11, 0, 1234});
  m.SetHashtableValue([](int i) { return i * i / 10.0f; });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 0.4,  // 2-nd item
                                 0,    // Not found
                                 0.0,  // 0-th item
                                 0.1,  // 1-st item
                             })));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({
                              1,
                              0,
                              1,
                              1,
                          }));
}

TEST(HashtableLookupOpTest, TestString) {
  HashtableLookupOpModel m({4}, {3}, {3}, TensorType_STRING);

  m.SetLookup({1234, -292, -11, 0});
  m.SetHashtableKey({-11, 0, 1234});
  m.SetHashtableValue({"Hello", "", "Hi"});

  m.Invoke();

  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({
                                       "Hi",     // 2-nd item
                                       "",       // Not found
                                       "Hello",  // 0-th item
                                       "",       // 1-st item
                                   }));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({
                              1,
                              0,
                              1,
                              1,
                          }));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

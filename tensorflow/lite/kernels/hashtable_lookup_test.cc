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

#include <stdint.h>

#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
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

  void SetStringOffsets(int index, int32_t start_offset, int32_t end_offset) {
    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int32_t* offsets = reinterpret_cast<int32_t*>(tensor->data.raw);
    offsets[index + 1] = start_offset;
    offsets[index + 2] = end_offset;
  }

  void SetHashtableValue(const std::function<float(int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    for (int i = 0; i < rows; i++) {
      GetTensorData<float>(tensor)[i] = function(i);
    }
  }

  void SetHashtableValue(const std::function<float(int, int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    int features = tensor->dims->data[1];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < features; j++) {
        GetTensorData<float>(tensor)[i * features + j] = function(i, j);
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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

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

TEST(HashtableLookupOpTest, TestExtremeInt32KeysWithoutOverflow) {
  constexpr int kMin = std::numeric_limits<int32_t>::min();
  constexpr int kMax = std::numeric_limits<int32_t>::max();
  HashtableLookupOpModel m({5}, {3}, {3}, TensorType_FLOAT32);

  m.SetLookup({kMax, kMin, 42, 0, kMin + 1});
  m.SetHashtableKey({kMin, 0, kMax});
  m.SetHashtableValue([](int i) { return static_cast<float>(i + 1); });

  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({3.0f, 1.0f, 0.0f, 2.0f, 0.0f})));
  EXPECT_THAT(m.GetHit(), ElementsAreArray({1, 1, 0, 1, 0}));
}

TEST(HashtableLookupOpTest, EmptyTensorBoundaryCases) {
  // Empty lookup [0] with non-empty float table [3, 2] succeeds.
  {
    HashtableLookupOpModel m({0}, {3}, {3, 2}, TensorType_FLOAT32);
    m.SetHashtableKey({-11, 0, 1234});
    m.SetHashtableValue([](int i, int j) { return i + j / 10.0f; });
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutput().empty());
    EXPECT_TRUE(m.GetHit().empty());
  }

  // Empty lookup [0] with non-empty string table [3] succeeds.
  {
    HashtableLookupOpModel m({0}, {3}, {3}, TensorType_STRING);
    m.SetHashtableKey({-11, 0, 1234});
    m.SetHashtableValue({"Hello", "", "Hi"});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetStringOutput().empty());
    EXPECT_TRUE(m.GetHit().empty());
  }

  // Empty table [0, 2] with non-empty lookup [1] fails cleanly.
  {
    HashtableLookupOpModel m({1}, {0}, {0, 2}, TensorType_FLOAT32);
    m.SetLookup({0});
    EXPECT_EQ(m.Invoke(), kTfLiteError);
  }

  // Zero-feature table [3, 0] with both hit and miss succeeds without UB.
  {
    HashtableLookupOpModel m({2}, {3}, {3, 0}, TensorType_FLOAT32);
    m.SetLookup({1234, 999});
    m.SetHashtableKey({-11, 0, 1234});
    ASSERT_EQ(m.Invoke(), kTfLiteOk);
    EXPECT_TRUE(m.GetOutput().empty());
    EXPECT_THAT(m.GetHit(), ElementsAreArray({1, 0}));
  }
}

TEST(HashtableLookupOpTest, CorruptedStringTensorOffsetsRejected) {
  HashtableLookupOpModel m({1}, {1}, {1}, TensorType_STRING);
  m.SetLookup({10});
  m.SetHashtableKey({10});
  m.SetHashtableValue({"ok"});

  // Out-of-bounds end_offset (> value.bytes).
  m.SetStringOffsets(0, /*start_offset=*/12, /*end_offset=*/999);
  EXPECT_EQ(m.Invoke(), kTfLiteError);

  // Inverted offsets (end_offset < start_offset).
  m.SetStringOffsets(0, /*start_offset=*/12, /*end_offset=*/8);
  EXPECT_EQ(m.Invoke(), kTfLiteError);

  // Negative start_offset.
  m.SetStringOffsets(0, /*start_offset=*/-4, /*end_offset=*/12);
  EXPECT_EQ(m.Invoke(), kTfLiteError);
}

}  // namespace
}  // namespace tflite

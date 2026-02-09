// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_text/core/kernels/ngrams_tflite.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class NgramsModel : public SingleOpModel {
 public:
  // Constructor for testing the op with a tf.Tensor
  NgramsModel(int width, const std::string& string_separator,
              const std::vector<std::string>& input_values,
              const std::vector<int>& input_shape) {
    input_values_ = AddInput(TensorType_STRING);
    output_values_ = AddOutput(TensorType_STRING);

    BuildCustomOp(width, string_separator);

    BuildInterpreter({input_shape});
    PopulateStringTensor(input_values_, input_values);
    Invoke();
  }

  // Constructor for the op with a tf.RaggedTensor
  // Note: This interface uses row_lengths, as they're closer to the
  // dimensions in a TensorShape, but internally everything is row_splits.
  NgramsModel(int width, const std::string& string_separator,
              const std::vector<std::string>& input_values,
              const std::vector<std::vector<int64_t>> nested_row_lengths) {
    std::vector<std::vector<int>> input_shapes;
    input_shapes.reserve(nested_row_lengths.size() + 1);

    input_values_ = AddInput(TensorType_STRING);
    input_shapes.push_back({static_cast<int>(input_values.size())});
    output_values_ = AddOutput(TensorType_STRING);

    input_row_splits_.reserve(nested_row_lengths.size());
    output_row_splits_.reserve(nested_row_lengths.size());
    for (int i = 0; i < nested_row_lengths.size(); ++i) {
      input_row_splits_.push_back(AddInput(TensorType_INT64));
      input_shapes.push_back(
          {static_cast<int>(nested_row_lengths[i].size() + 1)});
      output_row_splits_.push_back(AddOutput(TensorType_INT64));
    }

    BuildCustomOp(width, string_separator);

    BuildInterpreter(input_shapes);
    PopulateStringTensor(input_values_, input_values);
    for (int i = 0; i < nested_row_lengths.size(); ++i) {
      std::vector<int64_t> row_splits;
      row_splits.reserve(nested_row_lengths[i].size() + 1);
      int64_t index = 0;
      row_splits.push_back(index);
      for (int64_t row_length : nested_row_lengths[i]) {
        index += row_length;
        row_splits.push_back(index);
      }
      PopulateTensor(input_row_splits_[i], row_splits);
    }
    Invoke();
  }

  std::vector<int> GetValuesTensorShape() {
    return GetTensorShape(output_values_);
  }

  std::vector<std::string> ExtractValuesTensorVector() {
    std::vector<std::string> r;
    TfLiteTensor* tensor = interpreter_->tensor(output_values_);
    int n = GetStringCount(tensor);
    for (int i = 0; i < n; ++i) {
      StringRef ref = GetString(tensor, i);
      r.emplace_back(ref.str, ref.len);
    }
    return r;
  }

  int GetNumNestedRowLengths() { return output_row_splits_.size(); }

  std::vector<int> GetRowLengthsTensorShape(int i) {
    std::vector<int> shape = GetTensorShape(output_row_splits_[i]);
    --shape[0];
    return shape;
  }

  std::vector<int64_t> ExtractRowLengthsTensorVector(int i) {
    std::vector<int64_t> row_splits =
        ExtractVector<int64_t>(output_row_splits_[i]);
    std::vector<int64_t> row_lengths;
    row_lengths.reserve(row_splits.size() - 1);
    int64_t head = row_splits[0];
    for (int i = 1; i < row_splits.size(); ++i) {
      int64_t tail = row_splits[i];
      row_lengths.push_back(tail - head);
      head = tail;
    }
    return row_lengths;
  }

 private:
  void BuildCustomOp(int width, const std::string& string_separator) {
    flexbuffers::Builder fbb;
    size_t start_map = fbb.StartMap();
    fbb.Int("width", width);
    fbb.String("string_separator", string_separator);
    fbb.Int("axis", -1);
    fbb.String("reduction_type", "STRING_JOIN");
    fbb.EndMap(start_map);
    fbb.Finish();

    SetCustomOp("TFText>NgramsStringJoin", fbb.GetBuffer(),
                Register_TFText_NgramsStringJoin);
  }

  int input_values_;
  std::vector<int> input_row_splits_;
  int output_values_;
  std::vector<int> output_row_splits_;
};

TEST(NgramsTest, TensorSingleSequenceWidthTwo) {
  NgramsModel m(2, " ", {"this", "is", "a", "test"}, std::vector<int>{4});
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(3));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAre("this is", "is a", "a test"));
}

TEST(NgramsTest, TensorSingleSequenceWidthThree) {
  NgramsModel m(3, " ", {"this", "is", "a", "test"}, std::vector<int>{4});
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(2));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAre("this is a", "is a test"));
}

TEST(NgramsTest, TensorSingleSequenceLongerSeparator) {
  NgramsModel m(2, "...", {"this", "is", "a", "test"}, std::vector<int>{4});
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(3));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAre("this...is", "is...a", "a...test"));
}

TEST(NgramsTest, TensorSingleSequenceWidthTooLong) {
  NgramsModel m(5, " ", {"this", "is", "a", "test"}, std::vector<int>{4});
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(0));
  EXPECT_THAT(m.ExtractValuesTensorVector(), ElementsAre());
}

TEST(NgramsTest, TensorMultidimensionalInputWidthTwo) {
  NgramsModel m(2, " ",
                {
                    "0,0,0", "0,0,1", "0,0,2", "0,0,3",  //
                    "0,1,0", "0,1,1", "0,1,2", "0,1,3",  //
                    "0,2,0", "0,2,1", "0,2,2", "0,2,3",  //
                    "1,0,0", "1,0,1", "1,0,2", "1,0,3",  //
                    "1,1,0", "1,1,1", "1,1,2", "1,1,3",  //
                    "1,2,0", "1,2,1", "1,2,2", "1,2,3",  //
                },
                std::vector<int>{2, 3, 4});
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(2, 3, 3));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAreArray({
                  "0,0,0 0,0,1", "0,0,1 0,0,2", "0,0,2 0,0,3",  //
                  "0,1,0 0,1,1", "0,1,1 0,1,2", "0,1,2 0,1,3",  //
                  "0,2,0 0,2,1", "0,2,1 0,2,2", "0,2,2 0,2,3",  //
                  "1,0,0 1,0,1", "1,0,1 1,0,2", "1,0,2 1,0,3",  //
                  "1,1,0 1,1,1", "1,1,1 1,1,2", "1,1,2 1,1,3",  //
                  "1,2,0 1,2,1", "1,2,1 1,2,2", "1,2,2 1,2,3",  //
              }));
}

TEST(NgramsTest, RaggedTensorSingleSequenceWidthTwo) {
  std::vector<std::vector<int64_t>> nested_row_lengths;
  nested_row_lengths.push_back({4});
  NgramsModel m(2, " ", {"this", "is", "a", "test"},
                nested_row_lengths);
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(3));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAre("this is", "is a", "a test"));
  ASSERT_THAT(m.GetNumNestedRowLengths(), 1);
  EXPECT_THAT(m.GetRowLengthsTensorShape(0), ElementsAre(1));
  EXPECT_THAT(m.ExtractRowLengthsTensorVector(0), ElementsAre(3));
}

TEST(NgramsTest, RaggedTensorSingleSequenceWidthThree) {
  std::vector<std::vector<int64_t>> nested_row_lengths;
  nested_row_lengths.push_back({4});
  NgramsModel m(3, " ", {"this", "is", "a", "test"}, nested_row_lengths);
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(2));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAre("this is a", "is a test"));
  ASSERT_THAT(m.GetNumNestedRowLengths(), 1);
  EXPECT_THAT(m.GetRowLengthsTensorShape(0), ElementsAre(1));
  EXPECT_THAT(m.ExtractRowLengthsTensorVector(0), ElementsAre(2));
}

TEST(NgramsTest, RaggedTensorSingleSequenceLongerSeparator) {
  std::vector<std::vector<int64_t>> nested_row_lengths;
  nested_row_lengths.push_back({4});
  NgramsModel m(2, "<>", {"this", "is", "a", "test"}, nested_row_lengths);
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(3));
  EXPECT_THAT(m.ExtractValuesTensorVector(),
              ElementsAre("this<>is", "is<>a", "a<>test"));
  ASSERT_THAT(m.GetNumNestedRowLengths(), 1);
  EXPECT_THAT(m.GetRowLengthsTensorShape(0), ElementsAre(1));
  EXPECT_THAT(m.ExtractRowLengthsTensorVector(0), ElementsAre(3));
}

TEST(NgramsTest, RaggedTensorSingleSequenceWidthTooLong) {
  std::vector<std::vector<int64_t>> nested_row_lengths;
  nested_row_lengths.push_back({4});
  NgramsModel m(5, " ", {"this", "is", "a", "test"}, nested_row_lengths);
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(0));
  EXPECT_THAT(m.ExtractValuesTensorVector(), ElementsAre());
  ASSERT_THAT(m.GetNumNestedRowLengths(), 1);
  EXPECT_THAT(m.GetRowLengthsTensorShape(0), ElementsAre(1));
  EXPECT_THAT(m.ExtractRowLengthsTensorVector(0), ElementsAre(0));
}

TEST(NgramsTest, RaggedTensorMultidimensionalInputWidthTwo) {
  std::vector<std::vector<int64_t>> nested_row_lengths;
  nested_row_lengths.push_back({4, 2, 1});
  nested_row_lengths.push_back({5, 4, 3, 2, 2, 3, 4, 6});
  NgramsModel m(2, " ",
                {
                    "0,0,0", "0,0,1", "0,0,2", "0,0,3", "0,0,4",           //
                    "0,1,0", "0,1,1", "0,1,2", "0,1,3",                    //
                    "0,2,0", "0,2,1", "0,2,2",                             //
                    "0,3,0", "0,3,1",                                      //
                    "1,0,0", "1,0,1",                                      //
                    "1,1,0", "1,1,1", "1,1,2",                             //
                    "1,2,0", "1,2,1", "1,2,2", "1,2,3",                    //
                    "2,0,0", "2,0,1", "2,0,2", "2,0,3", "2,0,4", "2,0,5",  //
                },
                nested_row_lengths);

  std::vector<std::string> expected_values = {
      "0,0,0 0,0,1", "0,0,1 0,0,2", "0,0,2 0,0,3", "0,0,3 0,0,4",  //
      "0,1,0 0,1,1", "0,1,1 0,1,2", "0,1,2 0,1,3",                 //
      "0,2,0 0,2,1", "0,2,1 0,2,2",                                //
      "0,3,0 0,3,1",                                               //
      "1,0,0 1,0,1",                                               //
      "1,1,0 1,1,1", "1,1,1 1,1,2",                                //
      "1,2,0 1,2,1", "1,2,1 1,2,2", "1,2,2 1,2,3",                 //
      "2,0,0 2,0,1", "2,0,1 2,0,2", "2,0,2 2,0,3", "2,0,3 2,0,4",
      "2,0,4 2,0,5",  //
  };
  EXPECT_THAT(m.GetValuesTensorShape(), ElementsAre(expected_values.size()));
  EXPECT_THAT(m.ExtractValuesTensorVector(), ElementsAreArray(expected_values));
  ASSERT_THAT(m.GetNumNestedRowLengths(), 2);
  EXPECT_THAT(m.GetRowLengthsTensorShape(0), ElementsAre(3));
  EXPECT_THAT(m.ExtractRowLengthsTensorVector(0), ElementsAre(4, 2, 1));
  EXPECT_THAT(m.GetRowLengthsTensorShape(1), ElementsAre(8));
  EXPECT_THAT(m.ExtractRowLengthsTensorVector(1),
              ElementsAre(4, 3, 2, 1, 1, 2, 3, 5));
}

}  // namespace
}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite

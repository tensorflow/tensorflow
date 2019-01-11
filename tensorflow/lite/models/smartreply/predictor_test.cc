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

#include "tensorflow/lite/models/smartreply/predictor.h"

#include <fstream>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace custom {
namespace smartreply {
namespace {

const char kSamples[] = "smartreply_samples.tsv";

string GetModelFilePath() {
  return "third_party/tensorflow/lite/models/testdata/"
         "smartreply_ondevice_model.bin";
}

string GetSamplesFilePath() {
  return string(absl::StrCat(tensorflow::testing::TensorFlowSrcRoot(), "/",
                             "lite/models/testdata/", kSamples));
}

MATCHER_P(IncludeAnyResponesIn, expected_response, "contains the response") {
  bool has_expected_response = false;
  for (const auto &item : *arg) {
    const string &response = item.GetText();
    if (expected_response.find(response) != expected_response.end()) {
      has_expected_response = true;
      break;
    }
  }
  return has_expected_response;
}

class PredictorTest : public ::testing::Test {
 protected:
  PredictorTest() {}
  ~PredictorTest() override {}

  void SetUp() override {
    model_ = tflite::FlatBufferModel::BuildFromFile(GetModelFilePath().c_str());
    ASSERT_NE(model_.get(), nullptr);
  }

  std::unique_ptr<::tflite::FlatBufferModel> model_;
};

TEST_F(PredictorTest, GetSegmentPredictions) {
  std::vector<PredictorResponse> predictions;

  GetSegmentPredictions({"Welcome"}, *model_, /*config=*/{{}}, &predictions);
  EXPECT_GT(predictions.size(), 0);

  float max = 0;
  for (const auto &item : predictions) {
    if (item.GetScore() > max) {
      max = item.GetScore();
    }
  }

  EXPECT_GT(max, 0.3);
  EXPECT_THAT(
      &predictions,
      IncludeAnyResponesIn(std::unordered_set<string>({"Thanks very much"})));
}

TEST_F(PredictorTest, TestTwoSentences) {
  std::vector<PredictorResponse> predictions;

  GetSegmentPredictions({"Hello", "How are you?"}, *model_, /*config=*/{{}},
                        &predictions);
  EXPECT_GT(predictions.size(), 0);

  float max = 0;
  for (const auto &item : predictions) {
    if (item.GetScore() > max) {
      max = item.GetScore();
    }
  }

  EXPECT_GT(max, 0.3);
  EXPECT_THAT(&predictions, IncludeAnyResponesIn(std::unordered_set<string>(
                                {"Hi, how are you doing?"})));
}

TEST_F(PredictorTest, TestBackoff) {
  std::vector<PredictorResponse> predictions;

  GetSegmentPredictions({"你好"}, *model_, /*config=*/{{}}, &predictions);
  EXPECT_EQ(predictions.size(), 0);

  // Backoff responses are returned in order.
  GetSegmentPredictions({"你好"}, *model_, /*config=*/{{"Yes", "Ok"}},
                        &predictions);
  EXPECT_EQ(predictions.size(), 2);
  EXPECT_EQ(predictions[0].GetText(), "Yes");
  EXPECT_EQ(predictions[1].GetText(), "Ok");
}

TEST_F(PredictorTest, BatchTest) {
  int total_items = 0;
  int total_responses = 0;
  int total_triggers = 0;

  string line;
  std::ifstream fin(GetSamplesFilePath());
  while (std::getline(fin, line)) {
    const std::vector<string> fields = absl::StrSplit(line, '\t');
    if (fields.empty()) {
      continue;
    }

    // Parse sample file and predict
    const string &msg = fields[0];
    std::vector<PredictorResponse> predictions;
    GetSegmentPredictions({msg}, *model_, /*config=*/{{}}, &predictions);

    // Validate response and generate stats.
    total_items++;
    total_responses += predictions.size();
    if (!predictions.empty()) {
      total_triggers++;
    }
    EXPECT_THAT(&predictions, IncludeAnyResponesIn(std::unordered_set<string>(
                                  fields.begin() + 1, fields.end())));
  }

  EXPECT_EQ(total_triggers, total_items);
  EXPECT_GE(total_responses, total_triggers);
}

}  // namespace
}  // namespace smartreply
}  // namespace custom
}  // namespace tflite

int main(int argc, char **argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

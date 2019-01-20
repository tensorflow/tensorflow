/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
// Unit test for speech models (Hotword, SpeakerId) using TFLite Ops.

#include <memory>
#include <string>

#include <fstream>

#include "testing/base/public/googletest.h"
#include <gtest/gtest.h>
#include "tensorflow/lite/testing/parse_testdata.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace {

const char kDataPath[] = "third_party/tensorflow/lite/models/testdata/";

bool Init(const string& in_file_name, testing::TfLiteDriver* driver,
          std::ifstream* in_file) {
  driver->SetModelBaseDir(kDataPath);
  in_file->open(string(kDataPath) + in_file_name, std::ifstream::in);
  return in_file->is_open();
}

// Converts a set of test files provided by the speech team into a single
// test_spec. Input CSV files are supposed to contain a number of sequences per
// line. Each sequence maps to a single invocation of the interpreter and the
// output tensor after all sequences have run is compared to the corresponding
// line in the output CSV file.
bool ConvertCsvData(const string& model_name, const string& in_name,
                    const string& out_name, const string& input_tensor,
                    const string& output_tensor,
                    const string& persistent_tensors, int sequence_size,
                    std::ostream* out) {
  auto data_path = [](const string& s) { return string(kDataPath) + s; };

  *out << "load_model: \"" << data_path(model_name) << "\"" << std::endl;

  *out << "init_state: \"" << persistent_tensors << "\"" << std::endl;

  string in_file_name = data_path(in_name);
  std::ifstream in_file(in_file_name);
  if (!in_file.is_open()) {
    std::cerr << "Failed to open " << in_file_name << std::endl;
    return false;
  }
  string out_file_name = data_path(out_name);
  std::ifstream out_file(out_file_name);
  if (!out_file.is_open()) {
    std::cerr << "Failed to open " << out_file_name << std::endl;
    return false;
  }

  int invocation_count = 0;
  string in_values;
  while (std::getline(in_file, in_values, '\n')) {
    std::vector<string> input = testing::Split<string>(in_values, ",");
    int num_sequences = input.size() / sequence_size;

    for (int j = 0; j < num_sequences; ++j) {
      *out << "invoke {" << std::endl;
      *out << "  id: " << invocation_count << std::endl;
      *out << "  input: \"";
      for (int k = 0; k < sequence_size; ++k) {
        *out << input[k + j * sequence_size] << ",";
      }
      *out << "\"" << std::endl;

      if (j == num_sequences - 1) {
        string out_values;
        if (!std::getline(out_file, out_values, '\n')) {
          std::cerr << "Not enough lines in " << out_file_name << std::endl;
          return false;
        }
        *out << "  output: \"" << out_values << "\"" << std::endl;
      }

      *out << "}" << std::endl;
      ++invocation_count;
    }
  }
  return true;
}

class SpeechTest : public ::testing::TestWithParam<int> {
 protected:
  int GetMaxInvocations() { return GetParam(); }
};

TEST_P(SpeechTest, DISABLED_HotwordOkGoogleRank1Test) {
  std::stringstream os;
  ASSERT_TRUE(ConvertCsvData(
      "speech_hotword_model_rank1.tflite", "speech_hotword_model_in.csv",
      "speech_hotword_model_out_rank1.csv", /*input_tensor=*/"0",
      /*output_tensor=*/"18", /*persistent_tensors=*/"4",
      /*sequence_size=*/40, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

TEST_P(SpeechTest, DISABLED_HotwordOkGoogleRank2Test) {
  std::stringstream os;
  ASSERT_TRUE(ConvertCsvData(
      "speech_hotword_model_rank2.tflite", "speech_hotword_model_in.csv",
      "speech_hotword_model_out_rank2.csv", /*input_tensor=*/"17",
      /*output_tensor=*/"18", /*persistent_tensors=*/"1",
      /*sequence_size=*/40, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

TEST_P(SpeechTest, DISABLED_SpeakerIdOkGoogleTest) {
  std::stringstream os;
  ASSERT_TRUE(ConvertCsvData(
      "speech_speakerid_model.tflite", "speech_speakerid_model_in.csv",
      "speech_speakerid_model_out.csv", /*input_tensor=*/"0",
      /*output_tensor=*/"63",
      /*persistent_tensors=*/"18,19,38,39,58,59",
      /*sequence_size=*/80, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

TEST_P(SpeechTest, AsrAmTest) {
  std::stringstream os;
  ASSERT_TRUE(
      ConvertCsvData("speech_asr_am_model.tflite", "speech_asr_am_model_in.csv",
                     "speech_asr_am_model_out.csv", /*input_tensor=*/"0",
                     /*output_tensor=*/"104",
                     /*persistent_tensors=*/"18,19,38,39,58,59,78,79,98,99",
                     /*sequence_size=*/320, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

TEST_P(SpeechTest, AsrAmQuantizedTest) {
  std::stringstream os;
  ASSERT_TRUE(ConvertCsvData(
      "speech_asr_am_model_int8.tflite", "speech_asr_am_model_in.csv",
      "speech_asr_am_model_int8_out.csv", /*input_tensor=*/"0",
      /*output_tensor=*/"104",
      /*persistent_tensors=*/"18,19,38,39,58,59,78,79,98,99",
      /*sequence_size=*/320, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

// The original version of speech_asr_lm_model_test.cc ran a few sequences
// through the interpreter and stored the sum of all the output, which was them
// compared for correctness. In this test we are comparing all the intermediate
// results.
TEST_P(SpeechTest, DISABLED_AsrLmTest) {
  std::ifstream in_file;
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(Init("speech_asr_lm_model.test_spec", &test_driver, &in_file));
  ASSERT_TRUE(
      testing::ParseAndRunTests(&in_file, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

TEST_P(SpeechTest, DISABLED_EndpointerTest) {
  std::stringstream os;
  ASSERT_TRUE(ConvertCsvData(
      "speech_endpointer_model.tflite", "speech_endpointer_model_in.csv",
      "speech_endpointer_model_out.csv", /*input_tensor=*/"0",
      /*output_tensor=*/"56",
      /*persistent_tensors=*/"27,28,47,48",
      /*sequence_size=*/320, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

TEST_P(SpeechTest, DISABLED_TtsTest) {
  std::stringstream os;
  ASSERT_TRUE(ConvertCsvData("speech_tts_model.tflite",
                             "speech_tts_model_in.csv",
                             "speech_tts_model_out.csv", /*input_tensor=*/"0",
                             /*output_tensor=*/"71",
                             /*persistent_tensors=*/"24,25,44,45,64,65,70",
                             /*sequence_size=*/334, &os));
  testing::TfLiteDriver test_driver(/*use_nnapi=*/false);
  ASSERT_TRUE(testing::ParseAndRunTests(&os, &test_driver, GetMaxInvocations()))
      << test_driver.GetErrorMessage();
}

// Define two instantiations. The "ShortTests" instantiations is used when
// running the tests on Android, in order to prevent timeouts (It takes about
// 200s just to bring up the Android emulator.)
static const int kAllInvocations = -1;
static const int kFirstFewInvocations = 10;
INSTANTIATE_TEST_SUITE_P(LongTests, SpeechTest,
                         ::testing::Values(kAllInvocations));
INSTANTIATE_TEST_SUITE_P(ShortTests, SpeechTest,
                         ::testing::Values(kFirstFewInvocations));

}  // namespace
}  // namespace tflite

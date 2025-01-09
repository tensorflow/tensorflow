/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/toco/model_cmdline_flags.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/toco/args.h"

namespace toco {
namespace {

TEST(ModelCmdlineFlagsTest, ParseArgsStringMapList) {
  int args_count = 3;
  const char* args[] = {
      "toco", "--input_arrays=input_1",
      "--rnn_states={state_array:rnn/BasicLSTMCellZeroState/zeros,"
      "back_edge_source_array:rnn/basic_lstm_cell/Add_1,size:4},"
      "{state_array:rnn/BasicLSTMCellZeroState/zeros_1,"
      "back_edge_source_array:rnn/basic_lstm_cell/Mul_2,size:4}",
      nullptr};

  std::string expected_input_arrays = "input_1";
  std::vector<std::unordered_map<std::string, std::string>> expected_rnn_states;
  expected_rnn_states.push_back(
      {{"state_array", "rnn/BasicLSTMCellZeroState/zeros"},
       {"back_edge_source_array", "rnn/basic_lstm_cell/Add_1"},
       {"size", "4"}});
  expected_rnn_states.push_back(
      {{"state_array", "rnn/BasicLSTMCellZeroState/zeros_1"},
       {"back_edge_source_array", "rnn/basic_lstm_cell/Mul_2"},
       {"size", "4"}});

  std::string message;
  ParsedModelFlags result_flags;

  EXPECT_TRUE(ParseModelFlagsFromCommandLineFlags(
      &args_count, const_cast<char**>(args), &message, &result_flags));
  EXPECT_EQ(result_flags.input_arrays.value(), expected_input_arrays);
  EXPECT_EQ(result_flags.rnn_states.value().elements, expected_rnn_states);
}

}  // namespace
}  // namespace toco

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  ::toco::port::InitGoogleWasDoneElsewhere();
  return RUN_ALL_TESTS();
}

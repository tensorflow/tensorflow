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
#include "tensorflow/lite/tools/optimize/calibration/calibrator.h"

#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace calibration {
namespace {

std::unique_ptr<FlatBufferModel> ReadModel(const string& model_name) {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir, model_name);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

TEST(CalibratorTest, CalibrationStatsAreCollected) {
  auto model = ReadModel("multi_add.bin");
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status = BuildLoggingInterpreter(
      *model, ops::builtin::BuiltinOpResolver{}, &interpreter, &reader);
  EXPECT_EQ(kTfLiteOk, status);

  ASSERT_TRUE(interpreter);
  ASSERT_TRUE(reader);
  absl::flat_hash_map<int, CalibrationReader::CalibrationStats> stats;
  status = reader->GetTensorStatsAsMap(&stats);
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_TRUE(stats.empty());

  status = interpreter->AllocateTensors();
  ASSERT_EQ(kTfLiteOk, status);
  // Model does the following:
  // 0        1       2        3
  // |        |__ ____|        |
  // |           |             |
  // |          Add(tensor:4)  |
  // |____ ______|______ ______|
  //      |             |
  //      Add          Add
  //      |             |
  //    Output:5      Output:6

  const size_t tensor_size = 1 * 8 * 8 * 3;

  std::vector<float> ones(tensor_size, 1.0f);
  // Fill input tensor i with i+1, i.e. input[0] = 1.0f, input[1] = 2.0f,
  // input[2] = 3.0f

  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_tensor_idx = interpreter->inputs()[i];
    TfLiteTensor* tensor = interpreter->tensor(input_tensor_idx);
    ASSERT_EQ(tensor->bytes, tensor_size * sizeof(float));
    for (size_t j = 0; j < tensor_size; j++) {
      tensor->data.f[j] = i + 1;
    }
  }
  status = interpreter->Invoke();
  ASSERT_EQ(kTfLiteOk, status);
  const float eps = 1e-6f;
  // Verify that tensor 5: is 6
  // Verify that tensor 6: is 9
  TfLiteTensor* tensor = interpreter->tensor(interpreter->outputs()[0]);
  for (size_t i = 0; i < tensor_size; i++) {
    EXPECT_NEAR(tensor->data.f[i], 6.0f, eps);
  }
  tensor = interpreter->tensor(interpreter->outputs()[1]);
  for (size_t i = 0; i < tensor_size; i++) {
    EXPECT_NEAR(tensor->data.f[i], 9.0f, eps);
  }

  // Verify that min max of tensors.
  status = reader->GetTensorStatsAsMap(&stats);
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_EQ(7, stats.size());
  // Check inputs
  for (int tensor_idx = 0; tensor_idx < 4; tensor_idx++) {
    EXPECT_NEAR(stats.at(tensor_idx).min, tensor_idx + 1, eps);
    EXPECT_NEAR(stats.at(tensor_idx).max, tensor_idx + 1, eps);
  }
  // Check tensor 4 max.
  EXPECT_NEAR(stats.at(4).min, 5, eps);
  EXPECT_NEAR(stats.at(4).max, 5, eps);

  // Check outputs
  EXPECT_NEAR(stats.at(5).min, 6, eps);
  EXPECT_NEAR(stats.at(5).max, 6, eps);

  EXPECT_NEAR(stats.at(6).min, 9, eps);
  EXPECT_NEAR(stats.at(6).max, 9, eps);
}

TEST(CalibratorTest, MultipleInvokes) {
  auto model = ReadModel("multi_add.bin");
  ASSERT_TRUE(model);
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status = BuildLoggingInterpreter(
      *model, ops::builtin::BuiltinOpResolver{}, &interpreter, &reader);
  EXPECT_EQ(kTfLiteOk, status);

  ASSERT_TRUE(interpreter);
  ASSERT_TRUE(reader);
  status = interpreter->AllocateTensors();

  EXPECT_EQ(kTfLiteOk, status);
  const size_t tensor_size = 1 * 8 * 8 * 3;
  // Fill input tensor i with i+1, i.e. input[0] = 1.0f, input[1] = 2.0f,
  // input[2] = 3.0f

  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_tensor_idx = interpreter->inputs()[i];
    TfLiteTensor* tensor = interpreter->tensor(input_tensor_idx);
    ASSERT_EQ(tensor->bytes, tensor_size * sizeof(float));
    for (size_t j = 0; j < tensor_size; j++) {
      tensor->data.f[j] = i + 1;
    }
  }
  status = interpreter->Invoke();
  ASSERT_EQ(kTfLiteOk, status);
  const float eps = 1e-6f;
  // Verify that min max of tensors.
  absl::flat_hash_map<int, CalibrationReader::CalibrationStats> stats;
  status = reader->GetTensorStatsAsMap(&stats);
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_EQ(7, stats.size());
  const float expected_values[7] = {
      1.0f,  // input 0
      2.0f,  // input 1
      3.0f,  // input 2
      4.0f,  // input 3
      5.0f,  // Add(1, 2)
      6.0f,  // Output 5: Add(0, Add(1,2))
      9.0f,  // Output 6: Add(Add(1,2), 3)
  };
  for (int tensor_idx = 0; tensor_idx < 7; tensor_idx++) {
    EXPECT_NEAR(stats.at(tensor_idx).min, expected_values[tensor_idx], eps);
    EXPECT_NEAR(stats.at(tensor_idx).max, expected_values[tensor_idx], eps);
  }
  // Set input[0][0] = 1.5 and input[0][1] = 0.5 this should change the values
  // only for input[0] and tensor 4 and outputs 5, 6.
  TfLiteTensor* input0 = interpreter->tensor(0);
  input0->data.f[0] = 1.5f;
  input0->data.f[1] = 0.5f;
  status = interpreter->Invoke();
  ASSERT_EQ(kTfLiteOk, status);
  status = reader->GetTensorStatsAsMap(&stats);
  EXPECT_EQ(kTfLiteOk, status);
  EXPECT_EQ(7, stats.size());
  EXPECT_NEAR(stats.at(0).min, 0.5f, eps);
  EXPECT_NEAR(stats.at(0).max, 1.5f, eps);

  for (int tensor_idx = 1; tensor_idx < 5; tensor_idx++) {
    EXPECT_NEAR(stats.at(tensor_idx).min, expected_values[tensor_idx], eps);
    EXPECT_NEAR(stats.at(tensor_idx).max, expected_values[tensor_idx], eps);
  }

  EXPECT_NEAR(stats.at(5).min, 5.5f, eps);
  EXPECT_NEAR(stats.at(5).max, 6.5f, eps);

  EXPECT_NEAR(stats.at(6).min, 9.0f, eps);
  EXPECT_NEAR(stats.at(6).max, 9.0f, eps);
}

TEST(CalibratorTest, UpdateMinMax) {
  auto flatbuffer_model = ReadModel("multi_add.bin");
  ASSERT_TRUE(flatbuffer_model);
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status = BuildLoggingInterpreter(*flatbuffer_model,
                                        ops::builtin::BuiltinOpResolver{},
                                        &interpreter, &reader);
  EXPECT_EQ(kTfLiteOk, status);
  auto readonly_model = flatbuffer_model->GetModel();
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);

  ASSERT_TRUE(interpreter);
  ASSERT_TRUE(reader);
  status = interpreter->AllocateTensors();

  EXPECT_EQ(kTfLiteOk, status);
  const size_t tensor_size = 1 * 8 * 8 * 3;
  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_tensor_idx = interpreter->inputs()[i];
    TfLiteTensor* tensor = interpreter->tensor(input_tensor_idx);
    ASSERT_EQ(tensor->bytes, tensor_size * sizeof(float));
    for (size_t j = 0; j < tensor_size; j++) {
      tensor->data.f[j] = i + 1;
    }
  }
  auto input_0_quant_params =
      absl::make_unique<tflite::QuantizationParametersT>();
  input_0_quant_params->min.push_back(0.5);
  input_0_quant_params->max.push_back(1.5);
  model.subgraphs[0]->tensors[0]->quantization =
      std::move(input_0_quant_params);

  // Invoke with update == true.
  status = interpreter->Invoke();
  ASSERT_EQ(kTfLiteOk, status);
  const float eps = 1e-6f;
  // Verify that min max of tensors.
  const float expected_min[7] = {
      0.5f,  // input 0
      2.0f,  // input 1
      3.0f,  // input 2
      4.0f,  // input 3
      5.0f,  // Add(1, 2)
      6.0f,  // Output 5: Add(0, Add(1,2))
      9.0f,  // Output 6: Add(Add(1,2), 3)
  };
  const float expected_max[7] = {
      1.5f,  // input 0
      2.0f,  // input 1
      3.0f,  // input 2
      4.0f,  // input 3
      5.0f,  // Add(1, 2)
      6.0f,  // Output 5: Add(0, Add(1,2))
      9.0f,  // Output 6: Add(Add(1,2), 3)
  };
  status = reader->AddCalibrationToModel(&model, /*update=*/true);
  for (int tensor_idx = 0; tensor_idx < 7; tensor_idx++) {
    EXPECT_NEAR(model.subgraphs[0]->tensors[tensor_idx]->quantization->min[0],
                expected_min[tensor_idx], eps);
    EXPECT_NEAR(model.subgraphs[0]->tensors[tensor_idx]->quantization->max[0],
                expected_max[tensor_idx], eps);
  }

  // Invoke with update == false;
  // Verify that min max of tensors.
  const float expected_value[7] = {
      1.0f,  // input 0
      2.0f,  // input 1
      3.0f,  // input 2
      4.0f,  // input 3
      5.0f,  // Add(1, 2)
      6.0f,  // Output 5: Add(0, Add(1,2))
      9.0f,  // Output 6: Add(Add(1,2), 3)
  };
  status = reader->AddCalibrationToModel(&model, /*update=*/false);
  for (int tensor_idx = 0; tensor_idx < 7; tensor_idx++) {
    EXPECT_NEAR(model.subgraphs[0]->tensors[tensor_idx]->quantization->min[0],
                expected_value[tensor_idx], eps);
    EXPECT_NEAR(model.subgraphs[0]->tensors[tensor_idx]->quantization->max[0],
                expected_value[tensor_idx], eps);
  }
}

TEST(CalibratorTest, LSTM) {
  auto flatbuffer_model = ReadModel("lstm.bin");
  ASSERT_TRUE(flatbuffer_model);
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status = BuildLoggingInterpreter(*flatbuffer_model,
                                        ops::builtin::BuiltinOpResolver{},
                                        &interpreter, &reader);
  EXPECT_EQ(status, kTfLiteOk);

  auto readonly_model = flatbuffer_model->GetModel();
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);

  ASSERT_TRUE(interpreter);
  ASSERT_TRUE(reader);
  status = interpreter->AllocateTensors();

  EXPECT_EQ(kTfLiteOk, status);
  const std::vector<float> lstm_input = {0.3, 0.2};
  int input_tensor_idx = interpreter->inputs()[0];
  TfLiteTensor* tensor = interpreter->tensor(input_tensor_idx);
  for (size_t j = 0; j < lstm_input.size(); j++) {
    tensor->data.f[j] = lstm_input[j];
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  absl::flat_hash_map<int, CalibrationReader::CalibrationStats> stats;
  EXPECT_EQ(reader->GetTensorStatsAsMap(&stats), kTfLiteOk);

  // Check the results.
  const float eps = 1e-6f;
  const std::unordered_map<int, CalibrationReader::CalibrationStats>
      expected_calibration_result = {
          // Input.
          {0, {0.200000, 0.300000}},
          // State.
          {18, {0.000000, 0.468415}},
          // State.
          {19, {0.000000, 0.424350}},
          // Output.
          {24, {0.265968, 0.468415}},
          // Intemediate_0.
          {25, {0.080045, 0.170588}},
          // Intemediate_1.
          {26, {0.080045, 0.170588}},
          // Intemediate_2.
          {27, {0.080045, 0.170588}},
          // Intemediate_3.
          {28, {0.080045, 0.170588}},
          // Intemediate_4.
          {29, {0.000000, 0.270944}},
      };
  EXPECT_EQ(expected_calibration_result.size(), stats.size());
  for (const auto& e : stats) {
    auto expected_result = expected_calibration_result.at(e.first);
    EXPECT_NEAR(e.second.min, expected_result.min, eps);
    EXPECT_NEAR(e.second.max, expected_result.max, eps);
  }
}

TEST(CalibratorTest, UnidirectionalSequenceLSTM) {
  auto flatbuffer_model = ReadModel("unidirectional_sequence_lstm.bin");
  ASSERT_TRUE(flatbuffer_model);
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status = BuildLoggingInterpreter(*flatbuffer_model,
                                        ops::builtin::BuiltinOpResolver{},
                                        &interpreter, &reader);
  EXPECT_EQ(kTfLiteOk, status);

  auto readonly_model = flatbuffer_model->GetModel();
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);

  ASSERT_TRUE(interpreter);
  ASSERT_TRUE(reader);
  EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  const std::vector<float> lstm_input = {0.3, 0.2, 0.9, 0.8};
  int input_tensor_idx = interpreter->inputs()[0];
  TfLiteTensor* tensor = interpreter->tensor(input_tensor_idx);
  for (size_t j = 0; j < lstm_input.size(); j++) {
    tensor->data.f[j] = lstm_input[j];
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  absl::flat_hash_map<int, CalibrationReader::CalibrationStats> stats;
  EXPECT_EQ(reader->GetTensorStatsAsMap(&stats), kTfLiteOk);

  // Check the results.
  const float eps = 1e-6f;
  const std::unordered_map<int, CalibrationReader::CalibrationStats>
      expected_calibration_result = {
          // Input.
          {0, {0.200000, 0.900000}},
          // State.
          {18, {0.000000, 0.520999}},
          // State.
          {19, {0.000000, 0.711364}},
          // Output.
          {24, {0.247992, 0.520999}},
          // Intemediate_0.
          {25, {0.080045, 0.824241}},
          // Intemediate_1.
          {26, {0.080045, 0.824241}},
          // Intemediate_2.
          {27, {0.080045, 0.824241}},
          // Intemediate_3.
          {28, {0.080045, 0.824241}},
          // Intemediate_4.
          {29, {0.000000, 0.413618}},
      };
  EXPECT_EQ(expected_calibration_result.size(), stats.size());
  for (const auto& e : stats) {
    auto expected_result = expected_calibration_result.at(e.first);
    EXPECT_NEAR(e.second.min, expected_result.min, eps);
    EXPECT_NEAR(e.second.max, expected_result.max, eps);
  }
}

TEST(CalibratorTest, CustomLSTM) {
  auto flatbuffer_model = ReadModel("custom_lstm.bin");
  ASSERT_TRUE(flatbuffer_model);
  std::unique_ptr<Interpreter> interpreter;
  std::unique_ptr<CalibrationReader> reader;
  auto status = BuildLoggingInterpreter(*flatbuffer_model,
                                        ops::builtin::BuiltinOpResolver{},
                                        &interpreter, &reader);
  EXPECT_EQ(kTfLiteOk, status);

  auto readonly_model = flatbuffer_model->GetModel();
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);

  ASSERT_TRUE(interpreter);
  ASSERT_TRUE(reader);
  EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  const std::vector<float> lstm_input = {0.3, 0.2, 0.9, 0.8};
  int input_tensor_idx = interpreter->inputs()[0];
  TfLiteTensor* tensor = interpreter->tensor(input_tensor_idx);
  for (size_t j = 0; j < lstm_input.size(); j++) {
    tensor->data.f[j] = lstm_input[j];
  }

  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

  absl::flat_hash_map<int, CalibrationReader::CalibrationStats> stats;
  EXPECT_EQ(reader->GetTensorStatsAsMap(&stats), kTfLiteOk);

  // Check the results.
  const float eps = 1e-6f;
  const std::unordered_map<int, CalibrationReader::CalibrationStats>
      expected_calibration_result = {
          // input.
          {0, {0.200000, 0.300000}},
          // state.
          {18, {0.000000, 0.468415}},
          // state.
          {19, {0.000000, 0.424349}},
          // output.
          {24, {0.265968, 0.468415}},
          // intermediate 0.
          {25, {0.080045, 0.170588}},
          // intermediate 1.
          {26, {0.080045, 0.170588}},
          // intermediate 2.
          {27, {0.000000, 0.000000}},
          // intermediate 3.
          {28, {0.080045, 0.170588}},
          // intermediate 4.
          {29, {0.080045, 0.170588}},
          // intermediate 5.
          {30, {0.000000, 0.000000}},
          // intermediate 6.
          {31, {0.080045, 0.170588}},
          // intermediate 7.
          {32, {0.080045, 0.170588}},
          // intermediate 8.
          {33, {0.000000, 0.000000}},
          // intermediate 9.
          {34, {0.080045, 0.170588}},
          // intermediate 10.
          {35, {0.080045, 0.170588}},
          // intermediate 11.
          {36, {0.000000, 0.000000}},
      };
  EXPECT_EQ(expected_calibration_result.size(), stats.size());
  for (const auto& e : stats) {
    auto expected_result = expected_calibration_result.at(e.first);
    EXPECT_NEAR(e.second.min, expected_result.min, eps);
    EXPECT_NEAR(e.second.max, expected_result.max, eps);
  }
}

}  // namespace
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  tensorflow::string model_file;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_model_file", &model_file,
                       "Path to test tflite model file."),
  };

  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << "Required test_model_file\n";
    std::abort();
  }
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}

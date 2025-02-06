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
#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace {
struct batch_seq_config_test_flags {
  std::string* model_file_path;
  std::string* model_input_shapes;
  int max_batch_size = -1;
  float error_epsilon = 0.2;
} batch_seq_config_test_flags_values;

// Returns a randomly generated data of size 'num_elements'.
std::vector<uint8_t> GetData(int num_elements) {
  std::vector<uint8_t> result(num_elements);
  std::random_device random_engine;
  std::uniform_int_distribution<uint32_t> distribution(0, 254);
  std::generate_n(result.data(), num_elements, [&]() {
    return static_cast<uint8_t>(distribution(random_engine));
  });
  return result;
}

// Returns the total number of elements.
int NumElements(const std::vector<int>& shape) {
  int num_elements = 1;
  for (int dim : shape) num_elements *= dim;
  return num_elements;
}

// Returns true if 'control' and 'exp' values match up to 'epsilon'
bool DiffOutput(const std::vector<float>& control,
                const std::vector<float>& exp, double epsilon) {
  if (control.size() != exp.size()) {
    TFLITE_LOG(ERROR) << "Mismatch size Expected" << control.size() << " got "
                      << exp.size();
    return false;
  }
  bool has_diff = false;
  for (int i = 0; i < control.size(); ++i) {
    if (abs(control[i] - exp[i]) > epsilon) {
      TFLITE_LOG(ERROR) << control[i] << " " << exp[i];
      has_diff = true;
    }
  }
  return !has_diff;
}

bool DiffOutput(const std::vector<float>& control,
                const std::vector<float>& exp) {
  return DiffOutput(control, exp,
                    tflite::batch_seq_config_test_flags_values.error_epsilon);
}
}  // namespace

class TestModel {
 public:
  TestModel() : delegate_(nullptr, [](TfLiteDelegate* delegate) {}) {}

  // Initialize the model by reading the model from file and build
  // interpreter.
  void Init() {
    model_ = tflite::FlatBufferModel::BuildFromFile(
        tflite::batch_seq_config_test_flags_values.model_file_path->c_str());
    ASSERT_TRUE(model_ != nullptr);

    resolver_ = std::make_unique<ops::builtin::BuiltinOpResolver>();
    InterpreterBuilder(*model_, *resolver_)(&interpreter_);
    ASSERT_TRUE(interpreter_ != nullptr);
  }

  // Add Hexagon delegate to the graph.
  void ApplyDelegate(int max_batch_size,
                     const std::vector<int>& input_batch_dimensions,
                     const std::vector<int>& output_batch_dimensions) {
    TfLiteIntArray* input_batch_dim =
        TfLiteIntArrayCreate(input_batch_dimensions.size());
    TfLiteIntArray* output_batch_dim =
        TfLiteIntArrayCreate(output_batch_dimensions.size());
    for (int i = 0; i < input_batch_dimensions.size(); ++i)
      input_batch_dim->data[i] = input_batch_dimensions[i];
    for (int i = 0; i < output_batch_dimensions.size(); ++i)
      output_batch_dim->data[i] = output_batch_dimensions[i];
    ::TfLiteHexagonDelegateOptions options = {0};
    options.enable_dynamic_batch_size = true;
    options.max_batch_size = max_batch_size;
    options.input_batch_dimensions = input_batch_dim;
    options.output_batch_dimensions = output_batch_dim;
    TfLiteDelegate* delegate = TfLiteHexagonDelegateCreate(&options);
    ASSERT_TRUE(delegate != nullptr);
    delegate_ = std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
        delegate, [](TfLiteDelegate* delegate) {
          TfLiteHexagonDelegateDelete(delegate);
        });
    ASSERT_TRUE(interpreter_->ModifyGraphWithDelegate(delegate_.get()) ==
                kTfLiteOk);
  }

  void Run(const std::vector<int>& input_shape,
           const std::vector<uint8_t>& input_data) {
    // Resize Inputs.
    auto interpreter_inputs = interpreter_->inputs();
    interpreter_->ResizeInputTensor(interpreter_inputs[0], input_shape);
    ASSERT_EQ(kTfLiteOk, interpreter_->AllocateTensors());

    TfLiteTensor* input_tensor =
        interpreter_->tensor(interpreter_->inputs()[0]);
    memcpy(input_tensor->data.raw, input_data.data(),
           input_data.size() * sizeof(uint8_t));

    ASSERT_EQ(kTfLiteOk, interpreter_->Invoke());
  }

  std::vector<float> GetOutput(int output_index) {
    auto* tensor = interpreter_->output_tensor(output_index);
    uint8_t* data = interpreter_->typed_output_tensor<uint8_t>(output_index);
    std::vector<float> result;
    result.resize(NumElements(tensor));
    const auto scale =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params)
            ->scale->data[0];
    const auto zero_point =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params)
            ->zero_point->data[0];
    for (int i = 0; i < result.size(); ++i) {
      result[i] = scale * (data[i] - zero_point);
    }
    return result;
  }

 private:
  std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> delegate_;
  std::unique_ptr<FlatBufferModel> model_;
  std::unique_ptr<tflite::OpResolver> resolver_;
  std::unique_ptr<Interpreter> interpreter_;
};

std::vector<std::vector<int>> ParseInputShapes() {
  std::vector<string> str_input_shapes;
  benchmark::util::SplitAndParse(
      *tflite::batch_seq_config_test_flags_values.model_input_shapes, ':',
      &str_input_shapes);
  std::vector<std::vector<int>> input_shapes(str_input_shapes.size());
  for (int i = 0; i < str_input_shapes.size(); ++i) {
    benchmark::util::SplitAndParse(str_input_shapes[i], ',', &input_shapes[i]);
  }
  return input_shapes;
}

TEST(HexagonDynamicBatch, MultipleResizes) {
  int num_failed_tests = 0;
  int num_test = 0;
  auto test_input_shapes = ParseInputShapes();
  auto default_model = std::make_unique<TestModel>();
  auto delegated_model = std::make_unique<TestModel>();
  TFLITE_LOG(INFO)
      << "model_file_path: "
      << *tflite::batch_seq_config_test_flags_values.model_file_path << "\n";
  TFLITE_LOG(INFO)
      << "model_input_shapes: "
      << *tflite::batch_seq_config_test_flags_values.model_input_shapes << "\n";
  TFLITE_LOG(INFO) << "max_batch_size: "
                   << tflite::batch_seq_config_test_flags_values.max_batch_size
                   << "\n";
  TFLITE_LOG(INFO) << "error_epsilon: "
                   << tflite::batch_seq_config_test_flags_values.error_epsilon
                   << "\n";
  default_model->Init();
  delegated_model->Init();
  delegated_model->ApplyDelegate(
      tflite::batch_seq_config_test_flags_values.max_batch_size, {0}, {0});
  for (const auto& input_shape : test_input_shapes) {
    const auto input = GetData(NumElements(input_shape));
    default_model->Run(input_shape, input);
    delegated_model->Run(input_shape, input);
    const auto default_output = default_model->GetOutput(0);
    const auto delegated_output = delegated_model->GetOutput(0);
    if (!DiffOutput(default_output, delegated_output)) {
      TFLITE_LOG(ERROR) << "Failed for input " << num_test;
      num_failed_tests++;
    }
    num_test++;
  }
  if (num_failed_tests == 0) {
    TFLITE_LOG(INFO) << "All Tests PASSED";
  } else {
    TFLITE_LOG(INFO) << "Failed " << num_failed_tests << " out of " << num_test;
  }
}
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  std::string FLAGS_model_file_path;
  std::string FLAGS_model_input_shapes;
  int FLAGS_max_batch_size = -1;
  float FLAGS_error_epsilon = 0.2;
  std::vector<tensorflow::Flag> flags = {
      tensorflow::Flag("model_file_path", &FLAGS_model_file_path,
                       "Path to the test model file."),
      tensorflow::Flag(
          "model_input_shapes", &FLAGS_model_input_shapes,
          "List of different input shapes for testing, the input will "
          "resized for each one in order and tested. They Should be "
          "separated by : and each shape has dimensions separated by ,"),
      tensorflow::Flag("max_batch_size", &FLAGS_max_batch_size,
                       "Maximum batch size for a single run by hexagon."),
      tensorflow::Flag("error_epsilon", &FLAGS_error_epsilon,
                       "Maximum error allowed while diffing the output."),
  };
  bool no_inputs = argc == 1;
  bool success = tensorflow::Flags::Parse(&argc, argv, flags);
  if (!success || no_inputs || (argc == 2 && !strcmp(argv[1], "--helpfull"))) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  } else if (FLAGS_model_file_path.empty() ||
             FLAGS_model_input_shapes.empty()) {
    fprintf(stderr, "%s", tensorflow::Flags::Usage(argv[0], flags).c_str());
    return {};
  }
  tflite::batch_seq_config_test_flags_values.model_file_path =
      &FLAGS_model_file_path;
  tflite::batch_seq_config_test_flags_values.max_batch_size =
      FLAGS_max_batch_size;
  tflite::batch_seq_config_test_flags_values.model_input_shapes =
      &FLAGS_model_input_shapes;
  tflite::batch_seq_config_test_flags_values.error_epsilon =
      FLAGS_error_epsilon;
  testing::InitGoogleTest();

  TfLiteHexagonInit();
  int return_val = RUN_ALL_TESTS();
  TfLiteHexagonTearDown();
  return return_val;
}

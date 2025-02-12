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
// Parses tflite example input data.
// Format is ASCII
// TODO(aselle): Switch to protobuf, but the android team requested a simple
// ASCII file.
#include "tensorflow/lite/testing/parse_testdata.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/testing/message.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/test_runner.h"

namespace tflite {
namespace testing {
namespace {

const char kDefaultSignatureKey[] = "serving_default";

// Fatal error if parse error occurs
#define PARSE_CHECK_EQ(filename, current_line, x, y)                         \
  if ((x) != (y)) {                                                          \
    fprintf(stderr, "Parse Error @ %s:%d\n  File %s\n  Line %d, %s != %s\n", \
            __FILE__, __LINE__, filename, current_line + 1, #x, #y);         \
    return kTfLiteError;                                                     \
  }

// Breakup a "," delimited line into a std::vector<std::string>.
// This is extremely inefficient, and just used for testing code.
// TODO(aselle): replace with absl when we use it.
std::vector<std::string> ParseLine(const std::string& line) {
  size_t pos = 0;
  std::vector<std::string> elements;
  while (true) {
    size_t end = line.find(',', pos);
    if (end == std::string::npos) {
      elements.push_back(line.substr(pos));
      break;
    } else {
      elements.push_back(line.substr(pos, end - pos));
    }
    pos = end + 1;
  }
  return elements;
}

}  // namespace

// Given a `filename`, produce a vector of Examples corresponding
// to test cases that can be applied to a tflite model.
TfLiteStatus ParseExamples(const char* filename,
                           std::vector<Example>* examples) {
  std::ifstream fp(filename);
  if (!fp.good()) {
    fprintf(stderr, "Could not read '%s'\n", filename);
    return kTfLiteError;
  }
  std::string str((std::istreambuf_iterator<char>(fp)),
                  std::istreambuf_iterator<char>());
  size_t pos = 0;

  // \n and , delimit parse a file.
  std::vector<std::vector<std::string>> csv;
  while (true) {
    size_t end = str.find('\n', pos);

    if (end == std::string::npos) {
      csv.emplace_back(ParseLine(str.substr(pos)));
      break;
    }
    csv.emplace_back(ParseLine(str.substr(pos, end - pos)));
    pos = end + 1;
  }

  int current_line = 0;
  PARSE_CHECK_EQ(filename, current_line, csv[0][0], "test_cases");
  int example_count = std::stoi(csv[0][1]);
  current_line++;

  auto parse_tensor = [&filename, &current_line,
                       &csv](FloatTensor* tensor_ptr) {
    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "dtype");
    current_line++;
    // parse shape
    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "shape");
    size_t elements = 1;
    FloatTensor& tensor = *tensor_ptr;

    for (size_t i = 1; i < csv[current_line].size(); i++) {
      const auto& shape_part_to_parse = csv[current_line][i];
      if (shape_part_to_parse.empty()) {
        // Case of a 0-dimensional shape
        break;
      }
      int shape_part = std::stoi(shape_part_to_parse);
      elements *= shape_part;
      tensor.shape.push_back(shape_part);
    }
    current_line++;
    // parse data
    PARSE_CHECK_EQ(filename, current_line, csv[current_line].size() - 1,
                   elements);
    for (size_t i = 1; i < csv[current_line].size(); i++) {
      tensor.flat_data.push_back(std::stof(csv[current_line][i]));
    }
    current_line++;

    return kTfLiteOk;
  };

  for (int example_idx = 0; example_idx < example_count; example_idx++) {
    Example example;
    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "inputs");
    int inputs = std::stoi(csv[current_line][1]);
    current_line++;
    // parse dtype
    for (int input_index = 0; input_index < inputs; input_index++) {
      example.inputs.push_back(FloatTensor());
      TF_LITE_ENSURE_STATUS(parse_tensor(&example.inputs.back()));
    }

    PARSE_CHECK_EQ(filename, current_line, csv[current_line][0], "outputs");
    int outputs = std::stoi(csv[current_line][1]);
    current_line++;
    for (int input_index = 0; input_index < outputs; input_index++) {
      example.outputs.push_back(FloatTensor());
      TF_LITE_ENSURE_STATUS(parse_tensor(&example.outputs.back()));
    }
    examples->emplace_back(example);
  }
  return kTfLiteOk;
}

TfLiteStatus FeedExample(tflite::Interpreter* interpreter,
                         const Example& example) {
  // Resize inputs to match example & allocate.
  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_index = interpreter->inputs()[i];

    TF_LITE_ENSURE_STATUS(
        interpreter->ResizeInputTensor(input_index, example.inputs[i].shape));
  }
  TF_LITE_ENSURE_STATUS(interpreter->AllocateTensors());
  // Copy data into tensors.
  for (size_t i = 0; i < interpreter->inputs().size(); i++) {
    int input_index = interpreter->inputs()[i];
    if (float* data = interpreter->typed_tensor<float>(input_index)) {
      for (size_t idx = 0; idx < example.inputs[i].flat_data.size(); idx++) {
        data[idx] = example.inputs[i].flat_data[idx];
      }
    } else if (int32_t* data =
                   interpreter->typed_tensor<int32_t>(input_index)) {
      for (size_t idx = 0; idx < example.inputs[i].flat_data.size(); idx++) {
        data[idx] = example.inputs[i].flat_data[idx];
      }
    } else if (int64_t* data =
                   interpreter->typed_tensor<int64_t>(input_index)) {
      for (size_t idx = 0; idx < example.inputs[i].flat_data.size(); idx++) {
        data[idx] = example.inputs[i].flat_data[idx];
      }
    } else {
      fprintf(stderr, "input[%zu] was not float or int data\n", i);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CheckOutputs(tflite::Interpreter* interpreter,
                          const Example& example) {
  constexpr double kRelativeThreshold = 1e-2f;
  constexpr double kAbsoluteThreshold = 1e-4f;

  ErrorReporter* context = DefaultErrorReporter();
  int model_outputs = interpreter->outputs().size();
  TF_LITE_ENSURE_EQ(context, model_outputs, example.outputs.size());
  for (size_t i = 0; i < interpreter->outputs().size(); i++) {
    bool tensors_differ = false;
    int output_index = interpreter->outputs()[i];
    if (const float* data = interpreter->typed_tensor<float>(output_index)) {
      for (size_t idx = 0; idx < example.outputs[i].flat_data.size(); idx++) {
        float computed = data[idx];
        float reference = example.outputs[0].flat_data[idx];
        float diff = std::abs(computed - reference);
        // For very small numbers, try absolute error, otherwise go with
        // relative.
        bool local_tensors_differ =
            std::abs(reference) < kRelativeThreshold
                ? diff > kAbsoluteThreshold
                : diff > kRelativeThreshold * std::abs(reference);
        if (local_tensors_differ) {
          fprintf(stdout, "output[%zu][%zu] did not match %f vs reference %f\n",
                  i, idx, data[idx], reference);
          tensors_differ = local_tensors_differ;
        }
      }
    } else if (const int32_t* data =
                   interpreter->typed_tensor<int32_t>(output_index)) {
      for (size_t idx = 0; idx < example.outputs[i].flat_data.size(); idx++) {
        int32_t computed = data[idx];
        int32_t reference = example.outputs[0].flat_data[idx];
        if (std::abs(computed - reference) > 0) {
          fprintf(stderr, "output[%zu][%zu] did not match %d vs reference %d\n",
                  i, idx, computed, reference);
          tensors_differ = true;
        }
      }
    } else if (const int64_t* data =
                   interpreter->typed_tensor<int64_t>(output_index)) {
      for (size_t idx = 0; idx < example.outputs[i].flat_data.size(); idx++) {
        int64_t computed = data[idx];
        int64_t reference = example.outputs[0].flat_data[idx];
        if (std::abs(computed - reference) > 0) {
          fprintf(stderr,
                  "output[%zu][%zu] did not match %" PRId64
                  " vs reference %" PRId64 "\n",
                  i, idx, computed, reference);
          tensors_differ = true;
        }
      }
    } else {
      fprintf(stderr, "output[%zu] was not float or int data\n", i);
      return kTfLiteError;
    }
    fprintf(stderr, "\n");
    if (tensors_differ) return kTfLiteError;
  }
  return kTfLiteOk;
}

// Processes Protobuf map<string, string> like message.
// Supports format of
// field_name {key: "KEY1" value: "VAL1"}
// field_name {key: "KEY2" value: "VAL2"}
// field_name {key: "KEY3" value: "VAL3"}
//
// for field `map<string, string> field_name = TAG;`
//
// Note: The parent of this field should track the ownership of the repeated
// field. By calling KvMap::Finish() means a new entry is added to the map
// instead of finish parsing of the whole map.
class KvMap : public Message, public std::vector<std::pair<string, string>> {
 public:
  void SetField(const std::string& name, const std::string& value) override {
    if (name == "key") {
      key_ = value;
    } else if (name == "value") {
      value_ = value;
    }
  }
  void Finish() override {
    push_back(std::make_pair(key_, value_));
    key_.clear();
    value_.clear();
  }

 private:
  string key_;
  string value_;
};

// Processes an 'invoke' message, triggering execution of the test runner, as
// well as verification of outputs. An 'invoke' message looks like:
//   invoke {
//     id: "xyz"
//     input { key: "a" value: "1,2,1,1,1,2,3,4"}
//     input { key: "b" value: "1,2,1,1,1,2,3,4"}
//     output { key: "x" value: "4,5,6"}
//     output { key: "y" value: "14,15,16"}
//     output_shape { key: "x" value: "3"}
//     output_shape { key: "y" value: "1,3"}
//   }
class Invoke : public Message {
 public:
  explicit Invoke(TestRunner* test_runner) : test_runner_(test_runner) {}

  void SetField(const std::string& name, const std::string& value) override {
    if (name == "id") {
      test_runner_->SetInvocationId(value);
    }
  }

  Message* AddChild(const std::string& s) override {
    if (s == "input") {
      return MaybeInitializeChild(&inputs_);
    } else if (s == "output") {
      return MaybeInitializeChild(&expected_outputs_);
    } else if (s == "output_shape") {
      return MaybeInitializeChild(&expected_output_shapes_);
    }
    return nullptr;
  }

  // Invokes the test runner and checks expectations.
  void Finish() override {
    using VectorT = std::vector<std::pair<string, string>>;
    test_runner_->Invoke(inputs_ ? *inputs_ : VectorT());
    test_runner_->CheckResults(
        expected_outputs_ ? *expected_outputs_ : VectorT(),
        expected_output_shapes_ ? *expected_output_shapes_ : VectorT());
  }

 private:
  // Checks whether `*child` is initialized and return the message pointer.
  // Initializes and owns it if it's not initialized.
  Message* MaybeInitializeChild(KvMap** child) {
    if (*child == nullptr) {
      *child = new KvMap;
      Store(*child);
    }
    return *child;
  }

  TestRunner* test_runner_;

  KvMap* inputs_ = nullptr;
  KvMap* expected_outputs_ = nullptr;
  KvMap* expected_output_shapes_ = nullptr;
};

// Process an 'reshape' message, triggering resizing of the input tensors via
// the test runner. A 'reshape' message looks like:
//   reshape {
//     input { key: "a" value: "1,2,1,1,1,2,3,4"}
//     input { key: "b" value: "1,2,1,1,1,2,3,4"}
//   }
class Reshape : public Message {
 public:
  explicit Reshape(TestRunner* test_runner) : test_runner_(test_runner) {}

  Message* AddChild(const std::string& s) override {
    if (s != "input") return nullptr;
    if (input_shapes_ == nullptr) {
      input_shapes_ = new KvMap;
      Store(input_shapes_);
    }
    return input_shapes_;
  }

  // Reshapes tensors.
  void Finish() override {
    if (!input_shapes_) return;
    for (const auto& item : *input_shapes_) {
      test_runner_->ReshapeTensor(item.first, item.second);
    }
  }

 private:
  TestRunner* test_runner_;

  KvMap* input_shapes_ = nullptr;
};

// This is the top-level message in a test file.
class TestData : public Message {
 public:
  explicit TestData(TestRunner* test_runner)
      : test_runner_(test_runner), num_invocations_(0), max_invocations_(-1) {}
  void SetMaxInvocations(int max) { max_invocations_ = max; }
  void SetField(const std::string& name, const std::string& value) override {
    if (name == "load_model") {
      test_runner_->LoadModel(value, kDefaultSignatureKey);
    } else if (name == "init_state") {
      test_runner_->AllocateTensors();
      for (const auto& name : Split<string>(value, ",")) {
        test_runner_->ResetTensor(name);
      }
    }
  }
  Message* AddChild(const std::string& s) override {
    if (s == "invoke") {
      test_runner_->AllocateTensors();
      if (max_invocations_ == -1 || num_invocations_ < max_invocations_) {
        ++num_invocations_;
        return Store(new Invoke(test_runner_));
      } else {
        return nullptr;
      }
    } else if (s == "reshape") {
      return Store(new Reshape(test_runner_));
    }
    return nullptr;
  }

 private:
  TestRunner* test_runner_;
  int num_invocations_;
  int max_invocations_;
};

bool ParseAndRunTests(std::istream* input, TestRunner* test_runner,
                      int max_invocations) {
  TestData test_data(test_runner);
  test_data.SetMaxInvocations(max_invocations);
  Message::Read(input, &test_data);
  return test_runner->IsValid() && test_runner->GetOverallSuccess();
}

}  // namespace testing
}  // namespace tflite

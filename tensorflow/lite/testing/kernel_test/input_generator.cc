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
#include "tensorflow/lite/testing/kernel_test/input_generator.h"

#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/testing/join.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

namespace {
static constexpr char kDefaultServingSignatureDefKey[] = "serving_default";

template <typename T>
std::vector<T> GenerateRandomTensor(TfLiteIntArray* dims,
                                    const std::function<T(int)>& random_func) {
  int64_t num_elements = 1;
  for (int i = 0; i < dims->size; i++) {
    num_elements *= dims->data[i];
  }

  std::vector<T> result(num_elements);
  for (int i = 0; i < num_elements; i++) {
    result[i] = random_func(i);
  }
  return result;
}

template <typename T>
std::vector<T> GenerateUniform(TfLiteIntArray* dims, float min, float max) {
  auto random_float = [](float min, float max) {
    // TODO(yunluli): Change seed for each invocation if needed.
    // Used rand() instead of rand_r() here to make it runnable on android.
    return min + (max - min) * static_cast<float>(rand()) / RAND_MAX;
  };

  std::function<T(int)> random_t = [&](int) {
    return static_cast<T>(random_float(min, max));
  };
  std::vector<T> data = GenerateRandomTensor(dims, random_t);
  return data;
}

template <typename T>
std::vector<T> GenerateGaussian(TfLiteIntArray* dims, float min, float max) {
  auto random_float = [](float min, float max) {
    static std::default_random_engine generator;
    // We generate a float number within [0, 1) following a mormal distribution
    // with mean = 0.5 and stddev = 1/3, and use it to scale the final random
    // number into the desired range.
    static std::normal_distribution<double> distribution(0.5, 1.0 / 3);
    auto rand_n = distribution(generator);
    while (rand_n < 0 || rand_n >= 1) {
      rand_n = distribution(generator);
    }

    return min + (max - min) * static_cast<float>(rand_n);
  };

  std::function<T(int)> random_t = [&](int) {
    return static_cast<T>(random_float(min, max));
  };
  std::vector<T> data = GenerateRandomTensor(dims, random_t);
  return data;
}

}  // namespace

TfLiteStatus InputGenerator::LoadModel(const string& model_dir) {
  return LoadModel(model_dir, kDefaultServingSignatureDefKey);
}

TfLiteStatus InputGenerator::LoadModel(const string& model_dir,
                                       const string& signature) {
  model_ = FlatBufferModel::BuildFromFile(model_dir.c_str());
  if (!model_) {
    fprintf(stderr, "Cannot load model %s", model_dir.c_str());
    return kTfLiteError;
  }

  ::tflite::ops::builtin::BuiltinOpResolver builtin_ops;
  InterpreterBuilder(*model_, builtin_ops)(&interpreter_);
  if (!interpreter_) {
    fprintf(stderr, "Failed to build interpreter.");
    return kTfLiteError;
  }
  signature_runner_ = interpreter_->GetSignatureRunner(signature.c_str());
  if (!signature_runner_) {
    fprintf(stderr, "Failed to get SignatureRunner.\n");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus InputGenerator::ReadInputsFromFile(const string& filename) {
  if (filename.empty()) {
    fprintf(stderr, "Empty input file name.");
    return kTfLiteError;
  }

  std::ifstream input_file(filename);
  string input;
  while (std::getline(input_file, input, '\n')) {
    std::vector<string> parts = Split<string>(input, ":");
    if (parts.size() != 2) {
      fprintf(stderr, "Expected <name>:<value>, got %s", input.c_str());
      return kTfLiteError;
    }
    inputs_.push_back(std::make_pair(parts[0], parts[1]));
  }
  input_file.close();
  return kTfLiteOk;
}

TfLiteStatus InputGenerator::WriteInputsToFile(const string& filename) {
  if (filename.empty()) {
    fprintf(stderr, "Empty input file name.");
    return kTfLiteError;
  }

  std::ofstream output_file;
  output_file.open(filename, std::fstream::out | std::fstream::trunc);
  if (!output_file) {
    fprintf(stderr, "Failed to open output file %s.", filename.c_str());
    return kTfLiteError;
  }

  for (const auto& input : inputs_) {
    output_file << input.first << ":" << input.second << "\n";
  }
  output_file.close();

  return kTfLiteOk;
}

// TODO(yunluli): Support more tensor types when needed.
TfLiteStatus InputGenerator::GenerateInput(const string& distribution) {
  auto input_tensor_names = signature_runner_->input_names();
  for (const char* name : input_tensor_names) {
    auto* tensor = signature_runner_->input_tensor(name);
    if (distribution == "UNIFORM") {
      switch (tensor->type) {
        case kTfLiteInt8: {
          auto data = GenerateUniform<int8_t>(
              tensor->dims, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteUInt8: {
          auto data = GenerateUniform<uint8_t>(
              tensor->dims, std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteFloat32: {
          auto data = GenerateUniform<float>(tensor->dims, -1, 1);
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        default:
          fprintf(stderr, "Unsupported input tensor type %s.",
                  TfLiteTypeGetName(tensor->type));
          break;
      }
    } else if (distribution == "GAUSSIAN") {
      switch (tensor->type) {
        case kTfLiteInt8: {
          auto data = GenerateGaussian<int8_t>(
              tensor->dims, std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteUInt8: {
          auto data = GenerateGaussian<uint8_t>(
              tensor->dims, std::numeric_limits<uint8_t>::min(),
              std::numeric_limits<uint8_t>::max());
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        case kTfLiteFloat32: {
          auto data = GenerateGaussian<float>(tensor->dims, -1, 1);
          inputs_.push_back(
              std::make_pair(name, Join(data.data(), data.size(), ",")));
          break;
        }
        default:
          fprintf(stderr, "Unsupported input tensor type %s.",
                  TfLiteTypeGetName(tensor->type));
          break;
      }
    } else {
      fprintf(stderr, "Unsupported distribution %s.", distribution.c_str());
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace testing
}  // namespace tflite

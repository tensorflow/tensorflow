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
#ifndef TENSORFLOW_LITE_TESTING_KERNEL_TEST_UTIL_H_
#define TENSORFLOW_LITE_TESTING_KERNEL_TEST_UTIL_H_

#include <fstream>

#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/testing/kernel_test/input_generator.h"
#include "tensorflow/lite/testing/split.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace kernel_test {

struct TestOptions {
  // Path of tensorflow lite model.
  string tflite_model;
  // Path of the input file. If empty, generate at runtime.
  string read_input_from_file;
  // Path to dump the input file.
  string dump_input_to_file;
  // Path to dump the output.
  string dump_output_to_file;
  // Input distribution.
  string input_distribution;
  // Kernel type.
  string kernel_type;
};

inline TestOptions ParseTfliteKernelTestFlags(int* argc, char** argv) {
  TestOptions options;
  std::vector<tensorflow::Flag> flags = {
      tensorflow::Flag("tflite_model", &options.tflite_model,
                       "Path of tensorflow lite model."),
      tensorflow::Flag("read_input_from_file", &options.read_input_from_file,
                       "File to read input data from. If empty, generates "
                       "input at runtime."),
      tensorflow::Flag("dump_input_to_file", &options.dump_input_to_file,
                       "File to dump randomly generated input."),
      tensorflow::Flag("dump_output_to_file", &options.dump_output_to_file,
                       "File to dump output."),
      tensorflow::Flag("input_distribution", &options.input_distribution,
                       "Input distribution. Default: Gaussian."),
      tensorflow::Flag("kernel_type", &options.kernel_type, "Kernel type."),
  };

  tensorflow::Flags::Parse(argc, argv, flags);

  return options;
}

inline TfLiteStatus RunKernelTest(const kernel_test::TestOptions& options,
                                  TestRunner* runner) {
  InputGenerator input_generator;

  if (options.read_input_from_file.empty()) {
    TF_LITE_ENSURE_STATUS(input_generator.LoadModel(options.tflite_model));
    TF_LITE_ENSURE_STATUS(
        input_generator.GenerateInput(options.input_distribution));
  } else {
    TF_LITE_ENSURE_STATUS(
        input_generator.ReadInputsFromFile(options.read_input_from_file));
  }

  runner->LoadModel(options.tflite_model);
  runner->AllocateTensors();
  if (!runner->IsValid()) return kTfLiteError;
  auto inputs = input_generator.GetInputs();

  runner->Invoke(inputs);

  if (!options.dump_input_to_file.empty()) {
    TF_LITE_ENSURE_STATUS(
        input_generator.WriteInputsToFile(options.dump_input_to_file));
  }

  if (!options.dump_output_to_file.empty()) {
    std::ofstream output_file;
    output_file.open(options.dump_output_to_file,
                     std::fstream::out | std::fstream::trunc);
    if (!output_file) {
      return kTfLiteError;
    }

    for (const auto& name : runner->GetOutputNames()) {
      output_file << name << ":" << runner->ReadOutput(name) << "\n";
    }
    output_file.close();
  }

  return kTfLiteOk;
}

}  // namespace kernel_test
}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_KERNEL_TEST_UTIL_H_

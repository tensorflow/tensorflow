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
#ifndef TENSORFLOW_LITE_TESTING_TFLITE_DIFF_UTIL_H_
#define TENSORFLOW_LITE_TESTING_TFLITE_DIFF_UTIL_H_

#include <vector>

#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {

// Configurations to run Tflite diff test.
struct DiffOptions {
  // Path of tensorflow model.
  string tensorflow_model;
  // Path of tensorflow lite model.
  string tflite_model;
  // Names of input tensors.
  // Example: input_1,input_2
  std::vector<string> input_layer;
  // Data types of input tensors.
  // Example: float,int
  std::vector<string> input_layer_type;
  // Shapes of input tensors, separated by comma.
  // Example: 1,3,4,1
  std::vector<string> input_layer_shape;
  // Names of output tensors.
  // Example output_1,output_2
  std::vector<string> output_layer;
  // Number of full runs (from building interpreter to checking outputs) in
  // each of the passes. The first pass has a single inference, while the
  // second pass does multiple inferences back to back.
  int num_runs_per_pass;
  // The type of delegate to apply during inference.
  TfLiteDriver::DelegateType delegate;
  // Path of tflite model used to generate golden values.
  std::string reference_tflite_model = "";
};

// Run a single TensorFLow Lite diff test with a given options.
bool RunDiffTest(const DiffOptions& options, int num_invocations);

// Runs diff test for custom TestRunner identified by the factory methiodd
// 'runner_factory' against TFLite CPU given 'options' 'runner_factory' should
// return instance of TestRunner, caller will take ownership of the returned
// object.
// Function returns True if test pass, false otherwise.
bool RunDiffTestWithProvidedRunner(const tflite::testing::DiffOptions& options,
                                   TestRunner* (*runner_factory)());

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TFLITE_DIFF_UTIL_H_

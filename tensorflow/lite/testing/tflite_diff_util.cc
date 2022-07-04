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
#include "tensorflow/lite/testing/tflite_diff_util.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#include "tensorflow/lite/testing/generate_testspec.h"
#include "tensorflow/lite/testing/parse_testdata.h"
#include "tensorflow/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {
namespace {
bool SingleRunDiffTestWithProvidedRunner(::tflite::testing::DiffOptions options,
                                         int num_invocations,
                                         TestRunner* (*runner_factory)()) {
  std::stringstream tflite_stream;
  std::string reference_tflite_model = options.reference_tflite_model.empty()
                                           ? options.tflite_model
                                           : options.reference_tflite_model;
  if (!GenerateTestSpecFromTFLiteModel(
          tflite_stream, reference_tflite_model, num_invocations,
          options.input_layer, options.input_layer_type,
          options.input_layer_shape, options.output_layer)) {
    return false;
  }

  std::unique_ptr<TestRunner> runner(runner_factory());
  runner->LoadModel(options.tflite_model);
  return ParseAndRunTests(&tflite_stream, runner.get());
}
}  // namespace

bool RunDiffTest(const DiffOptions& options, int num_invocations) {
  std::stringstream tflite_stream;
  if (!GenerateTestSpecFromTensorflowModel(
          tflite_stream, options.tensorflow_model, options.tflite_model,
          num_invocations, options.input_layer, options.input_layer_type,
          options.input_layer_shape, options.output_layer)) {
    return false;
  }
  TfLiteDriver tflite_driver(options.delegate);
  tflite_driver.LoadModel(options.tflite_model);
  return ParseAndRunTests(&tflite_stream, &tflite_driver);
}

bool RunDiffTestWithProvidedRunner(const tflite::testing::DiffOptions& options,
                                   TestRunner* (*runner_factory)()) {
  int failure_count = 0;
  for (int i = 0; i < options.num_runs_per_pass; i++) {
    if (!SingleRunDiffTestWithProvidedRunner(options,
                                             /*num_invocations=*/1,
                                             runner_factory)) {
      ++failure_count;
    }
  }
  int failures_in_first_pass = failure_count;

  if (failure_count == 0) {
    // Let's try again with num_invocations > 1 to make sure we can do multiple
    // invocations without resetting the interpreter.
    for (int i = 0; i < options.num_runs_per_pass; i++) {
      if (!SingleRunDiffTestWithProvidedRunner(options,
                                               /*num_invocations=*/2,
                                               runner_factory)) {
        ++failure_count;
      }
    }
  }

  fprintf(stderr, "Num errors in single-inference pass: %d\n",
          failures_in_first_pass);
  fprintf(stderr, "Num errors in multi-inference pass : %d\n",
          failure_count - failures_in_first_pass);

  return failure_count == 0;
}
}  // namespace testing

}  // namespace tflite

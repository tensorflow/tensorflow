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
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "tensorflow/contrib/lite/testing/generate_testspec.h"
#include "tensorflow/contrib/lite/testing/parse_testdata.h"
#include "tensorflow/contrib/lite/testing/tflite_diff_util.h"
#include "tensorflow/contrib/lite/testing/tflite_driver.h"

namespace tflite {
namespace testing {

bool RunDiffTest(const DiffOptions& options, int num_invocations) {
  std::stringstream tflite_stream;
  if (!GenerateTestSpecFromTensorflowModel(
          tflite_stream, options.tensorflow_model, options.tflite_model,
          num_invocations, options.input_layer, options.input_layer_type,
          options.input_layer_shape, options.output_layer)) {
    return false;
  }
  TfLiteDriver tflite_driver(/*use_nnapi=*/true);
  tflite_driver.LoadModel(options.tflite_model);
  return tflite::testing::ParseAndRunTests(&tflite_stream, &tflite_driver);
}
}  // namespace testing

}  // namespace tflite
